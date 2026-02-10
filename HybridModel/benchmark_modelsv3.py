import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
import time
import random
import gc
import os
import math

# ==========================================
# 0. 动态导入 (兼容性检查)
# ==========================================
try:
    from fla.layers import GatedLinearAttention
    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    print("⚠️ Warning: 'flash-linear-attention' not installed. Skipping FLA model.")

try:
    from mamba_ssm import Mamba2
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("⚠️ Warning: 'mamba-ssm' not installed. Skipping Mamba model.")

# ==========================================
# 1. 全局配置
# ==========================================
CONFIG = {
    "vocab_size": 50257,    # GPT-2 Tokenizer
    "d_model": 512,         # 统一维度
    "n_layers": 4,          # 统一层数
    "n_heads": 8,           # 统一头数 (Mamba 会自动适配)
    "train_seq_len": 2048,  # 训练时长度
    "test_seq_len": 4096,   # 测试/外推长度
    "batch_size": 8,        # H100 上可以开大
    "steps": 200,           # 训练步数 (足够看到 Induction 出现)
    "device": "cuda",
    "dtype": torch.bfloat16, # H100 标配
    "dataset": "flytech/python-codes-25k"
}

print(f"🚀 Ultimate Benchmark | Device: {CONFIG['device']} | Dtype: {CONFIG['dtype']}")

# ==========================================
# 2. 模型定义 (四种架构)
# ==========================================

# --- A. RNN (LSTM) ---
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model'])
        self.lstm = nn.LSTM(CONFIG['d_model'], CONFIG['d_model'], CONFIG['n_layers'], batch_first=True)
        self.head = nn.Linear(CONFIG['d_model'], CONFIG['vocab_size'])
    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        return self.head(x)

# --- B. Transformer (O(N^2)) ---
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model'])
        # 简单位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, CONFIG['test_seq_len'] * 2, CONFIG['d_model']))
        layer = nn.TransformerEncoderLayer(CONFIG['d_model'], CONFIG['n_heads'], batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, CONFIG['n_layers'])
        self.head = nn.Linear(CONFIG['d_model'], CONFIG['vocab_size'])
    def forward(self, x):
        B, T = x.size()
        if T > self.pos_embed.size(1):
            # 简单的截断/扩展处理，防止 crash
            x = self.embed(x)
        else:
            x = self.embed(x) + self.pos_embed[:, :T, :]
            
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        return self.head(x)

# --- C. Hybrid FLA (Linear Attention) ---
class FLAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model'])
        self.layers = nn.ModuleList([GLABlock(CONFIG['d_model'], CONFIG['n_heads']) for _ in range(CONFIG['n_layers'])])
        self.norm = nn.LayerNorm(CONFIG['d_model'])
        self.head = nn.Linear(CONFIG['d_model'], CONFIG['vocab_size'])
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

class GLABlock(nn.Module):
    def __init__(self, h, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(h)
        # 显式参数防止报错
        self.attn = GatedLinearAttention(hidden_size=h, num_heads=heads, mode='chunk')
        self.norm2 = nn.LayerNorm(h)
        self.mlp = nn.Sequential(nn.Linear(h, h*4), nn.GELU(), nn.Linear(h*4, h))
    def forward(self, x):
        out = self.attn(self.norm1(x))
        if isinstance(out, tuple): out = out[0]
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x

# --- D. Mamba2 (Industrial SSM) ---
class MambaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model'])
        self.layers = nn.ModuleList([
            Mamba2(
                d_model=CONFIG['d_model'], 
                d_state=64, 
                d_conv=4, 
                expand=2
            ) for _ in range(CONFIG['n_layers'])
        ])
        self.norm = nn.LayerNorm(CONFIG['d_model'])
        self.head = nn.Linear(CONFIG['d_model'], CONFIG['vocab_size'])

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x) # Mamba2 forward 只需要 x
        x = self.norm(x)
        return self.head(x)

# ==========================================
# 3. 核心评测指标 (Passkey & Induction)
# ==========================================

def test_passkey(model, tokenizer, seq_len):
    """草堆寻针：测试长距离记忆召回"""
    model.eval()
    passkey = str(random.randint(1000, 9999))
    prefix = f"# API_KEY = {passkey}\n"
    suffix = "\n# Retrieve API_KEY: "
    
    # 填充代码
    dummy = "def fast_math(x): return x**2 + 1\n" * 200
    
    pre_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suf_ids = tokenizer.encode(suffix, add_special_tokens=False)
    dum_ids = tokenizer.encode(dummy, add_special_tokens=False)
    
    # 动态调整长度
    fill_len = seq_len - len(pre_ids) - len(suf_ids)
    if fill_len <= 0: return 0.0
    
    # 循环填充直到填满
    full_dum = (dum_ids * (fill_len // len(dum_ids) + 1))[:fill_len]
    input_ids = pre_ids + full_dum + suf_ids
    input_tensor = torch.tensor([input_ids], device=CONFIG['device'])
    
    with torch.no_grad():
        logits = model(input_tensor)
        pred_id = torch.argmax(logits[0, -1, :]).item()
        pred_str = tokenizer.decode([pred_id]).strip()
        
    success = (pred_str == passkey)
    print(f"  [Passkey] Len={len(input_ids)} | Target: {passkey} | Pred: {pred_str} | {'✅' if success else '❌'}")
    return 1.0 if success else 0.0

def test_induction(model, vocab_size, seq_len=512):
    """归纳头测试：Context Learning 能力"""
    model.eval()
    half = seq_len // 2
    rand_tok = torch.randint(0, vocab_size, (1, half), device=CONFIG['device'])
    input_ids = torch.cat([rand_tok, rand_tok], dim=1)
    
    with torch.no_grad():
        logits = model(input_ids)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        # 比较前半段和后半段的 loss
        loss = loss_fct(logits[..., :-1, :].reshape(-1, vocab_size), input_ids[..., 1:].reshape(-1))
        loss = loss.view(1, -1)
        
        first_half = loss[:, :half-1].mean().item()
        second_half = loss[:, half-1:].mean().item()
        
    score = first_half - second_half
    print(f"  [Induction] 1st Loss: {first_half:.2f} | 2nd Loss: {second_half:.2f} | Score: {score:.2f}")
    return score

# ==========================================
# 4. 主程序
# ==========================================
def get_data():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(CONFIG['dataset'], split="train", streaming=True)
    def gen():
        for item in ds:
            txt = item.get('output', item.get('text', ''))
            ids = tokenizer.encode(txt, truncation=True, max_length=CONFIG['train_seq_len']+1)
            if len(ids) > 50:
                yield torch.tensor(ids[:-1]), torch.tensor(ids[1:])
    return tokenizer, gen

def run_benchmark():
    tokenizer, data_gen = get_data()
    
    # 注册所有模型
    models_to_test = {
        "RNN (LSTM)": RNNModel(),
        "Transformer": TransformerModel(),
    }
    if HAS_FLA: models_to_test["Hybrid (FLA)"] = FLAModel()
    if HAS_MAMBA: models_to_test["Mamba2 (SSM)"] = MambaModel()
    
    results = {}

    for name, model in models_to_test.items():
        print(f"\n⚡ Testing: [{name}]")
        print("="*40)
        
        model = model.to(CONFIG['device']).to(dtype=CONFIG['dtype'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4) # 稍微激进的学习率
        criterion = nn.CrossEntropyLoss()
        
        # --- Training ---
        model.train()
        train_iter = iter(data_gen())
        
        # Warmup (Compile Kernels)
        print("  🔥 Warming up (JIT)...")
        try:
            for _ in range(5):
                x, y = next(train_iter)
                model(x.to(CONFIG['device']).unsqueeze(0))
            torch.cuda.synchronize()
        except: pass

        start_time = time.time()
        total_tokens = 0
        losses = []
        
        print(f"  🏃 Training for {CONFIG['steps']} steps...")
        for step in range(CONFIG['steps']):
            try:
                # 简单 Batching
                x_list, y_list = [], []
                for _ in range(CONFIG['batch_size']):
                    x, y = next(train_iter)
                    x_list.append(x); y_list.append(y)
                
                max_l = max([len(t) for t in x_list])
                # Pad
                x_b = torch.stack([torch.nn.functional.pad(t, (0, max_l-len(t)), value=tokenizer.eos_token_id) for t in x_list]).to(CONFIG['device'])
                y_b = torch.stack([torch.nn.functional.pad(t, (0, max_l-len(t)), value=-100) for t in y_list]).to(CONFIG['device'])
                
                optimizer.zero_grad()
                logits = model(x_b)
                loss = criterion(logits.view(-1, CONFIG['vocab_size']), y_b.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                losses.append(loss.item())
                total_tokens += x_b.numel()
                
                if step % 50 == 0:
                    print(f"    Step {step}: Loss {loss.item():.4f}")
            except StopIteration: break
            
        duration = time.time() - start_time
        speed = total_tokens / duration
        avg_loss = sum(losses)/len(losses) if losses else 0
        ppl = math.exp(avg_loss) if avg_loss < 20 else 99999
        
        # --- Evaluation ---
        print("  🧪 Running Advanced Evals...")
        passkey = test_passkey(model, tokenizer, seq_len=CONFIG['test_seq_len'])
        induction = test_induction(model, CONFIG['vocab_size'])
        
        results[name] = {
            "Speed": speed, 
            "PPL": ppl, 
            "Passkey": passkey, 
            "Induction": induction,
            "Mem": torch.cuda.max_memory_allocated() / 1024**3
        }
        
        # Cleanup
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    # --- Final Report ---
    print("\n\n🏆 ULTIMATE BENCHMARK REPORT")
    print("="*95)
    print(f"{'Model':<15} | {'Speed (tok/s)':<15} | {'PPL':<8} | {'Passkey':<8} | {'Induction':<10} | {'Mem (GB)':<8}")
    print("-" * 95)
    for name, res in results.items():
        print(f"{name:<15} | {res['Speed']:<15.0f} | {res['PPL']:<8.2f} | {res['Passkey']:<8.0f} | {res['Induction']:<10.2f} | {res['Mem']:.1f}")
    print("="*95)

if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted.")
    finally:
        torch.cuda.synchronize()