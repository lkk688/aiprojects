import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
import time
import random
import gc
import os
from fla.layers import GatedLinearAttention

# ==========================================
# 1. 全局配置
# ==========================================
CONFIG = {
    "vocab_size": 50257,    # GPT-2 Tokenizer
    "d_model": 512,         # 小模型维度
    "n_layers": 4,
    "n_heads": 8,
    "train_seq_len": 2048,  # 训练长度
    "test_seq_len": 4096,   # 测试长度 (外推测试)
    "batch_size": 8,        # H100 显存够大，开大一点
    "steps": 150,           # 稍微多跑几步，保证学会
    "device": "cuda",
    "dtype": torch.bfloat16, # H100 默认用 BF16，如果 Loss 不降，改用 torch.float32
    "dataset": "flytech/python-codes-25k"
}

print(f"🚀 Benchmarking on {CONFIG['dataset']} | Device: {CONFIG['device']} | Dtype: {CONFIG['dtype']}")

# ==========================================
# 2. 模型定义
# ==========================================

class TransformerBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model'])
        # 简单的位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, CONFIG['test_seq_len'] * 2, CONFIG['d_model']))
        layer = nn.TransformerEncoderLayer(CONFIG['d_model'], CONFIG['n_heads'], batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, CONFIG['n_layers'])
        self.head = nn.Linear(CONFIG['d_model'], CONFIG['vocab_size'])
    def forward(self, x):
        B, T = x.size()
        pos = self.pos_embed[:, :T, :]
        x = self.embed(x) + pos
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        return self.head(x)

class HybridGLA(nn.Module):
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
        # ✅ 修复点：显式指定参数名，防止位置参数错乱
        self.attn = GatedLinearAttention(
            hidden_size=h, 
            num_heads=heads, 
            mode='chunk'
        )
        self.norm2 = nn.LayerNorm(h)
        self.mlp = nn.Sequential(nn.Linear(h, h*4), nn.GELU(), nn.Linear(h*4, h))
    def forward(self, x):
        # ✅ 修复点：兼容 API 返回值，只取第一个 output
        out = self.attn(self.norm1(x))
        if isinstance(out, tuple): out = out[0]
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x

# ==========================================
# 3. 核心测试逻辑：Passkey & Induction
# ==========================================

def test_passkey_retrieval(model, tokenizer, seq_len=4096):
    model.eval()
    passkey = str(random.randint(1000, 9999))
    prefix = f"# The API secret key is {passkey}.\n" # Python 风格注释
    suffix = "\n# What is the API secret key? The key is"
    
    dummy_code = "def process_data(x):\n    return x * 2\n" * 200
    
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    dummy_ids = tokenizer.encode(dummy_code, add_special_tokens=False)
    
    fill_len = seq_len - len(prefix_ids) - len(suffix_ids)
    if fill_len <= 0: return 0.0
    
    input_ids = prefix_ids + dummy_ids[:fill_len] + suffix_ids
    input_tensor = torch.tensor([input_ids], device=CONFIG['device'])
    
    with torch.no_grad():
        logits = model(input_tensor)
        pred_id = torch.argmax(logits[0, -1, :]).item()
        pred_str = tokenizer.decode([pred_id]).strip()
    
    is_correct = (pred_str == passkey) or (passkey.startswith(pred_str))
    print(f"  [Passkey] Len={len(input_ids)} | Target: {passkey} | Pred: {pred_str} | {'✅' if is_correct else '❌'}")
    return 1.0 if is_correct else 0.0

def test_induction_heads(model, vocab_size, seq_len=512):
    model.eval()
    half_len = seq_len // 2
    random_tokens = torch.randint(0, vocab_size, (1, half_len), device=CONFIG['device'])
    input_ids = torch.cat([random_tokens, random_tokens], dim=1)
    
    with torch.no_grad():
        logits = model(input_ids)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        # Shift
        loss = loss_fct(logits[..., :-1, :].reshape(-1, vocab_size), input_ids[..., 1:].reshape(-1))
        loss = loss.view(1, -1)
        
        first_half = loss[:, :half_len-1].mean().item()
        second_half = loss[:, half_len-1:].mean().item()
        
    score = first_half - second_half
    print(f"  [Induction] 1st Loss: {first_half:.2f} | 2nd Loss: {second_half:.2f} | Score: {score:.2f}")
    return score

# ==========================================
# 4. 训练流程
# ==========================================

def get_coding_data():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(CONFIG['dataset'], split="train", streaming=True)
    def generator():
        for item in ds:
            text = item.get('output', item.get('text', ''))
            ids = tokenizer.encode(text, truncation=True, max_length=CONFIG['train_seq_len']+1)
            if len(ids) > 100:
                yield torch.tensor(ids[:-1]), torch.tensor(ids[1:])
    return tokenizer, generator

def run_benchmark():
    tokenizer, data_gen = get_coding_data()
    models = {
        "Transformer": TransformerBaseline(),
        "Hybrid (GLA)": HybridGLA()
    }
    
    results = {}

    for name, model in models.items():
        print(f"\n⚡ Testing Model: [{name}]")
        # ✅ 使用配置的 dtype (BF16 或 FP32)
        model = model.to(CONFIG['device']).to(dtype=CONFIG['dtype'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4) # 稍微调大 LR 让小模型学快点
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        train_iter = iter(data_gen())
        
        # Warmup (JIT Compile)
        print("  🔥 Warming up...")
        try:
            for _ in range(5): 
                x, y = next(train_iter)
                model(x.to(CONFIG['device']).unsqueeze(0))
            torch.cuda.synchronize()
        except: pass
        
        start_time = time.time()
        total_tokens = 0
        losses = []
        
        for step in range(CONFIG['steps']):
            try:
                # 简单 Batch 组装
                x_list, y_list = [], []
                for _ in range(CONFIG['batch_size']):
                    x, y = next(train_iter)
                    x_list.append(x); y_list.append(y)
                
                max_len = max([len(t) for t in x_list])
                x_batch = torch.stack([torch.nn.functional.pad(t, (0, max_len-len(t)), value=tokenizer.eos_token_id) for t in x_list])
                y_batch = torch.stack([torch.nn.functional.pad(t, (0, max_len-len(t)), value=-100) for t in y_list])
                
                x_batch = x_batch.to(CONFIG['device'])
                y_batch = y_batch.to(CONFIG['device'])
                
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits.view(-1, CONFIG['vocab_size']), y_batch.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪防止爆炸
                optimizer.step()
                
                losses.append(loss.item())
                total_tokens += x_batch.numel()
                
                if step % 30 == 0:
                    print(f"    Step {step}: Loss {loss.item():.4f}")
                    
            except StopIteration: break
        
        speed = total_tokens / (time.time() - start_time)
        avg_loss = sum(losses)/len(losses) if losses else 0
        
        print("  🧪 Running Advanced Evals...")
        passkey_score = test_passkey_retrieval(model, tokenizer, seq_len=CONFIG['test_seq_len'])
        induction_score = test_induction_heads(model, CONFIG['vocab_size'])
        
        results[name] = {"Loss": avg_loss, "Speed": speed, "Passkey": passkey_score, "Induction": induction_score}
        
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    print("\n\n🏆 Final Benchmark Results")
    print(f"{'Model':<15} | {'Speed (t/s)':<12} | {'Loss':<6} | {'Passkey':<8} | {'Induction':<10}")
    print("-" * 75)
    for name, res in results.items():
        print(f"{name:<15} | {res['Speed']:<12.0f} | {res['Loss']:<6.2f} | {res['Passkey']:<8.0f} | {res['Induction']:<10.2f}")

if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt: pass