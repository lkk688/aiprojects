import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import time
import math
from fla.layers import GatedLinearAttention  # 核心：引入线性注意力层

# ==========================================
# 1. 配置与超参数
# ==========================================
CONFIG = {
    "vocab_size": 50257,    # GPT-2 词表大小
    "d_model": 512,         # 隐藏层维度 (保持一致以公平对比)
    "n_layers": 4,          # 层数
    "n_heads": 8,           # 头数
    "seq_len": 16384, #4096,        # 序列长度 (关键：越长线性优势越明显)
    "batch_size": 1, #8,       # 批次大小
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "steps": 20            # 每个模型跑多少步 (用于测试速度)
}

print(f"🚀 Running on {CONFIG['device']} with SeqLen={CONFIG['seq_len']}")

# ==========================================
# 2. 模型定义
# ==========================================

# --- A. 基准 RNN (LSTM) ---
class RNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model'])
        # LSTM 无法并行训练，通常很慢
        self.lstm = nn.LSTM(
            input_size=CONFIG['d_model'],
            hidden_size=CONFIG['d_model'],
            num_layers=CONFIG['n_layers'],
            batch_first=True
        )
        self.head = nn.Linear(CONFIG['d_model'], CONFIG['vocab_size'])

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        return self.head(x)

# --- B. 基准 Transformer (O(N^2)) ---
class TransformerBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model'])
        self.pos_embed = nn.Parameter(torch.zeros(1, CONFIG['seq_len'], CONFIG['d_model']))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CONFIG['d_model'],
            nhead=CONFIG['n_heads'],
            dim_feedforward=CONFIG['d_model']*4,
            dropout=0.0,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=CONFIG['n_layers'])
        self.head = nn.Linear(CONFIG['d_model'], CONFIG['vocab_size'])

    def forward(self, x):
        B, T = x.size()
        # 简单的位置编码截断
        pos = self.pos_embed[:, :T, :]
        x = self.embed(x) + pos
        # 生成因果掩码 (Causal Mask)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        return self.head(x)

# --- C. Hybrid/Linear Attention (O(N)) ---
# 使用 Gated Linear Attention (GLA) - Qwen3/Mamba 风格的代表
class HybridGLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model'])
        
        self.layers = nn.ModuleList([
            GLABlock(CONFIG['d_model'], CONFIG['n_heads']) 
            for _ in range(CONFIG['n_layers'])
        ])
        self.norm = nn.LayerNorm(CONFIG['d_model'])
        self.head = nn.Linear(CONFIG['d_model'], CONFIG['vocab_size'])

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

class GLABlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        # 这里调用 fla 库的 GatedLinearAttention
        # 它是 FlashAttention 的线性变体，支持并行训练和递归推理
        self.attn = GatedLinearAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mode='chunk' # H100 必开：使用 Triton 融合算子加速
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x):
        # 这里的 attention 不需要 mask，因为 GLA 内部处理了因果性
        x = x + self.attn(self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# ==========================================
# 3. 数据加载 (TinyStories)
# ==========================================
def get_dataloader():
    print("📚 Loading TinyStories dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 使用流式加载，无需下载整个数据集
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        encodings = tokenizer(
            texts, 
            truncation=True, 
            padding='max_length', 
            max_length=CONFIG['seq_len'] + 1,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids']
        return input_ids[:, :-1], input_ids[:, 1:] # x, y

    # 取一个简单的迭代器
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], collate_fn=collate_fn)
    return dataloader

# ==========================================
# 4. 训练与评估循环
# ==========================================
def train_and_evaluate(model, name, dataloader):
    print(f"\n⚡ Training Model: [{name}]")
    print("-" * 40)
    
    model = model.to(CONFIG['device']).bfloat16() # H100 建议使用 BF16
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    print("  🔥 Warming up (Compiling Triton Kernels)...")
    # 先空跑 5 步，触发 Triton 编译，不计入时间
    warmup_steps = 5
    warmup_iter = iter(dataloader)
    for _ in range(warmup_steps):
        try:
            wx, wy = next(warmup_iter)
            wx, wy = wx.to(CONFIG['device']), wy.to(CONFIG['device'])
            optimizer.zero_grad()
            loss = criterion(model(wx).view(-1, CONFIG['vocab_size']), wy.view(-1))
            loss.backward()
            optimizer.step()
        except StopIteration:
            break
    
    print("  🚀 Benchmark started!")
    # 重置计时器和计数器
    torch.cuda.synchronize() # 确保 GPU 此时空闲
    start_time = time.time()
    total_loss = 0
    
    # 预热显存
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    
    total_loss = 0
    start_time = time.time()
    
    step = 0
    for x, y in dataloader:
        if step >= CONFIG['steps']: break
        
        x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
        
        optimizer.zero_grad()
        
        # 记录前向传播显存
        if step == 10: 
            torch.cuda.reset_peak_memory_stats()
            
        logits = model(x)
        loss = criterion(logits.view(-1, CONFIG['vocab_size']), y.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if step % 20 == 0:
            print(f"  Step {step}/{CONFIG['steps']} | Loss: {loss.item():.4f}")
        step += 1

    end_time = time.time()
    avg_loss = total_loss / CONFIG['steps']
    ppl = math.exp(avg_loss)
    
    # 统计数据
    total_tokens = CONFIG['batch_size'] * CONFIG['seq_len'] * CONFIG['steps']
    duration = end_time - start_time
    tokens_per_sec = total_tokens / duration
    max_mem = torch.cuda.max_memory_allocated() / 1024**2 # MB
    
    print(f"\n📊 [{name}] Results:")
    print(f"  > Perplexity (PPL): {ppl:.2f}")
    print(f"  > Speed: {tokens_per_sec:.0f} tokens/sec")
    print(f"  > Peak Memory: {max_mem:.0f} MB")
    
    return {
        "model": name,
        "ppl": ppl,
        "speed": tokens_per_sec,
        "mem": max_mem
    }

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    dataloader = get_dataloader()
    
    results = []
    
    # 1. 测试 RNN (基准)
    rnn = RNNBaseline()
    results.append(train_and_evaluate(rnn, "RNN (LSTM)", dataloader))
    del rnn
    
    # 2. 测试 Transformer (标准)
    # 注意：如果显存不够，可能需要减小 batch_size
    tf = TransformerBaseline()
    results.append(train_and_evaluate(tf, "Transformer (O(N^2))", dataloader))
    del tf
    
    # 3. 测试 Hybrid/Linear (前沿)
    gla = HybridGLA()
    results.append(train_and_evaluate(gla, "Hybrid (GLA/Linear)", dataloader))
    del gla
    
    print("\n\n🏆 最终对比总结 (H100 Performance)")
    print("=" * 60)
    print(f"{'Model':<20} | {'PPL (Lower Better)':<20} | {'Speed (Higher Better)':<20} | {'Mem (MB)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<20} | {r['ppl']:<20.2f} | {r['speed']:<20.0f} | {r['mem']:<10.0f}")
    print("=" * 60)