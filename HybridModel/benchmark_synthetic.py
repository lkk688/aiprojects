import torch
import torch.nn as nn
import time
import os
import shutil
from fla.layers import GatedLinearAttention

# ==========================================
# 0. 强制清理 Triton 缓存 (修复速度问题的关键)
# ==========================================
triton_cache = os.path.expanduser("~/.triton/cache")
if os.path.exists(triton_cache):
    print(f"🧹 Clearing Triton cache at {triton_cache} to fix kernel issues...")
    try:
        shutil.rmtree(triton_cache)
    except:
        print("   (Warning: Could not clear cache, requires manual deletion)")

# ==========================================
# 1. 配置：针对 H100 优化的合成任务
# ==========================================
CONFIG = {
    "vocab_size": 128,      # 极小词表，保证能学会
    "d_model": 256,
    "n_layers": 2,
    "n_heads": 4,
    "seq_len": 4096,        # 长序列，测试记忆力
    "batch_size": 8,
    "steps": 200,           # 足够学会简单规律
    "device": "cuda"
}

print(f"🚀 Running Synthetic Benchmark (Focus: Induction & Speed)")

# ==========================================
# 2. 数据生成器：专门训练 Induction 能力
# ==========================================
def get_synthetic_batch():
    """
    生成 "Induction" 数据：
    [Key, Val, ..., Key] -> Target: Val
    这迫使模型学会 '查找历史'
    """
    #前半段随机
    half_len = CONFIG['seq_len'] // 2
    rand_tokens = torch.randint(0, CONFIG['vocab_size']-1, (CONFIG['batch_size'], half_len), device=CONFIG['device'])
    
    # 后半段完全复制前半段 (Induction Task)
    # Input:  [A B C ... A B C]
    # Target: [B C ... A B C .]
    input_ids = torch.cat([rand_tokens, rand_tokens], dim=1)
    
    # 构造 Target (右移一位)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    target_ids[:, -1] = -100 # 最后一位不预测
    
    return input_ids, target_ids

# ==========================================
# 3. 模型定义
# ==========================================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model'])
        self.layers = nn.ModuleList([
            GLABlock(CONFIG['d_model'], CONFIG['n_heads']) 
            for _ in range(CONFIG['n_layers'])
        ])
        self.head = nn.Linear(CONFIG['d_model'], CONFIG['vocab_size'])

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

class GLABlock(nn.Module):
    def __init__(self, h, heads):
        super().__init__()
        self.norm = nn.LayerNorm(h)
        # 显式指定参数，防止报错
        self.attn = GatedLinearAttention(
            hidden_size=h, 
            num_heads=heads, 
            mode='chunk' # H100 高性能模式
        )
        self.mlp = nn.Sequential(nn.Linear(h, h*2), nn.GELU(), nn.Linear(h*2, h))

    def forward(self, x):
        # 这里的 [0] 是取 output，丢弃 state
        x = x + self.attn(self.norm(x))[0]
        x = x + self.mlp(self.norm(x))
        return x

# ==========================================
# 4. 训练与验证循环
# ==========================================
def run():
    model = HybridModel().to(CONFIG['device']).bfloat16()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\n⚡ Training Hybrid Model on Induction Task...")
    model.train()
    
    # 预热 (触发编译)
    print("  🔥 Warming up JIT compiler...")
    x, y = get_synthetic_batch()
    for _ in range(5): model(x)
    torch.cuda.synchronize()
    
    start_time = time.time()
    total_tokens = 0
    
    for step in range(CONFIG['steps']):
        x, y = get_synthetic_batch()
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, CONFIG['vocab_size']), y.view(-1))
        loss.backward()
        optimizer.step()
        
        total_tokens += x.numel()
        
        if step % 20 == 0:
            print(f"  Step {step:03d} | Loss: {loss.item():.4f}")
            if loss.item() < 0.1:
                print("  ✅ Converged! (Learned Induction)")
                break
                
    duration = time.time() - start_time
    speed = total_tokens / duration
    
    print(f"\n📊 Results:")
    print(f"  > Final Loss: {loss.item():.4f} (Should be near 0)")
    print(f"  > Speed: {speed:.0f} tokens/sec")
    
    # 简单的 Passkey 测试
    print("\n🧪 Quick Passkey Test:")
    model.eval()
    # 构造: "Key is 42 ... (4000 tokens) ... Key is" -> 应该预测 42
    test_seq = torch.zeros((1, CONFIG['seq_len']), dtype=torch.long, device=CONFIG['device'])
    test_seq[0, 0] = 42  # Secret Key
    test_seq[0, -1] = 42 # Prompt
    
    with torch.no_grad():
        out = model(test_seq)
        pred = out[0, -2].argmax().item() # 预测倒数第二个位置的下一个
        
    print(f"  > Input: [42, 0, 0, ..., 42]")
    print(f"  > Target: 42")
    print(f"  > Pred:   {pred}")
    print(f"  > Result: {'✅ Success' if pred == 42 else '❌ Fail'}")

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Hint: If speed is slow (~1000), Triton is broken. Try reinstalling fla.")