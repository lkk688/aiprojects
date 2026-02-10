import torch
import torch.nn as nn
from fla.layers import GatedLinearAttention
import os

# ================= 配置 =================
CONFIG = {
    "d_model": 64,     # 极小模型
    "n_heads": 2,
    "seq_len": 32,     # 极短序列
    "device": "cuda"
}

print(f"🔍 Diagnostic Test running on {CONFIG['device']}...")

# ================= 定义最小化模型 =================
class DebugModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 float32 (FP32) 以排除 BF16 的数值稳定性问题
        self.layer = GatedLinearAttention(
            hidden_size=CONFIG['d_model'], 
            num_heads=CONFIG['n_heads'], 
            mode='chunk'
        )
        self.head = nn.Linear(CONFIG['d_model'], 10)

    def forward(self, x):
        # ✅ 修复：兼容所有返回值数量，只取第一个
        outputs = self.layer(x)
        if isinstance(outputs, tuple):
            x = outputs[0]
        else:
            x = outputs
        return self.head(x)

def run_diagnostic():
    # 1. 初始化模型 (FP32)
    model = DebugModel().to(CONFIG['device']).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 2. 造假数据 (FP32)
    x = torch.randn(2, CONFIG['seq_len'], CONFIG['d_model'], device=CONFIG['device']).float()
    target = torch.randint(0, 10, (2, CONFIG['seq_len']), device=CONFIG['device'])
    
    print("\n🧪 Step 1: Forward Pass (前向传播)...")
    try:
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits.view(-1, 10), target.view(-1))
        print(f"  ✅ Forward successful. Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"  ❌ Forward Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🧪 Step 2: Backward Pass (反向传播)...")
    try:
        loss.backward()
        print("  ✅ Backward successful.")
    except Exception as e:
        print(f"  ❌ Backward Failed: {e}")
        return

    print("\n🧪 Step 3: Gradient Check (梯度检查)...")
    has_grad = False
    
    # 检查关键参数：Gate (g) 和 Value (v) 的投影层
    # 如果这些层有梯度，说明线性注意力机制生效了
    print(f"  {'Param Name':<40} | {'Grad Mean':<12} | {'Status'}")
    print("-" * 70)
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            status = "✅ OK" if grad_mean > 0 else "⚠️ ZERO"
            print(f"  {name:<40} | {grad_mean:.6f}     | {status}")
            if grad_mean > 0:
                has_grad = True
        else:
            print(f"  {name:<40} | None         | ❌ NO GRAD")
            
    if has_grad:
        print("\n🎉 DIAGNOSIS: Gradients are flowing! The kernel is HEALTHY.")
        print("   Next Step: Run your benchmark script again.")
    else:
        print("\n❌ DIAGNOSIS: All Gradients are ZERO or None.")
        print("   Reason: The Triton kernel is compiled incorrectly.")

if __name__ == "__main__":
    run_diagnostic()