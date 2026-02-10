import torch
import time
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from fla.ops.gated_linear_attention import chunk_gla

print(f"🔍 System Diagnosis:")
print(f"   Python: {torch.__version__}")
print(f"   CUDA:   {torch.version.cuda}")
print(f"   GPU:    {torch.cuda.get_device_name(0)}")

# ------------------------------------------------------
# 1. 测试 Mamba 内核 (Selective Scan)
# ------------------------------------------------------
print("\n🧪 Testing Mamba Kernel (Selective Scan)...")
B, L, D, N = 16, 4096, 1024, 16
u = torch.randn(B, D, L, device='cuda', dtype=torch.bfloat16)
delta = torch.randn(B, D, L, device='cuda', dtype=torch.float32) # Mamba delta is fp32
A = torch.randn(D, N, device='cuda', dtype=torch.float32)
B_ = torch.randn(B, N, L, device='cuda', dtype=torch.bfloat16)
C = torch.randn(B, N, L, device='cuda', dtype=torch.bfloat16)
D_ = torch.randn(D, device='cuda', dtype=torch.float32)

torch.cuda.synchronize()
start = time.time()
# 强制调用 CUDA 内核
out = selective_scan_fn(u, delta, A, B_, C, D_, z=None, delta_bias=None, delta_softplus=True)
torch.cuda.synchronize()
dur = time.time() - start

print(f"   ✅ Mamba Kernel Status: RUNNING")
print(f"   ⏱️ Execution Time: {dur*1000:.2f} ms")
if dur > 0.5: # 正常应该在 10ms 以内
    print("   ⚠️ WARNING: Mamba is surprisingly slow. Is it falling back to CPU?")
else:
    print("   🚀 SPEED: Excellent! (Hardware Accelerated)")

# ------------------------------------------------------
# 2. 测试 FLA 内核 (Chunk GLA)
# ------------------------------------------------------
print("\n🧪 Testing FLA Kernel (Triton Chunk GLA)...")
q = torch.randn(B, L, 8, 128, device='cuda', dtype=torch.bfloat16) # [B, L, H, D]
k = torch.randn(B, L, 8, 128, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, L, 8, 128, device='cuda', dtype=torch.bfloat16)
g = torch.randn(B, L, 8, 128, device='cuda', dtype=torch.bfloat16)

# 预热 Triton 编译器
try:
    chunk_gla(q, k, v, g)
    torch.cuda.synchronize()
    
    start = time.time()
    chunk_gla(q, k, v, g)
    torch.cuda.synchronize()
    dur = time.time() - start
    
    print(f"   ✅ FLA Triton Kernel: COMPILED & RUNNING")
    print(f"   ⏱️ Execution Time: {dur*1000:.2f} ms")
    if dur > 0.5:
         print("   ⚠️ WARNING: FLA is slow.")
    else:
         print("   🚀 SPEED: Excellent! (Triton Accelerated)")

except Exception as e:
    print(f"   ❌ FLA Kernel Failed: {e}")
    print("   👉 Solution: Re-install fla with 'pip install .'")