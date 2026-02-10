#https://github.com/fla-org/flash-linear-attention?tab=readme-ov-file#installation
import torch
from fla.layers import MultiScaleRetention
batch_size, num_heads, seq_len, hidden_size = 32, 4, 2048, 1024
device, dtype = 'cuda:0', torch.bfloat16

retnet = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)

retnet
x = torch.randn(batch_size, seq_len, hidden_size).to(device=device, dtype=dtype)
y, *_ = retnet(x)
y.shape

from fla.models import GLAConfig
from transformers import AutoModelForCausalLM
config = GLAConfig()
config

AutoModelForCausalLM.from_config(config)