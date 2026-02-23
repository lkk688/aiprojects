import torch
import numpy as np
import os

def convert_to_mlx(pth_path, output_path):
    print(f"Loading {pth_path}...")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    
    mlx_weights = {}
    for key, tensor in state_dict.items():
        # MLX expects standard float32 numpy arrays for the import
        numpy_array = tensor.float().numpy()
        mlx_weights[key] = numpy_array
        
    np.savez(output_path, **mlx_weights)
    print(f"✅ Saved MLX compatible weights to {output_path}")

convert_to_mlx("h100_multi_layer_drafter.pth", "mlx_drafter_weights.npz")
convert_to_mlx("h100_elastic_router.pth", "mlx_router_weights.npz")