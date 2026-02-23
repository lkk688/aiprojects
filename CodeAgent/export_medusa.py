import torch
import numpy as np

def convert_medusa_to_mlx(pth_path, output_path):
    print(f"Loading {pth_path}...")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    
    mlx_weights = {}
    for key, tensor in state_dict.items():
        # Convert to float32 numpy arrays for Apple MLX
        numpy_array = tensor.float().numpy()
        mlx_weights[key] = numpy_array
        print(f"Exported: {key} -> Shape: {numpy_array.shape}")
        
    np.savez(output_path, **mlx_weights)
    print(f"✅ Saved MLX compatible Medusa weights to {output_path}")

convert_medusa_to_mlx("h100_medusa_heads.pth", "mlx_medusa_heads.npz")