import torch

def check_pytorch():
    """
    Checks the PyTorch version and CUDA availability.
    """
    print(f"PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_pytorch()
