import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import qwen3_asr
    print("qwen3_asr imported successfully!")
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print("sys.path:", sys.path)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

try:
    from backend.qwen_asr_api.asr import Qwen3ASR
    print("Qwen3ASR from backend.qwen_asr_api.asr imported successfully!")
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError for Qwen3ASR: {e}")
    print("sys.path:", sys.path)
except Exception as e:
    print(f"An unexpected error occurred for Qwen3ASR: {e}")
