import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id = "unsloth/GLM-4.7-Flash-GGUF",
#     local_dir = "unsloth/GLM-4.7-Flash-GGUF",
#     allow_patterns = ["*UD-Q4_K_XL*"],
# )

#unsloth/Qwen3-Coder-Next-GGUF:UD-Q4_K_XL
snapshot_download(
    repo_id = "unsloth/Qwen3-Coder-Next-GGUF",
    local_dir = "unsloth/Qwen3-Coder-Next-GGUF",
    allow_patterns = ["*UD-Q4_K_XL*"],
)