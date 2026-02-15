import os
import json
import argparse
from tqdm import tqdm

# ================= 配置区域 =================
# 你希望扫描的文件后缀
ALLOWED_EXTENSIONS = {
    ".py", ".ipynb",  # Python
    ".js", ".ts", ".jsx", ".tsx", ".vue", # Frontend
    ".java", ".kt", ".scala", # JVM
    ".c", ".cpp", ".h", ".hpp", ".cs", # C-family
    ".go", ".rs", # Modern backend
    ".sh", ".bash", ".zsh", # Shell
    ".sql", ".json", ".yaml", ".yml", ".md" # Config & Docs
}

# 必须忽略的目录 (黑名单)
IGNORE_DIRS = {
    ".git", ".svn", ".hg", ".idea", ".vscode",
    "node_modules", "dist", "build", "target",
    "__pycache__", "env", "venv", ".env",
    "site-packages", "migrations"
}

# 文件大小限制 (避免把 huge.json 或 minified.js 读进去)
MAX_FILE_SIZE = 1024 * 1024  # 1MB
MIN_FILE_SIZE = 10           # 10 Bytes (过滤空文件)

def is_text_file(filepath):
    """简单的二进制文件检查"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            f.read(1024)
            return True
    except UnicodeDecodeError:
        return False
    except Exception:
        return False

def process_directory(root_dir, output_file):
    print(f"Scanning directory: {root_dir} ...")
    
    samples = []
    skipped_count = 0
    
    # 1. 遍历目录
    # os.walk 返回: 当前路径, 当前路径下的文件夹列表, 当前路径下的文件列表
    for current_root, dirs, files in os.walk(root_dir):
        # 修改 dirs 列表，实现原地剪枝 (Pruning)，跳过忽略目录
        # 注意：必须反向遍历或切片，否则删除元素会影响遍历
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            file_path = os.path.join(current_root, file)
            _, ext = os.path.splitext(file)
            
            # 2. 检查后缀
            if ext.lower() not in ALLOWED_EXTENSIONS:
                continue
                
            # 3. 检查文件大小
            try:
                size = os.path.getsize(file_path)
                if size < MIN_FILE_SIZE or size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            # 4. 读取内容
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                # 二次检查：内容是否为空
                if not content.strip():
                    continue
                    
                # 5. 构造数据
                # 提取相对路径，方便模型学习文件结构
                rel_path = os.path.relpath(file_path, root_dir)
                
                entry = {
                    "content": content,
                    "file_path": rel_path,
                    "language": ext.replace(".", ""), # 简单的语言标记
                    "source": "private_repo"
                }
                
                samples.append(entry)

            except Exception as e:
                # print(f"Error reading {file_path}: {e}")
                skipped_count += 1
                continue

    # 6. 写入 JSONL
    print(f"Found {len(samples)} valid code files. Writing to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in tqdm(samples, desc="Writing JSONL"):
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"\n[Success] Processed {len(samples)} files.")
    print(f"[Info] Skipped binary/error files: {skipped_count}")
    print(f"[Output] Saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Code Folder to JSONL")
    parser.add_argument("--input", type=str, required=True, help="Input folder path (your repo)")
    parser.add_argument("--output", type=str, default="my_private_code.jsonl", help="Output JSONL file path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
    else:
        process_directory(args.input, args.output)

"""
(py312) [010796032@g20 CodeAgent]$ python prepare_private_datav2.py --input ../../MyRepo/DeepDataMiningLearning/
"""