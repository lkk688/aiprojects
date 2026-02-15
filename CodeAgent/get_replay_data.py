import os
import json
import gc
import time
from datasets import load_dataset
from tqdm import tqdm

# ================= 配置区域 =================
OUTPUT_FILE = "my_replay_data_mixed.jsonl"

# 定义你需要的语言和对应数量 (用于防止遗忘的 Replay 数据)
# 推荐比例：主要语言(如Python)占大头，其他常用语言作为辅助
TARGET_CONFIG = {
    "python": 2000,      # 主要目标
    "javascript": 1500,  # 前端/Nodejs
    "shell": 1000,       # 对应 "Terminal" / Bash 脚本
    "java": 500,         # 强类型语言补充
}

# 数据集设置
DATASET_NAME = "bigcode/starcoderdata"
# ===========================================

def download_lang(language, target_count, file_handle, global_counter):
    """下载特定语言的数据并写入文件"""
    print(f"\n[Task] Starting download for: {language} (Target: {target_count})")
    
    try:
        # 1. 加载数据集 (Streaming模式)
        # 注意: split="train" 是必须的
        ds = load_dataset(
            DATASET_NAME, 
            data_dir=language, 
            split="train", 
            streaming=True
        )
        
        # 2. 稍微打乱数据 (Shuffle)
        # buffer_size=1000 意味着在流式读取时，会先加载1000条进内存打乱
        # 这有助于避免连续下载到同一个项目的代码
        ds = ds.shuffle(buffer_size=1000, seed=42)

        local_count = 0
        pbar = tqdm(total=target_count, desc=f"Processing {language}")

        # 3. 遍历并保存
        for sample in ds:
            if local_count >= target_count:
                break
            
            # 获取代码内容
            code_content = sample.get("content", "")
            
            # 简单过滤：太短的通常是无意义片段，太长的可能导致 OOM
            if len(code_content) < 50 or len(code_content) > 100000:
                continue

            # 构造符合 CPT 训练格式的 JSON
            entry = {
                "content": code_content,
                "language": language,  # 记录语言类型
                "source": "starcoder_replay",
                "file_path": f"replay_{language}_{global_counter}.{language}"
            }
            
            file_handle.write(json.dumps(entry) + "\n")
            
            local_count += 1
            global_counter += 1
            pbar.update(1)
        
        pbar.close()
        print(f"[Done] Extracted {local_count} samples for {language}")
        
        return global_counter

    except Exception as e:
        print(f"\n[Warning] Failed to download {language}: {e}")
        print(f"Skipping {language}...")
        return global_counter
    finally:
        # === 关键修复 ===
        # 显式删除 dataset 对象并强制垃圾回收
        # 这能有效防止之前的 'PyGILState_Release' 崩溃
        if 'ds' in locals():
            del ds
        gc.collect()

def main():
    print(f"Initializing Replay Data Collection...")
    print(f"Target Config: {TARGET_CONFIG}")
    
    total_samples = 0
    
    # 使用 'a' (append) 还是 'w' (write)? 
    # 这里用 'w' 每次覆盖重新生成，避免重复
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        
        for lang, count in TARGET_CONFIG.items():
            total_samples = download_lang(lang, count, f, total_samples)
            # 稍微暂停一下，让网络和线程喘口气
            time.sleep(1)

    print(f"\n" + "="*50)
    print(f"[Success] All done!")
    print(f"Total samples saved: {total_samples}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"="*50)

if __name__ == "__main__":
    main()