#!/usr/bin/env python3
"""
prepare_private_data.py

Downloads partial code from bigcode/the-stack-dedup for training.
Target Languages: JSON, Terminal Script (Shell), Python, JavaScript.

Outputs a JSONL file compatible with cpt_train.py (keys: "content").

Note: 'bigcode/the-stack-dedup' is a gated dataset. 
Ensure you have accepted the terms and logged in via `huggingface-cli login`.
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_FILE = "my_private_code.jsonl"
SAMPLES_PER_LANG = 5000  # Adjust valid samples per language
MAX_CHAR_LENGTH = 100000 # Skip extremely large files
MIN_CHAR_LENGTH = 50     # Skip empty/tiny files

# Mapping readable name -> data_dir in the-stack-dedup
# "Terminal Script" is mapped to "Shell"
LANGUAGES = {
    "Python": "data/Python",
    "JavaScript": "data/JavaScript", 
    "JSON": "data/JSON", 
    "Shell": "data/Shell"
}

def main():
    print(f"Starting download from bigcode/the-stack-dedup...")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Target count per language: {SAMPLES_PER_LANG}")
    print("Ensure you have run `huggingface-cli login` and have access to bigcode/the-stack-dedup.\n")

    total_saved = 0

    # Overwrite output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for lang_name, data_dir in LANGUAGES.items():
            print(f"--- Fetching {lang_name} ({data_dir}) ---")
            
            try:
                # Use streaming to avoid downloading the entire TB-scale dataset
                ds = load_dataset(
                    "bigcode/the-stack-dedup", 
                    data_dir=data_dir, 
                    split="train", 
                    streaming=True,
                    
                )
                
                count = 0
                pbar = tqdm(total=SAMPLES_PER_LANG, desc=f"Saving {lang_name}")
                
                for sample in ds:
                    content = sample.get("content", "")
                    
                    # Basic filtering to ensure quality/compatibility
                    if len(content) < MIN_CHAR_LENGTH:
                        continue
                    if len(content) > MAX_CHAR_LENGTH:
                        continue
                    
                    # Construct record compatible with training scripts
                    record = {
                        "content": content,
                        "language": lang_name,
                        "meta": {
                            "source": "the-stack-dedup",
                            "repo_name": sample.get("max_stars_repo_name", ""),
                            "path": sample.get("max_stars_repo_path", "")
                        }
                    }
                    
                    f_out.write(json.dumps(record) + "\n")
                    
                    count += 1
                    pbar.update(1)
                    
                    if count >= SAMPLES_PER_LANG:
                        break
                
                pbar.close()
                total_saved += count
                print(f"Finished {lang_name}: {count} docs.")
                
            except Exception as e:
                print(f"\n[ERROR] Failed to load {lang_name}: {e}")
                print("Tip: If this is an authentication error, verify your HF token.")

    print(f"\nAll done! Total documents collected: {total_saved}")
    print(f"Saved to {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()
