import os
import torch
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Union
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments, HfArgumentParser, set_seed
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 参数定义 (Arguments)
# ==========================================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="unsloth/Qwen2.5-Coder-14B",
        metadata={"help": "Model name or path to checkpoint"}
    )
    load_in_4bit: bool = field(default=True, metadata={"help": "Use 4-bit quantization"})
    max_seq_length: int = field(default=8192, metadata={"help": "Context window size"})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha"})

@dataclass
class DataArguments:
    train_files: List[str] = field(
        default_factory=lambda: ["my_private_code.jsonl"],
        metadata={"help": "List of JSONL files for training (e.g., private + replay)"}
    )
    fim_rate: float = field(default=0.5, metadata={"help": "Probability of applying FIM transformation (0.5 is standard)"})
    fim_spm_rate: float = field(default=0.0, metadata={"help": "Probability of SPM (Suffix-Prefix-Middle). Usually keep 0 for PSM."})

# ==========================================
# 2. FIM (Fill-In-the-Middle) 核心逻辑
# ==========================================
class FIMProcessor:
    """
    处理 FIM 变换的类。
    Qwen2.5-Coder FIM Tokens:
    <|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>
    """
    def __init__(self, tokenizer, fim_rate=0.5):
        self.tokenizer = tokenizer
        self.fim_rate = fim_rate
        # 确保 special tokens 存在，虽然 Qwen 自带，但为了保险
        self.prefix_token = "<|fim_prefix|>"
        self.suffix_token = "<|fim_suffix|>"
        self.middle_token = "<|fim_middle|>"
        self.eos_token = tokenizer.eos_token

    def __call__(self, examples):
        output_texts = []
        for content in examples["content"]:
            # 1. 过滤太短的代码
            if len(content) < 50:
                continue

            # 2. 决定是否应用 FIM
            if np.random.rand() < self.fim_rate:
                # === FIM 模式 (PSM: Prefix-Suffix-Middle) ===
                try:
                    # 随机切分点。为了保持代码语义，尽量不在开头和结尾切。
                    # 更好的做法是基于行切分，但基于字符切分通用性更强。
                    boundary = np.random.randint(low=int(len(content) * 0.2), high=int(len(content) * 0.8))
                    
                    prefix = content[:boundary]
                    suffix = content[boundary:]
                    
                    # 构造格式: <|fim_prefix|>Prefix<|fim_suffix|>Suffix<|fim_middle|>Middle<|endoftext|>
                    # 注意：Middle 部分就是我们要让模型预测的部分，这里通常不需要显式写 Middle 内容，
                    # 但在 CPT Packing 训练中，我们需要构造完整的序列让模型去学。
                    # 对于 PSM 任务，模型的输入是 Prefix + Suffix，目标是生成 Middle (这里其实是 Suffix 的一部分，还是有些概念混淆)
                    
                    # 修正的 FIM 切分逻辑 (Standard Way):
                    # 将文档切成三段: A (Prefix), B (Middle), C (Suffix)
                    # 变换为: Prefix(A) + Suffix(C) + Middle(B)
                    
                    # 这里为了简化，我们采用 "Cut and Predict Later"
                    # 切分点 split_point
                    # Prefix = content[:split_point]
                    # Suffix = content[split_point:]
                    # 这种其实是 CLM。
                    
                    # === 真正的 FIM 实现 ===
                    # 随机选一段作为 Middle
                    doc_len = len(content)
                    # 随机选取 Middle 的长度 (例如文档的 10%-30%)
                    middle_len = int(doc_len * np.random.uniform(0.1, 0.3))
                    # 随机选取 Middle 的起始位置
                    start = np.random.randint(0, doc_len - middle_len + 1)
                    end = start + middle_len
                    
                    prefix = content[:start]
                    middle = content[start:end]
                    suffix = content[end:]
                    
                    # 构造符合 Qwen 格式的字符串
                    # 训练目标：给定 <|fim_prefix|>P<|fim_suffix|>S<|fim_middle|>，预测 M
                    fmt_text = f"{self.prefix_token}{prefix}{self.suffix_token}{suffix}{self.middle_token}{middle}{self.eos_token}"
                    output_texts.append(fmt_text)
                    
                except ValueError:
                    # 如果切分失败，回退到 CLM
                    output_texts.append(content + self.eos_token)
            else:
                # === CLM 模式 (普通从左到右) ===
                output_texts.append(content + self.eos_token)
                
        return {"text": output_texts}

# ==========================================
# 3. 主程序
# ==========================================
def main():
    # 1. 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # 允许从命令行或 json/yaml 文件读取
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    logger.info(f"Model Parameters: {model_args}")
    logger.info(f"Data Parameters: {data_args}")

    # 2. 加载数据
    if not data_args.train_files:
        raise ValueError("No training files provided! Use --train_files file1.jsonl file2.jsonl")
    
    logger.info(f"Loading datasets from: {data_args.train_files}")
    dataset = load_dataset("json", data_files=data_args.train_files, split="train")
    
    # 3. 加载模型 (Unsloth)
    logger.info("Loading Unsloth model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_args.model_name_or_path,
        max_seq_length = model_args.max_seq_length,
        dtype = None, # Auto detection
        load_in_4bit = model_args.load_in_4bit,
    )

    # 4. 配置 LoRA
    logger.info("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = model_args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = model_args.lora_alpha,
        lora_dropout = 0, # Unsloth 建议为 0
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = training_args.seed,
    )

    # 5. 数据处理 (应用 FIM)
    logger.info(f"Processing data with FIM rate: {data_args.fim_rate}...")
    fim_processor = FIMProcessor(tokenizer, fim_rate=data_args.fim_rate)
    
    # 使用 .map 处理，batched=True 提高效率
    # remove_columns 很重要，防止旧列干扰 Trainer
    processed_dataset = dataset.map(
        fim_processor,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Applying FIM transformation"
    )

    # 6. 配置 Trainer
    # SFTTrainer 在 packing=True 时，会自动处理 input_ids 和 labels 的对齐
    # 它会将多个短样本拼成 max_seq_length，大幅提升训练效率
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = processed_dataset,
        dataset_text_field = "text",
        max_seq_length = model_args.max_seq_length,
        packing = True, 
        args = training_args,
    )

    # 7. 开始训练
    logger.info("Starting training...")
    trainer_stats = trainer.train()

    # 8. 保存模型
    logger.info("Saving model...")
    # 保存 LoRA 适配器
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    # 保存为 GGUF (可选，方便后续 ollama 使用)
    # model.save_pretrained_gguf(training_args.output_dir, tokenizer, quantization_method = "f16")

    logger.info(f"Training complete. Output saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()