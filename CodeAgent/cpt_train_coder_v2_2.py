import os
import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import TrainingArguments, HfArgumentParser, set_seed

from unsloth import FastLanguageModel
from trl import SFTTrainer


# ============================================================
# Logging
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Argument Definitions
# ============================================================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="unsloth/Qwen2.5-Coder-14B",
        metadata={"help": "Main model name or path to checkpoint (student / trainable model)."},
    )
    load_in_4bit: bool = field(default=True, metadata={"help": "Use 4-bit quantization for the trainable model."})
    max_seq_length: int = field(default=8192, metadata={"help": "Context window size."})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout. Unsloth often recommends 0."})
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        metadata={"help": "Which modules to apply LoRA to."},
    )


@dataclass
class DataArguments:
    # Local JSONL datasets (your private code, replay buffers, etc.)
    train_files: List[str] = field(
        default_factory=lambda: ["my_private_code.jsonl"],
        metadata={"help": "List of local JSON/JSONL files for training (must contain a text field)."},
    )

    # Add optional public datasets (Hugging Face datasets)
    use_hf_datasets: bool = field(default=True, metadata={"help": "Whether to load additional HF datasets."})

    hf_datasets: List[str] = field(
        default_factory=lambda: [
            "bigcode/the-stack-dedup",
            "codeparrot/github-code",
            "HuggingFaceH4/CodeAlpaca_20K",
            "iamtarun/python_code_instructions_18k",
        ],
        metadata={"help": "HF datasets to concatenate. Each entry can be 'repo' or 'repo:subset'."},
    )

    hf_splits: List[str] = field(
        default_factory=lambda: ["train"],
        metadata={"help": "Dataset splits to use for HF datasets (default: ['train'])."},
    )

    text_fields: List[str] = field(
        default_factory=lambda: ["content", "text", "code", "prompt", "completion", "instruction", "output"],
        metadata={"help": "Candidate fields that might contain text/code in datasets."},
    )

    # --------------------------------------------
    # NEW: control public dataset size by private size
    # --------------------------------------------
    public_to_private_ratio: float = field(
        default=4.0,
        metadata={"help": "Target #public_samples = #private_samples * ratio. Set 0 to disable public sampling cap."},
    )
    public_sampling_strategy: str = field(
        default="random",
        metadata={"help": "How to pick public subset: 'random' or 'head'."},
    )
    public_max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Hard cap on public samples (after ratio). None means no cap."},
    )
    public_min_samples: int = field(
        default=0,
        metadata={"help": "Minimum public samples to keep (after ratio)."},
    )
    balance_across_public_sources: bool = field(
        default=False,
        metadata={"help": "If True, sample public data roughly evenly across sources instead of pooling first."},
    )

    # Existing: optionally cap TOTAL samples after final merge
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optionally cap total number of training samples after concatenation."},
    )

    # --------------------------------------------
    # NEW: optional dataset reuse across runs
    # --------------------------------------------
    processed_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "If set and exists, load processed (FIM-applied) dataset from disk and skip preprocessing."},
    )
    save_processed_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "If set, save processed (FIM-applied) dataset to this path for future reuse."},
    )

    # FIM options
    fim_rate: float = field(default=0.5, metadata={"help": "Probability of applying FIM transformation."})
    fim_spm_rate: float = field(
        default=0.0,
        metadata={"help": "Probability of using SPM (Suffix-Prefix-Middle). Keep 0 for PSM in most cases."},
    )
    min_chars: int = field(default=50, metadata={"help": "Skip samples shorter than this number of characters."})


@dataclass
class DistillArguments:
    enable_distillation: bool = field(
        default=False,
        metadata={"help": "Enable knowledge distillation to train a draft model specialized for high acceptance."},
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Teacher model name/path. If None, distillation is disabled."},
    )
    teacher_load_in_4bit: bool = field(default=True, metadata={"help": "Load teacher in 4-bit to save memory."})

    distill_alpha: float = field(default=0.3, metadata={"help": "Final loss = (1-alpha)*CE + alpha*KD."})
    distill_temperature: float = field(default=2.0, metadata={"help": "Temperature for soft targets in KD."})
    distill_prob: float = field(default=1.0, metadata={"help": "Probability of applying KD per batch."})
    teacher_device: str = field(default="cuda", metadata={"help": "Device for teacher model: cuda/cpu/mps."})


# ============================================================
# 2. FIM (Fill-In-the-Middle) Processor
# ============================================================
class FIMProcessor:
    def __init__(self, tokenizer, fim_rate: float = 0.5, min_chars: int = 50):
        self.tokenizer = tokenizer
        self.fim_rate = fim_rate
        self.min_chars = min_chars
        self.prefix_token = "<|fim_prefix|>"
        self.suffix_token = "<|fim_suffix|>"
        self.middle_token = "<|fim_middle|>"
        self.eos_token = tokenizer.eos_token or "<|endoftext|>"

    def __call__(self, examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        output_texts: List[str] = []
        for content in examples["content"]:
            if not isinstance(content, str):
                continue
            if len(content) < self.min_chars:
                continue

            if np.random.rand() < self.fim_rate:
                try:
                    doc_len = len(content)
                    middle_len = int(doc_len * np.random.uniform(0.1, 0.3))
                    middle_len = max(1, min(middle_len, doc_len - 1))
                    start = np.random.randint(0, doc_len - middle_len + 1)
                    end = start + middle_len

                    prefix = content[:start]
                    middle = content[start:end]
                    suffix = content[end:]

                    fmt_text = (
                        f"{self.prefix_token}{prefix}"
                        f"{self.suffix_token}{suffix}"
                        f"{self.middle_token}{middle}"
                        f"{self.eos_token}"
                    )
                    output_texts.append(fmt_text)
                except Exception:
                    output_texts.append(content + self.eos_token)
            else:
                output_texts.append(content + self.eos_token)

        return {"text": output_texts}


# ============================================================
# 3. Dataset Utilities (keep preprocessing intact)
# ============================================================
def _parse_repo_and_subset(name: str) -> Tuple[str, Optional[str]]:
    if ":" in name:
        repo, subset = name.split(":", 1)
        return repo.strip(), subset.strip()
    return name.strip(), None


def _extract_text_from_example(example: Dict[str, Any], text_fields: List[str]) -> Optional[str]:
    if "prompt" in example and "completion" in example:
        p, c = example.get("prompt"), example.get("completion")
        if isinstance(p, str) and isinstance(c, str):
            return p + "\n" + c

    if "instruction" in example and "output" in example:
        ins, out = example.get("instruction"), example.get("output")
        if isinstance(ins, str) and isinstance(out, str):
            return ins + "\n" + out

    for k in text_fields:
        v = example.get(k, None)
        if isinstance(v, str) and v.strip():
            return v

    return None


def _normalize_local_ds(local_ds: Dataset, text_fields: List[str]) -> Dataset:
    def normalize_local(ex):
        txt = None
        if "content" in ex and isinstance(ex["content"], str):
            txt = ex["content"]
        elif "text" in ex and isinstance(ex["text"], str):
            txt = ex["text"]
        else:
            txt = _extract_text_from_example(ex, text_fields)
        return {"content": txt if isinstance(txt, str) else ""}

    local_ds = local_ds.map(normalize_local, desc="Normalizing local dataset")
    local_ds = local_ds.filter(lambda x: isinstance(x["content"], str) and len(x["content"]) > 0)
    return local_ds


def _normalize_hf_ds(hf_ds: Dataset, text_fields: List[str], name: str, split: str) -> Dataset:
    def normalize_hf(ex):
        txt = _extract_text_from_example(ex, text_fields)
        return {"content": txt if isinstance(txt, str) else ""}

    hf_ds = hf_ds.map(normalize_hf, desc=f"Normalizing HF dataset {name}/{split}")
    hf_ds = hf_ds.filter(lambda x: isinstance(x["content"], str) and len(x["content"]) > 0)
    return hf_ds


def _sample_dataset(ds: Dataset, n: int, strategy: str, seed: int) -> Dataset:
    n = int(max(0, min(n, len(ds))))
    if n <= 0:
        return ds.select([])
    if n >= len(ds):
        return ds

    if strategy == "head":
        return ds.select(range(n))

    # default: random
    # shuffle has its own cache behavior; but this happens AFTER heavy normalize map
    # and only on the pooled public ds, so it won't invalidate your long normalize caches.
    ds = ds.shuffle(seed=seed)
    return ds.select(range(n))


def load_and_build_dataset(data_args: DataArguments, seed: int = 42) -> Dataset:
    datasets_to_concat: List[Dataset] = []

    # 1) Local JSON/JSONL files (保持不动)
    private_ds = None
    if data_args.train_files:
        logger.info(f"Loading local datasets from: {data_args.train_files}")
        local_ds = load_dataset("json", data_files=data_args.train_files, split="train")

        def normalize_local(ex):
            txt = None
            if "content" in ex and isinstance(ex["content"], str):
                txt = ex["content"]
            elif "text" in ex and isinstance(ex["text"], str):
                txt = ex["text"]
            else:
                txt = _extract_text_from_example(ex, data_args.text_fields)
            return {"content": txt if isinstance(txt, str) else ""}

        local_ds = local_ds.map(normalize_local, desc="Normalizing local dataset")
        local_ds = local_ds.filter(lambda x: isinstance(x["content"], str) and len(x["content"]) > 0)
        private_ds = local_ds
        datasets_to_concat.append(private_ds)

    # ---- NEW: compute public target BEFORE loading huge HF datasets ----
    n_private = len(private_ds) if private_ds is not None else 0
    if data_args.public_to_private_ratio and n_private > 0:
        n_public_target = int(n_private * float(data_args.public_to_private_ratio))
        if data_args.public_max_samples is not None:
            n_public_target = min(n_public_target, int(data_args.public_max_samples))
        n_public_target = max(n_public_target, int(getattr(data_args, "public_min_samples", 0)))
    else:
        n_public_target = None  # means load all (not recommended for the-stack)

    # 2) HF datasets (只改这里：按 quota slice 加载)
    if data_args.use_hf_datasets and data_args.hf_datasets:
        # 简单策略：把 public quota 均分到每个 (dataset, split)
        num_sources = len(data_args.hf_datasets) * len(data_args.hf_splits)
        per_source = None
        if n_public_target is not None and num_sources > 0:
            per_source = max(1, n_public_target // num_sources)

        for name in data_args.hf_datasets:
            repo, subset = _parse_repo_and_subset(name)
            for split in data_args.hf_splits:
                quota = per_source

                # 关键：只加载 train[:quota]，避免全量 the-stack normalize
                if quota is not None:
                    split_spec = f"{split}[:{quota}]"
                else:
                    split_spec = split  # 不限量（慎用）

                logger.info(f"Loading HF dataset: repo={repo}, subset={subset}, split={split_spec}")
                hf_ds = load_dataset(repo, subset, split=split_spec)

                def normalize_hf(ex):
                    txt = _extract_text_from_example(ex, data_args.text_fields)
                    return {"content": txt if isinstance(txt, str) else ""}

                hf_ds = hf_ds.map(normalize_hf, desc=f"Normalizing HF dataset {name}/{split_spec}")
                hf_ds = hf_ds.filter(lambda x: isinstance(x["content"], str) and len(x["content"]) > 0)
                datasets_to_concat.append(hf_ds)

    if not datasets_to_concat:
        raise ValueError("No datasets loaded. Provide --train_files and/or --hf_datasets.")

    logger.info(f"Concatenating {len(datasets_to_concat)} datasets...")
    merged = concatenate_datasets(datasets_to_concat)

    if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
        merged = merged.select(range(min(len(merged), data_args.max_train_samples)))

    logger.info(f"Final merged dataset size: {len(merged)}")
    return merged


# ============================================================
# 4. Knowledge Distillation Trainer
# ============================================================
class DistillSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        teacher_model=None,
        distill_alpha: float = 0.3,
        distill_temperature: float = 2.0,
        distill_prob: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_alpha = float(distill_alpha)
        self.distill_temperature = float(distill_temperature)
        self.distill_prob = float(distill_prob)

        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        ce_loss = outputs.loss

        if (self.teacher_model is None) or (self.distill_alpha <= 0.0):
            return (ce_loss, outputs) if return_outputs else ce_loss

        if self.distill_prob < 1.0 and np.random.rand() > self.distill_prob:
            return (ce_loss, outputs) if return_outputs else ce_loss

        with torch.no_grad():
            teacher_out = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
            )

        student_logits = outputs.logits
        teacher_logits = teacher_out.logits

        T = self.distill_temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        labels = inputs.get("labels", None)
        if labels is not None:
            mask = (labels != -100).unsqueeze(-1)
            kd = F.kl_div(student_log_probs, teacher_probs, reduction="none")
            kd = (kd * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

        kd_loss = kd * (T * T)
        loss = (1.0 - self.distill_alpha) * ce_loss + self.distill_alpha * kd_loss

        return (loss, outputs) if return_outputs else loss


# ============================================================
# 5. Main
# ============================================================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, DistillArguments, TrainingArguments))
    model_args, data_args, distill_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    logger.info(f"ModelArguments: {model_args}")
    logger.info(f"DataArguments: {data_args}")
    logger.info(f"DistillArguments: {distill_args}")
    logger.info(f"TrainingArguments: {training_args}")

    # 1) Load processed dataset if exists (NEW)
    processed_dataset = None
    if data_args.processed_dataset_path and os.path.isdir(data_args.processed_dataset_path):
        logger.info(f"Loading processed dataset from disk: {data_args.processed_dataset_path}")
        processed_dataset = Dataset.load_from_disk(data_args.processed_dataset_path)

    if processed_dataset is None:
        # 2) Load raw datasets (normalize/filter) - same logic, but now with public cap
        dataset = load_and_build_dataset(data_args, seed=training_args.seed)

        # 3) Load student model
        logger.info("Loading trainable model (student) with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_args.model_name_or_path,
            max_seq_length=model_args.max_seq_length,
            dtype=None,
            load_in_4bit=model_args.load_in_4bit,
        )

        # 4) Configure LoRA
        logger.info("Configuring LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=model_args.lora_r,
            target_modules=model_args.target_modules,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=training_args.seed,
        )

        # 5) FIM processing (same as before)
        logger.info(f"Applying FIM transformation with fim_rate={data_args.fim_rate}...")
        fim_processor = FIMProcessor(tokenizer, fim_rate=data_args.fim_rate, min_chars=data_args.min_chars)

        processed_dataset = dataset.map(
            fim_processor,
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
            desc="Applying FIM transformation",
        )

        # Optional save for reuse (NEW)
        if data_args.save_processed_dataset_path:
            os.makedirs(os.path.dirname(data_args.save_processed_dataset_path), exist_ok=True)
            logger.info(f"Saving processed dataset to: {data_args.save_processed_dataset_path}")
            processed_dataset.save_to_disk(data_args.save_processed_dataset_path)
    else:
        # If we loaded processed dataset, we still need model/tokenizer
        logger.info("Loading trainable model (student) with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_args.model_name_or_path,
            max_seq_length=model_args.max_seq_length,
            dtype=None,
            load_in_4bit=model_args.load_in_4bit,
        )

        logger.info("Configuring LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=model_args.lora_r,
            target_modules=model_args.target_modules,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=training_args.seed,
        )

    # 6) Load teacher model (optional)
    teacher_model = None
    if distill_args.enable_distillation:
        if not distill_args.teacher_model_name_or_path:
            raise ValueError("Distillation enabled but --teacher_model_name_or_path is not provided.")
        logger.info("Loading teacher model for distillation (frozen)...")
        teacher_model, _teacher_tokenizer = FastLanguageModel.from_pretrained(
            model_name=distill_args.teacher_model_name_or_path,
            max_seq_length=model_args.max_seq_length,
            dtype=None,
            load_in_4bit=distill_args.teacher_load_in_4bit,
        )

        if _teacher_tokenizer.get_vocab() != tokenizer.get_vocab():
            logger.warning(
                "Teacher and student tokenizers appear different. "
                "For best results, ensure they share the same tokenizer/vocab."
            )

        teacher_model.eval()
        if distill_args.teacher_device:
            teacher_model.to(distill_args.teacher_device)

    # 7) Trainer
    trainer_cls = DistillSFTTrainer if (teacher_model is not None) else SFTTrainer

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed_dataset,
        dataset_text_field="text",
        max_seq_length=model_args.max_seq_length,
        packing=True,
        args=training_args,
    )

    if trainer_cls is DistillSFTTrainer:
        trainer_kwargs.update(
            dict(
                teacher_model=teacher_model,
                distill_alpha=distill_args.distill_alpha,
                distill_temperature=distill_args.distill_temperature,
                distill_prob=distill_args.distill_prob,
            )
        )

    trainer = trainer_cls(**trainer_kwargs)

    # 8) Train
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    logger.info(f"Training finished. Stats: {trainer_stats}")

    # 9) Save
    logger.info("Saving model + tokenizer...")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Done. Output saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()

"""
#public = private × 4
## codeparrot/github-code \
    
python cpt_train_coder_v2_1.py \
  --model_name_or_path unsloth/Qwen2.5-Coder-7B \
  --train_files my_private_code.jsonl \
  --use_hf_datasets True \
  --hf_datasets bigcode/the-stack-dedup \ 
  --public_to_private_ratio 4.0 \
  --public_sampling_strategy random \
  --output_dir out_cpt \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 500

#option2: FIM first
python cpt_trainv3.py \
  --model_name_or_path unsloth/Qwen2.5-Coder-7B \
  --train_files my_private_code.jsonl \
  --use_hf_datasets True \
  --hf_datasets bigcode/the-stack-dedup codeparrot/github-code \
  --public_to_private_ratio 4.0 \
  --save_processed_dataset_path cache/processed_fim_ds \
  --output_dir out_cpt_7b \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 1

python cpt_trainv3.py \
  --model_name_or_path unsloth/Qwen2.5-Coder-1.5B \
  --processed_dataset_path cache/processed_fim_ds \
  --output_dir out_cpt_1p5b \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 8e-5 \
  --num_train_epochs 1

#add teacher
python cpt_trainv3.py \
  --model_name_or_path unsloth/Qwen2.5-Coder-1.5B \
  --processed_dataset_path cache/processed_fim_ds \
  --enable_distillation True \
  --teacher_model_name_or_path unsloth/Qwen2.5-Coder-7B \
  --distill_alpha 0.3 \
  --distill_temperature 2.0 \
  --distill_prob 0.5 \
  --output_dir out_draft_kd
"""