import os
import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import TrainingArguments, HfArgumentParser, set_seed, TrainerCallback

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
            # Code-heavy / general instruction sources (adjust to your taste)
            # NOTE: Choose datasets you are licensed to use.
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

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optionally cap total number of training samples after concatenation."},
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
    # Knowledge distillation for coder-specialized draft model
    enable_distillation: bool = field(
        default=False,
        metadata={"help": "Enable knowledge distillation to train a draft model specialized for high acceptance."},
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Teacher model name/path. If None, distillation is disabled."},
    )
    teacher_load_in_4bit: bool = field(default=True, metadata={"help": "Load teacher in 4-bit to save memory."})

    # Distillation weighting
    distill_alpha: float = field(
        default=0.3,
        metadata={
            "help": "Weight of the distillation loss. Final loss = (1-alpha)*CE + alpha*KD."
        },
    )
    distill_temperature: float = field(default=2.0, metadata={"help": "Temperature for soft targets in KD."})

    # To reduce overhead, distill on a fraction of steps
    distill_prob: float = field(
        default=1.0,
        metadata={"help": "Probability of applying KD per batch (1.0 = every batch)."},
    )

    # Optional: keep teacher on GPU or offload
    teacher_device: str = field(default="cuda", metadata={"help": "Device for teacher model: cuda/cpu/mps."})


# ============================================================
# 2. FIM (Fill-In-the-Middle) Processor
# ============================================================
class FIMProcessor:
    """
    Applies Fill-In-the-Middle (FIM) transformation for code pretraining.
    Qwen2.5-Coder FIM tokens:
      <|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>
    """

    def __init__(self, tokenizer, fim_rate: float = 0.5, min_chars: int = 50):
        self.tokenizer = tokenizer
        self.fim_rate = fim_rate
        self.min_chars = min_chars

        # Qwen FIM special tokens (usually already present in tokenizer)
        self.prefix_token = "<|fim_prefix|>"
        self.suffix_token = "<|fim_suffix|>"
        self.middle_token = "<|fim_middle|>"
        self.eos_token = tokenizer.eos_token or "<|endoftext|>"

    def __call__(self, examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        # Expect a "content" field already normalized upstream to "content"
        output_texts: List[str] = []
        for content in examples["content"]:
            if not isinstance(content, str):
                continue
            if len(content) < self.min_chars:
                continue

            if np.random.rand() < self.fim_rate:
                # Standard FIM: choose a random middle span; reorder as P + S + M.
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
                    # Fallback to standard causal LM format
                    output_texts.append(content + self.eos_token)
            else:
                # Standard causal LM
                output_texts.append(content + self.eos_token)

        return {"text": output_texts}


# ============================================================
# 3. Dataset Utilities
# ============================================================
def _parse_repo_and_subset(name: str) -> Tuple[str, Optional[str]]:
    """
    Parse 'repo' or 'repo:subset' string.
    """
    if ":" in name:
        repo, subset = name.split(":", 1)
        return repo.strip(), subset.strip()
    return name.strip(), None


def _extract_text_from_example(example: Dict[str, Any], text_fields: List[str]) -> Optional[str]:
    """
    Try to extract a usable text/code string from an arbitrary dataset row.
    - If an entry has {prompt, completion}, we join them.
    - If an entry has {instruction, output}, we join them.
    - Otherwise find the first string field in text_fields.
    """
    # Pair fields first (common in instruction datasets)
    if "prompt" in example and "completion" in example:
        p, c = example.get("prompt"), example.get("completion")
        if isinstance(p, str) and isinstance(c, str):
            return p + "\n" + c

    if "instruction" in example and "output" in example:
        ins, out = example.get("instruction"), example.get("output")
        if isinstance(ins, str) and isinstance(out, str):
            return ins + "\n" + out

    # Single field fallback
    for k in text_fields:
        v = example.get(k, None)
        if isinstance(v, str) and v.strip():
            return v

    return None


def load_and_build_dataset(data_args: DataArguments) -> Dataset:
    """
    Load local datasets + (optional) HF datasets, normalize to a single 'content' column,
    and return a concatenated dataset.
    """
    datasets_to_concat: List[Dataset] = []

    # 1) Local JSON/JSONL files
    if data_args.train_files:
        logger.info(f"Loading local datasets from: {data_args.train_files}")
        local_ds = load_dataset("json", data_files=data_args.train_files, split="train")

        # Normalize to content
        def normalize_local(ex):
            # Prefer 'content' if it exists; else 'text'; else try to infer
            txt = None
            if "content" in ex and isinstance(ex["content"], str):
                txt = ex["content"]
            elif "text" in ex and isinstance(ex["text"], str):
                txt = ex["text"]
            else:
                txt = _extract_text_from_example(ex, data_args.text_fields)

            return {"content": txt if isinstance(txt, str) else ""}

        local_ds = local_ds.map(normalize_local, desc="Normalizing local dataset")
        # Filter empty
        local_ds = local_ds.filter(lambda x: isinstance(x["content"], str) and len(x["content"]) > 0)
        datasets_to_concat.append(local_ds)

    # 2) HF datasets
    if data_args.use_hf_datasets and data_args.hf_datasets:
        for name in data_args.hf_datasets:
            repo, subset = _parse_repo_and_subset(name)
            for split in data_args.hf_splits:
                logger.info(f"Loading HF dataset: repo={repo}, subset={subset}, split={split}")
                hf_ds = load_dataset(repo, subset, split=split)

                def normalize_hf(ex):
                    txt = _extract_text_from_example(ex, data_args.text_fields)
                    return {"content": txt if isinstance(txt, str) else ""}

                hf_ds = hf_ds.map(normalize_hf, desc=f"Normalizing HF dataset {name}/{split}")
                hf_ds = hf_ds.filter(lambda x: isinstance(x["content"], str) and len(x["content"]) > 0)
                datasets_to_concat.append(hf_ds)

    if not datasets_to_concat:
        raise ValueError("No datasets loaded. Provide --train_files and/or --hf_datasets.")

    logger.info(f"Concatenating {len(datasets_to_concat)} datasets...")
    merged = concatenate_datasets(datasets_to_concat)

    if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
        logger.info(f"Subsampling to max_train_samples={data_args.max_train_samples}")
        merged = merged.select(range(min(len(merged), data_args.max_train_samples)))

    logger.info(f"Final merged dataset size: {len(merged)}")
    return merged


# ============================================================
# 4. Knowledge Distillation Trainer (Acceptance-Oriented Draft Training)
# ============================================================
class DistillSFTTrainer(SFTTrainer):
    """
    SFTTrainer + Knowledge Distillation:
      loss = (1 - alpha) * CE(student) + alpha * KD(student || teacher)

    KD uses KL-divergence between teacher and student distributions on the same tokens.

    This is useful if you train a smaller "draft" model to match the larger teacher,
    increasing speculative-decoding acceptance rate in deployment.
    """

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
        # Standard CE loss from the underlying model (TRL handles label shifting)
        outputs = model(**inputs)
        ce_loss = outputs.loss

        # If distillation is disabled or skipped for this batch
        if (self.teacher_model is None) or (self.distill_alpha <= 0.0):
            return (ce_loss, outputs) if return_outputs else ce_loss

        if self.distill_prob < 1.0 and np.random.rand() > self.distill_prob:
            return (ce_loss, outputs) if return_outputs else ce_loss

        with torch.no_grad():
            teacher_out = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
            )

        # KD on logits
        student_logits = outputs.logits
        teacher_logits = teacher_out.logits

        T = self.distill_temperature
        # Soft targets: KL(teacher || student) in log-prob space
        # kl_div expects: input=log_probs, target=probs
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        # Mask padding tokens (if labels exist)
        # TRL SFTTrainer typically provides labels with -100 for ignore positions.
        labels = inputs.get("labels", None)
        if labels is not None:
            mask = (labels != -100).unsqueeze(-1)  # [B, L, 1]
            # Compute KL only on valid tokens
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

    # 1) Load datasets
    dataset = load_and_build_dataset(data_args)

    # 2) Load student model (trainable)
    logger.info("Loading trainable model (student) with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=model_args.max_seq_length,
        dtype=None,  # auto
        load_in_4bit=model_args.load_in_4bit,
    )

    # 3) Configure LoRA
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

    # 4) FIM processing
    logger.info(f"Applying FIM transformation with fim_rate={data_args.fim_rate}...")
    fim_processor = FIMProcessor(tokenizer, fim_rate=data_args.fim_rate, min_chars=data_args.min_chars)

    processed_dataset = dataset.map(
        fim_processor,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Applying FIM transformation",
    )

    # 5) Load teacher model (optional, for distillation)
    teacher_model = None
    if distill_args.enable_distillation:
        if not distill_args.teacher_model_name_or_path:
            raise ValueError(
                "Distillation enabled but --teacher_model_name_or_path is not provided."
            )
        logger.info("Loading teacher model for distillation (frozen)...")
        teacher_model, _teacher_tokenizer = FastLanguageModel.from_pretrained(
            model_name=distill_args.teacher_model_name_or_path,
            max_seq_length=model_args.max_seq_length,
            dtype=None,
            load_in_4bit=distill_args.teacher_load_in_4bit,
        )

        # Important: tokenizer compatibility matters for draft/teacher distillation.
        # If tokenizers differ, distillation may be incorrect.
        if _teacher_tokenizer.get_vocab() != tokenizer.get_vocab():
            logger.warning(
                "Teacher and student tokenizers appear different. "
                "For best results, ensure they share the same tokenizer/vocab."
            )

        teacher_model.eval()
        # Move teacher to desired device if possible
        if distill_args.teacher_device:
            teacher_model.to(distill_args.teacher_device)

    # 6) Trainer
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

    # 7) Train
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    logger.info(f"Training finished. Stats: {trainer_stats}")

    # 8) Save
    logger.info("Saving model + tokenizer...")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    logger.info(f"Done. Output saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()

"""
Added a distillation option (DistillArguments) and a DistillSFTTrainer that combines CE loss + KD loss to train a coder-specialized draft model (student) to match a teacher—this is the most direct path toward higher speculative-decoding acceptance.

#Plain CPT/SFT with FIM on local + HF datasets
python cpt_trainv2.py \
  --model_name_or_path unsloth/Qwen2.5-Coder-7B \
  --train_files my_private_code.jsonl \
  --use_hf_datasets True \
  --hf_datasets bigcode/the-stack-dedup codeparrot/github-code \
  --fim_rate 0.5 \
  --output_dir out_cpt \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 500

#Train a draft model with KD to maximize acceptance
python cpt_trainv2.py \
  --model_name_or_path unsloth/Qwen2.5-Coder-1.5B \
  --enable_distillation True \
  --teacher_model_name_or_path unsloth/Qwen2.5-Coder-7B \
  --distill_alpha 0.3 \
  --distill_temperature 2.0 \
  --distill_prob 0.5 \
  --train_files my_private_code.jsonl \
  --use_hf_datasets True \
  --hf_datasets bigcode/the-stack-dedup HuggingFaceH4/CodeAlpaca_20K \
  --fim_rate 0.5 \
  --output_dir out_draft_kd \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 500
  

"""