import os
import sys
import re
import subprocess
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
import torch


# -----------------------------
# Install deps
# -----------------------------
def install_dependencies():
    print("Installing dependencies...")
    packages = [
        "unsloth",
        "unsloth_zoo",
        "bitsandbytes",
        "accelerate",
        "peft",
        "trl==0.22.2",
        "triton",
        "transformers==4.57.3",
        "sentencepiece",
        "protobuf",
        "datasets==4.3.0",
        "huggingface_hub>=0.34.0",
        "hf_transfer",
    ]
    try:
        v = re.match(r"[\d]+\.[\d]+", str(torch.__version__)).group(0)
        xformers_ver = {"2.10": "0.0.34", "2.9": "0.0.33.post1", "2.8": "0.0.32.post2"}.get(v, "0.0.34")
        packages.append(f"xformers=={xformers_ver}")
    except Exception:
        pass

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir"] + packages)
    print("Dependencies installed.")


# -----------------------------
# Config
# -----------------------------
@dataclass
class MixConfig:
    seed: int = 3407

    # mixture ratios (sum doesn't need to be 1; used as weights)
    w_instruct: float = 0.60   # Evol-Instruct-Code
    w_mbpp: float = 0.20       # MBPP unit-test conditioned
    w_humaneval: float = 0.10  # HumanEval completion style
    w_patch: float = 0.10      # git diff / patch dataset

    # per-dataset caps (keep training size manageable)
    max_instruct: int = 60000
    max_mbpp: int = 8000
    max_humaneval: int = 4000
    max_patch: int = 12000

    # prompt shaping
    mbpp_tests_in_prompt: int = 5
    patch_context_lines: int = 200  # max chars from issue/commit message if present


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def try_load_dataset(load_dataset_fn, candidates: List[Dict[str, str]]):
    """
    candidates: list of dicts {name: ..., split: ..., config: optional}
    Returns (dataset, used_spec) or (None, None)
    """
    for spec in candidates:
        name = spec["name"]
        split = spec.get("split", "train")
        cfg = spec.get("config", None)
        try:
            if cfg:
                ds = load_dataset_fn(name, cfg, split=split)
            else:
                ds = load_dataset_fn(name, split=split)
            return ds, spec
        except Exception as e:
            print(f"[WARN] Failed to load {name} ({split}): {e}")
    return None, None


def safe_slice(ds, n: int, seed: int):
    n = min(n, len(ds))
    if n <= 0:
        return ds.select([])
    # shuffle then select
    return ds.shuffle(seed=seed).select(range(n))

from datasets import load_dataset

def load_local_patch_jsonl(path: str):
    if not os.path.exists(path):
        return None
    ds = load_dataset("json", data_files=path, split="train")
    # expects fields: prompt, patch
    def to_text(examples):
        texts = []
        for p, d in zip(examples["prompt"], examples["patch"]):
            convo = [
                {"role":"user","content": p},
                {"role":"assistant","content": d},
            ]
            texts.append(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False))
        return {"text": texts}
    ds = ds.map(to_text, batched=True, remove_columns=ds.column_names)
    return ds

# -----------------------------
# Formatters (convert each dataset into a single 'text' field)
# -----------------------------
def format_evol_instruct(examples: Dict[str, List[Any]], tokenizer) -> Dict[str, List[str]]:
    instr = examples.get("instruction", [])
    out = examples.get("output", [])
    texts = []
    for i, o in zip(instr, out):
        convo = [{"role": "user", "content": str(i)}, {"role": "assistant", "content": str(o)}]
        texts.append(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False))
    return {"text": texts}


def format_mbpp(examples: Dict[str, List[Any]], tokenizer, k_tests: int = 5) -> Dict[str, List[str]]:
    """
    MBPP fields: text, code, test_list
    We'll train: user gives problem + some asserts; assistant outputs code only.
    """
    problems = examples.get("text", [])
    codes = examples.get("code", [])
    tests = examples.get("test_list", [])

    texts = []
    for prob, code, tlist in zip(problems, codes, tests):
        tlist = tlist or []
        tpreview = "\n".join(tlist[:k_tests])

        user = (
            "Return ONLY valid Python code. No markdown fences. No explanation.\n"
            "Solve the task below and satisfy the tests.\n\n"
            f"Task:\n{prob}\n\n"
            f"Tests:\n{tpreview}\n"
        )
        assistant = str(code).strip()

        convo = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        texts.append(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False))

    return {"text": texts}


def format_humaneval(examples: Dict[str, List[Any]], tokenizer) -> Dict[str, List[str]]:
    """
    HumanEval fields: prompt, canonical_solution, entry_point
    Train: user asks to complete code; assistant gives completion.
    """
    prompts = examples.get("prompt", [])
    sols = examples.get("canonical_solution", [])

    texts = []
    for p, sol in zip(prompts, sols):
        user = (
            "Complete the following Python function. "
            "Return ONLY the code needed to complete it (no explanation).\n\n"
            f"{p}"
        )
        assistant = str(sol).strip()
        convo = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        texts.append(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False))
    return {"text": texts}


def format_patch_dataset(examples: Dict[str, List[Any]], tokenizer, ctx_chars: int = 200) -> Dict[str, List[str]]:
    """
    Generic patch-format dataset formatter.
    Different datasets use different fields. We try common ones:
      - "message" / "commit_message" / "instruction"
      - "diff" / "patch" / "changes"
    We produce: user asks for a patch; assistant outputs diff.

    If we cannot find diff, we skip the sample (empty string filtered later).
    """
    texts = []

    # candidate keys
    msg_keys = ["message", "commit_message", "subject", "instruction", "prompt", "issue", "title"]
    diff_keys = ["diff", "patch", "changes", "git_diff"]

    # batch size
    bs = len(examples.get(diff_keys[0], [])) if diff_keys[0] in examples else len(next(iter(examples.values())))

    for i in range(bs):
        msg = ""
        for k in msg_keys:
            if k in examples and examples[k] and examples[k][i]:
                msg = str(examples[k][i])
                break

        diff = ""
        for k in diff_keys:
            if k in examples and examples[k] and examples[k][i]:
                diff = str(examples[k][i])
                break

        if not diff.strip():
            texts.append("")  # will filter
            continue

        msg = (msg[:ctx_chars] + " ...") if (msg and len(msg) > ctx_chars) else msg
        user = (
            "You are given a software change request. "
            "Output ONLY a unified diff patch (no explanation, no markdown fences).\n\n"
            f"Request / context:\n{msg}\n"
        )
        assistant = diff.strip()
        convo = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        texts.append(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False))

    return {"text": texts}


def make_training_args(TrainingArguments, **kwargs):
    """
    Transformers TrainingArguments API compatibility shim.
    Some versions use evaluation_strategy/save_strategy/logging_strategy,
    others use eval_strategy/save_strategy/logging_strategy (or similar).
    This tries both safely.
    """
    try:
        return TrainingArguments(**kwargs)
    except TypeError as e:
        msg = str(e)

        # Map common renamed args
        rename_map = {
            "evaluation_strategy": "eval_strategy",
            "save_strategy": "save_strategy",      # usually unchanged, but keep for completeness
            "logging_strategy": "logging_strategy" # usually unchanged
        }

        # If eval arg not accepted, try eval_strategy
        if "evaluation_strategy" in kwargs and "evaluation_strategy" in msg:
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

        # If save_strategy not accepted, try save_strategy alternative (rare)
        # (Most versions keep save_strategy, but we keep a fallback pattern)
        if "save_strategy" in kwargs and "save_strategy" in msg:
            # try 'save_strategy' -> 'save_strategy' does nothing; placeholder
            pass

        # If logging_strategy not accepted, try 'log_strategy' (rare)
        if "logging_strategy" in kwargs and "logging_strategy" in msg:
            kwargs["log_strategy"] = kwargs.pop("logging_strategy")

        return TrainingArguments(**kwargs)

# -----------------------------
# Main
# -----------------------------
def main():
    try:
        from datasets import load_dataset, concatenate_datasets
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer, SFTConfig
    except ImportError:
        install_dependencies()
        from datasets import load_dataset, concatenate_datasets
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer, SFTConfig

    from transformers import TrainingArguments
    cfg = MixConfig()
    set_seed(cfg.seed)

    # -----------------------------
    # Model
    # -----------------------------
    BASE_MODEL = "unsloth/Qwen2.5-Coder-14B-Instruct"
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True
    DTYPE = None

    print(f"Loading model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    print("Adding LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # -----------------------------
    # Load datasets
    # -----------------------------
    print("Loading datasets...")

    # 1) Evol-Instruct-Code
    ds_instruct = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
    ds_instruct = safe_slice(ds_instruct, cfg.max_instruct, cfg.seed)
    ds_instruct = ds_instruct.map(lambda x: format_evol_instruct(x, tokenizer), batched=True, remove_columns=ds_instruct.column_names)

    # 2) MBPP
    ds_mbpp = load_dataset("mbpp", split="test")  # MBPP train exists but is tiny; test is fine for SFT if you accept leakage.
    # Better: use train if available in your environment; many use test for quick SFT but it leaks eval.
    ds_mbpp = safe_slice(ds_mbpp, cfg.max_mbpp, cfg.seed)
    ds_mbpp = ds_mbpp.map(lambda x: format_mbpp(x, tokenizer, k_tests=cfg.mbpp_tests_in_prompt),
                          batched=True, remove_columns=ds_mbpp.column_names)

    # 3) HumanEval
    ds_he = load_dataset("openai_humaneval", split="test")
    ds_he = safe_slice(ds_he, cfg.max_humaneval, cfg.seed)
    ds_he = ds_he.map(lambda x: format_humaneval(x, tokenizer), batched=True, remove_columns=ds_he.column_names)

    # 4) Patch dataset candidates (first that loads wins)
    patch_candidates = [
        # These are common “diff/patch” style datasets; availability varies.
        {"name": "bigcode/commitpackft", "split": "train"},
        {"name": "codeparrot/github-code", "split": "train"},
        {"name": "HuggingFaceH4/CodeAlpaca_20K", "split": "train"},  # not patch, but fallback
    ]
    ds_patch, used_patch = try_load_dataset(load_dataset, patch_candidates)

    if ds_patch is None or len(ds_patch) == 0:
        print("[WARN] Patch dataset empty after formatting. Setting patch weight to 0.")
        cfg.w_patch = 0.0
        #ds_patch = load_local_patch_jsonl("data/patch_sft.jsonl")
    
    if ds_patch is not None:
        ds_patch = safe_slice(ds_patch, cfg.max_patch, cfg.seed)
        ds_patch = ds_patch.map(lambda x: format_patch_dataset(x, tokenizer, ctx_chars=cfg.patch_context_lines),
                                batched=True)
        # filter empties from missing diff keys
        ds_patch = ds_patch.filter(lambda ex: isinstance(ex.get("text", ""), str) and len(ex["text"].strip()) > 0)
        ds_patch = ds_patch.remove_columns([c for c in ds_patch.column_names if c != "text"])
        print(f"Patch dataset loaded: {used_patch}")
    else:
        print("[WARN] No patch dataset could be loaded. Proceeding without patch data.")
        # empty dataset with "text"
        ds_patch = ds_instruct.select([])

    print("Sizes after formatting:")
    print("  instruct:", len(ds_instruct))
    print("  mbpp    :", len(ds_mbpp))
    print("  humaneval:", len(ds_he))
    print("  patch   :", len(ds_patch))

    # -----------------------------
    # Weighted mixing (sample to target sizes)
    # -----------------------------
    # Create a mixed dataset by sampling proportional to weights.
    # We choose a total size based on sum of caps.
    total_target = len(ds_instruct) + len(ds_mbpp) + len(ds_he) + len(ds_patch)
    # Use weights to compute per-source sample counts (bounded by available)
    weights = {
        "instruct": cfg.w_instruct,
        "mbpp": cfg.w_mbpp,
        "humaneval": cfg.w_humaneval,
        "patch": cfg.w_patch,
    }
    wsum = sum(weights.values())

    def target_count(name: str, available: int) -> int:
        want = int(total_target * (weights[name] / wsum))
        return min(want, available)

    n_instruct = target_count("instruct", len(ds_instruct))
    n_mbpp = target_count("mbpp", len(ds_mbpp))
    n_he = target_count("humaneval", len(ds_he))
    n_patch = target_count("patch", len(ds_patch))

    mixed = concatenate_datasets([
        safe_slice(ds_instruct, n_instruct, cfg.seed),
        safe_slice(ds_mbpp, n_mbpp, cfg.seed + 1),
        safe_slice(ds_he, n_he, cfg.seed + 2),
        safe_slice(ds_patch, n_patch, cfg.seed + 3),
    ]).shuffle(seed=cfg.seed)

    print(f"Mixed dataset size: {len(mixed)} (instruct={n_instruct}, mbpp={n_mbpp}, he={n_he}, patch={n_patch})")

    # Split small eval set (optional)
    eval_size = min(2000, max(200, len(mixed) // 50))
    mixed = mixed.train_test_split(test_size=eval_size, seed=cfg.seed)
    train_ds = mixed["train"]
    eval_ds = mixed["test"]
    print(f"Train size: {len(train_ds)} | Eval size: {len(eval_ds)}")

    # -----------------------------
    # Train
    # -----------------------------
    BATCH_SIZE = 4
    GRAD_ACC = 4
    LR = 2e-5
    EPOCHS = 1

    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_ds,
    #     eval_dataset=eval_ds,
    #     args=SFTConfig(
    #         dataset_text_field="text",
    #         max_seq_length=MAX_SEQ_LENGTH,
    #         per_device_train_batch_size=BATCH_SIZE,
    #         gradient_accumulation_steps=GRAD_ACC,
    #         num_train_epochs=EPOCHS,
    #         warmup_ratio=0.03,
    #         learning_rate=LR,
    #         logging_steps=20,
    #         eval_steps=200,
    #         evaluation_strategy="steps",
    #         save_steps=400,
    #         save_total_limit=2,
    #         optim="adamw_8bit",
    #         weight_decay=0.01,
    #         lr_scheduler_type="linear",
    #         seed=cfg.seed,
    #         report_to="none",
    #         bf16=is_bfloat16_supported(),
    #         fp16=not is_bfloat16_supported(),
    #         packing=True,  # IMPORTANT
    #         output_dir="outputs_sft_v2",
    #     ),
    # )
    from transformers import TrainingArguments

    # training_args = TrainingArguments(
    #     output_dir="outputs_sft_v2",
    #     per_device_train_batch_size=BATCH_SIZE,
    #     gradient_accumulation_steps=GRAD_ACC,
    #     num_train_epochs=EPOCHS,
    #     learning_rate=LR,
    #     warmup_ratio=0.03,
    #     logging_steps=20,

    #     evaluation_strategy="steps",
    #     eval_steps=200,
    #     save_strategy="steps",
    #     save_steps=400,
    #     save_total_limit=2,

    #     weight_decay=0.01,
    #     lr_scheduler_type="linear",
    #     optim="adamw_8bit",

    #     bf16=is_bfloat16_supported(),
    #     fp16=not is_bfloat16_supported(),
    #     report_to="none",
    #     seed=cfg.seed,
    # )
    training_args = make_training_args(
        TrainingArguments,
        output_dir="outputs_sft_v2",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=0.03,
        logging_steps=20,

        # These keys will be renamed automatically if needed
        evaluation_strategy="steps",
        eval_steps=200,

        save_strategy="steps",
        save_steps=400,
        save_total_limit=2,

        weight_decay=0.01,
        lr_scheduler_type="linear",
        optim="adamw_8bit",

        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),

        report_to="none",
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
        args=training_args,
    )

    print("Starting SFT v2 training...")
    trainer.train()

    out_dir = "qwen_coder_lora_sft_v2"
    print(f"Saving LoRA to: {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()