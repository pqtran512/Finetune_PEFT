"""Plan B — LoRA trên base bf16 (KHÔNG QLoRA 4-bit).

Khác với baseline:
- Base load bf16 trực tiếp (bỏ BitsAndBytesConfig 4-bit).
- attn_implementation="flash_attention_2"
- LoRA r=64, alpha=128
- neftune_noise_alpha=5
- num_train_epochs=3
- learning_rate=8e-5 (hạ vs 1e-4 vì bf16 LoRA cần update mạnh ít hơn)
- DataCollatorForSeq2Seq (packing chưa bật để giữ flow đơn giản — có thể thêm sau).

Tài nguyên dự kiến: VRAM 32-42GB, wall-clock 10-14h trên H100 80GB.
"""
import os
import time
import torch
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint


def _load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_env(Path(__file__).parent / ".env")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
if WANDB_API_KEY:
    import wandb
    wandb.login(key=WANDB_API_KEY)


def main():
    MODEL_ID = "Qwen/CodeQwen1.5-7B"
    OUTPUT_DIR = "./models/codeqwen/planB"
    DATA_PATH = str(Path(__file__).parent / "data" / "java_completion_train.jsonl")
    MAX_LENGTH = 1536
    CACHE_DIR = os.environ.get(
        "HF_HOME",
        str(Path.home() / ".cache" / "huggingface"),
    )

    if WANDB_API_KEY:
        import wandb
        wandb.init(project="codeqwen-java", name="planB-bf16-lora-r64")

    print("--- Loading dataset ---")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Không thấy {DATA_PATH}. Chạy `python build_completion_dataset.py` trước.")
    start = time.time()
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"Load: {time.time() - start:.2f}s, n={len(dataset)}")

    print("--- Loading tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_with_mask(ex):
        prefix_ids = tokenizer(ex["prefix"], add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(ex["target"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        if len(prefix_ids) + len(target_ids) > MAX_LENGTH:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids = prefix_ids + target_ids
        labels = [-100] * len(prefix_ids) + target_ids
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    split = dataset.train_test_split(test_size=0.02, seed=42)
    print("--- Tokenizing ---")
    train_ds = split["train"].map(tokenize_with_mask, remove_columns=split["train"].column_names, num_proc=4)
    eval_ds = split["test"].map(tokenize_with_mask, remove_columns=split["test"].column_names, num_proc=4)

    def has_target(ex):
        return any(l != -100 for l in ex["labels"])

    train_ds = train_ds.filter(has_target)
    eval_ds = eval_ds.filter(has_target)
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    s = train_ds[0]
    print(f"Sample 0: total={len(s['input_ids'])}, prefix_masked={sum(1 for l in s['labels'] if l == -100)}, target_tokens={sum(1 for l in s['labels'] if l != -100)}")

    print("--- Loading model (bf16, FA2) ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    effective_batch = 1 * 32
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    eval_steps = max(50, int(0.1 * steps_per_epoch))
    print(f"Steps/epoch: {steps_per_epoch}, eval_steps: {eval_steps}")

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=8e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.01,
            optim="paged_adamw_8bit",
            bf16=True,
            tf32=True,
            neftune_noise_alpha=5,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=10,
            dataloader_num_workers=4,
            report_to="wandb" if WANDB_API_KEY else "none",
            run_name="planB-bf16-lora-r64",
        ),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    print("--- Starting training ---")
    checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    print(f"Checkpoint: {checkpoint}")

    start = time.time()
    trainer.train(resume_from_checkpoint=checkpoint)
    print(f"Training: {(time.time() - start) / 60:.2f} min")

    trainer.save_model(OUTPUT_DIR)
    print(f"Done. Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
