"""Qwen2.5-Coder-32B plan — QLoRA 4-bit (chỉ option khả thi với 80GB H100 đơn).

Tính toán VRAM:
- Base bf16 32B: 64GB → KHÔNG fit trên 80GB nếu thêm activations + grad
- Base 4-bit nf4: ~16GB → còn dư ~50GB cho activations + LoRA states
- Full bf16 FT: ~256GB cần — KHÔNG khả thi đơn GPU

Config:
- Base: Qwen/Qwen2.5-Coder-32B (base, không phải Instruct — completion format)
- QLoRA 4-bit nf4, double quant, compute dtype bf16
- FlashAttention 2
- LoRA r=64, alpha=128 (same as Plan B; r=128 cũng OK nếu VRAM còn dư)
- NEFTune α=5
- num_train_epochs=2 (vs 3 với 7B — 32B model lớn, 2 epoch là đủ + tránh memorize)
- learning_rate=5e-5 cosine (hạ vs 1e-4 với 7B — 32B cần update chậm hơn)
- per_device_train_batch_size=1, grad_accum=16 (giảm vs 32 vì step chậm hơn)

Tài nguyên dự kiến: VRAM 30-40GB peak, wall-clock 22-40h trên H100 80GB (1 run).
"""
import os
import time
import torch
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    MODEL_ID = "Qwen/Qwen2.5-Coder-32B"
    OUTPUT_DIR = "./models/qwen32b/plan"
    DATA_PATH = str(Path(__file__).parent / "data" / "java_completion_train.jsonl")
    MAX_LENGTH = 1536
    CACHE_DIR = os.environ.get("HF_HOME", "D:/cache/hugging_face")

    if WANDB_API_KEY:
        import wandb
        wandb.init(project="qwen32b-java", name="qwen32b-qlora-r64")

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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"--- Loading model {MODEL_ID} (QLoRA 4-bit, FA2) — sẽ mất 5-15 phút lần đầu ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

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

    # grad_accum=16 (vs 32 cho 7B) — step 32B chậm hơn, cần feedback nhanh hơn để eval/save sớm
    effective_batch = 1 * 16
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    eval_steps = max(50, int(0.1 * steps_per_epoch))
    print(f"Steps/epoch: {steps_per_epoch}, eval_steps: {eval_steps}")

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=2,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=5e-5,
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
            save_total_limit=2,  # 32B LoRA ckpt ~2GB mỗi cái, giữ 2 cái
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=10,
            dataloader_num_workers=4,
            report_to="wandb" if WANDB_API_KEY else "none",
            run_name="qwen32b-qlora-r64",
        ),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    print("--- Starting training ---")
    checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    print(f"Checkpoint: {checkpoint}")

    start = time.time()
    trainer.train(resume_from_checkpoint=checkpoint)
    print(f"Training: {(time.time() - start) / 3600:.2f}h")

    trainer.save_model(OUTPUT_DIR)
    print(f"Done. Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
