import os
import json
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
import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
wandb.login(key="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")


def main():
    # 1. Cấu hình
    MODEL_ID = "Qwen/CodeQwen1.5-7B"
    OUTPUT_DIR = "./models/codeqwen/stage_2_v2"
    DATA_PATH = str(Path(__file__).parent / "data" / "java_completion_train.jsonl")
    MAX_LENGTH = 1536
    CACHE_DIR = "D:/cache/hugging_face"

    wandb.init(project="my-hf-project", name="codeqwen-stage3-completion")

    # 2. Load dataset (function-completion: prefix/target)
    print("--- Loading completion dataset ---")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Không thấy {DATA_PATH}. Chạy `python build_completion_dataset.py` trước."
        )
    start = time.time()
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"⏱ Load dataset time: {time.time() - start:.2f}s, n={len(dataset)}")

    # 3. Tokenizer
    print("--- Loading Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. Tokenize với label masking (chỉ tính loss trên target)
    def tokenize_with_mask(ex):
        prefix_ids = tokenizer(ex["prefix"], add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(
            ex["target"] + tokenizer.eos_token, add_special_tokens=False
        )["input_ids"]
        input_ids = (prefix_ids + target_ids)[:MAX_LENGTH]
        labels = ([-100] * len(prefix_ids) + target_ids)[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    # Train/eval split trước khi tokenize để giữ raw fields debug nếu cần
    split = dataset.train_test_split(test_size=0.02, seed=42)
    train_raw = split["train"]
    eval_raw = split["test"]

    print("--- Tokenizing ---")
    train_ds = train_raw.map(tokenize_with_mask, remove_columns=train_raw.column_names, num_proc=4)
    eval_ds = eval_raw.map(tokenize_with_mask, remove_columns=eval_raw.column_names, num_proc=4)

    # Filter mẫu mà target bị truncate hết (labels toàn -100)
    def has_target(ex):
        return any(l != -100 for l in ex["labels"])

    train_ds = train_ds.filter(has_target)
    eval_ds = eval_ds.filter(has_target)
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Sanity check: in 1 mẫu để confirm mask
    sample = train_ds[0]
    n_prefix_masked = sum(1 for l in sample["labels"] if l == -100)
    n_target = sum(1 for l in sample["labels"] if l != -100)
    print(f"Sample 0: total={len(sample['input_ids'])}, prefix_masked={n_prefix_masked}, target_tokens={n_target}")

    # 5. Quantization config (QLoRA 4bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 6. Load model
    print("--- Loading Model ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 7. LoRA — scale = alpha/r = 1.0 (chống catastrophic forgetting)
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 8. Eval steps
    effective_batch = 1 * 32  # per_device_bs * grad_accum
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    eval_steps = max(50, int(0.1 * steps_per_epoch))
    print(f"Steps/epoch: {steps_per_epoch}, eval_steps: {eval_steps}")

    # 9. Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,

            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,

            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},

            learning_rate=1e-4,
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
            report_to="wandb",
            run_name="codeqwen-stage3-completion",
        ),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    print("--- Starting training ---")
    checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoint = get_last_checkpoint(OUTPUT_DIR)
    print(f"Checkpoint: {checkpoint}")

    start = time.time()
    trainer.train(resume_from_checkpoint=checkpoint)
    print(f"⏱ Training time: {(time.time() - start) / 60:.2f} minutes")

    trainer.save_model(OUTPUT_DIR)
    print(f"Done. Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
