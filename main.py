import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
import re
import wandb
import time
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
wandb.login(key="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")

def main():
    # 1. Cấu hình
    MODEL_ID = "codellama/CodeLlama-7b-hf" # Thay đổi thành 34b hoặc 70b
    OUTPUT_DIR = "./models/java-codellama-lora/stage_2_v1"
    MAX_LENGTH=1024
    wandb.init(
        project="my-hf-project",
        id="vww6xvhr",     # run_id cũ
        resume="allow"
    )

    # 2. Load dataset
    print("--- Loading Dataset ---")
    dataset_name = "nickrosh/Evol-Instruct-Code-80k-v1"
    
    start = time.time()
    dataset = load_dataset(dataset_name, split="train")
    end = time.time()
    print(f"⏱ Load dataset time: {end-start:.2f}s")
    
    start = time.time()
    dataset = dataset.filter(lambda x: "java" in x["instruction"].lower() or "java" in x["output"].lower())
    end = time.time()
    print(f"⏱ Filter time: {end-start:.2f}s")

    # dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    print(f"Số lượng mẫu Java tìm thấy: {len(dataset)}")
    # print("Dataset sample: ", dataset[0])

    # 3. Load Model & Tokenizer
    print("--- Loading Model & Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ## Tiền xử lý dữ liệu
    print("--- Formatting dataset ---")
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = f"<s>[INST] {instruction} [/INST] {output} </s>"
            texts.append(text)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched=True)

    dataset_split = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # 3. Cấu hình Quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 4. Chuẩn bị model cho training
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 5. Cấu hình LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Target các lớp attention
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    ## Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    # tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

    # 6. Training
    ## Tính toán số bước đánh giá (evaluation steps)
    steps_per_epoch = len(tokenized_train) // (8 * 2)
    eval_steps = max(1, int(0.25 * steps_per_epoch)) # 0.2

    print("Train size:", len(tokenized_train))
    print("Eval steps:", eval_steps)

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        args=TrainingArguments(

            dataloader_num_workers=4,

            fp16=True,                      # Dùng half-precision cho tốc độ

            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},

            eval_strategy="steps",    # Đánh giá theo bước
            eval_steps=eval_steps,

            learning_rate=3e-4,
            load_best_model_at_end=True,    # Chọn checkpoint có loss thấp nhất
            logging_steps=10,

            metric_for_best_model="eval_loss",

            num_train_epochs=2,             # Tinh chỉnh trong 5 epoch

            optim="paged_adamw_8bit",       # QLoRA 8-bit/4-bit optimization
            output_dir=OUTPUT_DIR,

            per_device_train_batch_size=4,  # Tổng Batch size = 4 * 2 = 8

            save_strategy="steps",
            save_steps=eval_steps,

            report_to="wandb",   
            run_name="codellama-stage2",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("--- Starting training ---")
    checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoint = get_last_checkpoint(OUTPUT_DIR)

    start = time.time()
    print("Checkpoint:", checkpoint)
    trainer.train(resume_from_checkpoint=checkpoint)
    end = time.time()
    print(f"⏱ Training time: {(end-start)/60:.2f} minutes")

    # 7. SAVE FINAL MODEL
    trainer.save_model(OUTPUT_DIR)
    print(f"Training hoàn tất! Model lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()