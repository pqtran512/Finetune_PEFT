python -m venv env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
env\Scripts\Activate.ps1
pip install -U -r requirements.txt

pip install torch --index-url https://download.pytorch.org/whl/cu118

python main.py

accelerate launch main.py --model ./final_merged_model --tasks multiple-java --max_length_generation 1024 --temperature 0.2 --precision fp16 --batch_size 1 --generation_only --save_generations --save_generations_path generations_java.json

# Install
git lfs install
git clone <repo>
cd repo
git lfs pull

# 1. Build dataset (~3h CPU, download datasets vào D:/cache/hugging_face)
python build_completion_dataset.py

# 2. Smoke test: edit tạm num_train_epochs=0.01 trong main_qwen.py để chỉ chạy ~50 step,
#    kiểm tra "Sample 0: prefix_masked=... target_tokens=..." in ra hợp lý,
#    nvidia-smi xem VRAM < 11.5GB. Sau đó revert về 1.
python main_qwen.py

# 3. Train chính (qua đêm)
python main_qwen.py

# 4. Merge LoRA + inference + evaluate
python merge_qwen.py
python inference_qwen.py
python evaluate.py


    MODEL_ID = "Qwen/CodeQwen1.5-7B"
    OUTPUT_DIR = "./models/java-qwen/stage_2_v3"
    DATA_PATH = str(Path(__file__).parent / "data" / "java_completion_train.jsonl")
    MAX_LENGTH = 1536
    CACHE_DIR = "D:/cache/hugging_face"
