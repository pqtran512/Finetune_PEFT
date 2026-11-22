pip install optuna pyyaml

$env:HF_HOME="D:\Temp\.cache"
python main/bayes_opt.py --export-best
              # BO → best_params.json

$env:HF_HOME="D:/Temp/.cache/huggingface"
$env:HF_CACHE_DIR="D:/Temp/.cache/huggingface"
python main/main.py
                   # full 3 epochs với best params

$env:HF_HOME="D:/Temp/.cache/huggingface"
$env:HUGGINGFACE_HUB_CACHE="D:\Temp\.cache\huggingface"
python merge/merge_model.py


$env:HF_HOME="D:/Temp/.cache/huggingface"
$env:HUGGINGFACE_HUB_CACHE="D:\Temp\.cache\huggingface"
python inference/inference.py `
  --model-path codellama/CodeLlama-7b-hf `
  --output inference/jsonl/inference_codellama_base.jsonl


Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env\Scripts\Activate.ps1
$env:HF_HOME = "D:/cache/huggingface"
python inference/inference.py `
  --model-path "D:/cache/huggingface/models--codellama--CodeLlama-7b-hf/snapshots/6c284d1468fe6c413cf56183e69b194dcfa27fe6" `
  --output inference/jsonl/inference_codellama_base.jsonl