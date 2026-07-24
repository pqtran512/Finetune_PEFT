pip install optuna pyyaml

$env:HF_HOME="D:\Temp\.cache"
python main/bayes_opt.py --export-best
              # BO → best_params.json

$env:HF_HOME="D:/Temp/.cache/huggingface"
$env:HF_CACHE_DIR="D:/Temp/.cache/huggingface"
python main/main.py
                   # full 3 epochs với best params


$env:HUGGINGFACE_HUB_CACHE="D:\Temp\.cache\huggingface"
python merge/merge_model.py674