# Bayesian Optimization — CodeLlama QLoRA (RTX 3060)

## Cài thêm

```powershell
pip install optuna pyyaml
```

## Chạy BO (từ repo root)

```powershell
python main/bayes_opt.py
# smoke nhanh:
python main/bayes_opt.py --n-trials 1
```

Kết quả: `main/studies/best_params.json` + SQLite `main/studies/codellama_bo.db` (resume được).

`best_params.json` được cập nhật sau mỗi trial COMPLETE. Nếu BO bị dừng giữa chừng mà chưa có file:

```powershell
python main/bayes_opt.py --export-best
```

## Full train với best params

```powershell
python main/main.py
# hoặc
python main/main.py --config main/studies/best_params.json
```

Proxy mỗi trial: ~8% data, 1 epoch, minimize `eval_loss`.
