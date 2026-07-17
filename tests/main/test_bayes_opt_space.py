from pathlib import Path
import json
import sys

import optuna

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main"))

from bayes_opt import suggest_hyperparams, save_best_params  # noqa: E402
from training_common import load_yaml_config  # noqa: E402


def test_suggest_hyperparams_keys():
    cfg = load_yaml_config()
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        hp = suggest_hyperparams(trial, cfg["search_space"])
        assert set(hp) >= {
            "learning_rate",
            "weight_decay",
            "warmup_ratio",
            "lora_dropout",
            "lora_r",
            "gradient_accumulation_steps",
        }
        assert hp["lora_r"] in {8, 16, 32, 64}
        return 0.0

    study.optimize(objective, n_trials=1)


def test_save_best_params_includes_lora_alpha(tmp_path: Path):
    out = tmp_path / "best_params.json"
    save_best_params(
        out,
        {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "lora_dropout": 0.05,
            "lora_r": 32,
            "gradient_accumulation_steps": 16,
        },
        value=1.23,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["lora_alpha"] == 64
    assert data["best_eval_loss"] == 1.23
