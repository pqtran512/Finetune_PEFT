from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main"))

from training_common import (  # noqa: E402
    load_yaml_config,
    merge_hyperparams,
    load_hyperparams_from_json,
    DEFAULT_HYPERPARAMS,
    is_oom_error,
    resolve_data_path,
    resolve_train_hyperparams,
    get_hf_cache_dir,
)


def test_default_hyperparams_baseline():
    assert DEFAULT_HYPERPARAMS["learning_rate"] == 1e-4
    assert DEFAULT_HYPERPARAMS["lora_r"] == 32
    assert DEFAULT_HYPERPARAMS["lora_alpha"] == 64
    assert DEFAULT_HYPERPARAMS["gradient_accumulation_steps"] == 16


def test_merge_sets_lora_alpha_from_r():
    hp = merge_hyperparams({"lora_r": 16, "learning_rate": 2e-5})
    assert hp["lora_r"] == 16
    assert hp["lora_alpha"] == 32
    assert hp["learning_rate"] == 2e-5
    assert hp["weight_decay"] == 0.01


def test_load_yaml_config_has_search_keys():
    cfg = load_yaml_config()
    assert cfg["n_trials"] == 40
    assert cfg["proxy_train_fraction"] == 0.08
    assert "learning_rate" in cfg["search_space"]
    assert cfg["search_space"]["lora_r"]["choices"] == [8, 16, 32, 64]


def test_load_hyperparams_from_json(tmp_path: Path):
    p = tmp_path / "best_params.json"
    p.write_text(
        json.dumps(
            {
                "learning_rate": 5e-5,
                "weight_decay": 0.02,
                "warmup_ratio": 0.05,
                "lora_dropout": 0.1,
                "lora_r": 64,
                "gradient_accumulation_steps": 32,
            }
        ),
        encoding="utf-8",
    )
    hp = load_hyperparams_from_json(p)
    assert hp["lora_alpha"] == 128
    assert hp["lora_r"] == 64


def test_is_oom_error_detects_cuda_oom():
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate..."))
    assert is_oom_error(RuntimeError("cuda OOM"))
    assert not is_oom_error(RuntimeError("something else"))


def test_resolve_data_path_relative_to_repo():
    cfg = load_yaml_config()
    p = resolve_data_path(cfg)
    assert p.is_absolute()
    assert p.as_posix().endswith("java_completion_train.jsonl")


def test_resolve_train_hyperparams_from_explicit(tmp_path: Path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"lora_r": 8, "learning_rate": 2e-5}), encoding="utf-8")
    hp = resolve_train_hyperparams(p)
    assert hp["lora_r"] == 8
    assert hp["lora_alpha"] == 16


def test_get_hf_cache_dir_uses_hf_home(monkeypatch):
    monkeypatch.setenv("HF_HOME", "D:/cache/huggingface")
    assert get_hf_cache_dir() == "D:/cache/huggingface"
