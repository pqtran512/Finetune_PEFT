from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train"))

from training_common import (  # noqa: E402
    load_yaml_config,
    merge_hyperparams,
    load_hyperparams_from_json,
    DEFAULT_HYPERPARAMS,
    is_oom_error,
    is_cuda_error,
    resolve_data_path,
    resolve_train_hyperparams,
    get_hf_cache_dir,
    training_runtime_from_cfg,
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


def test_is_cuda_error_detects_generic_cuda_failures():
    assert is_cuda_error(RuntimeError("CUDA error: unknown error"))
    assert is_cuda_error(RuntimeError("CUDA out of memory. Tried to allocate..."))
    assert not is_cuda_error(RuntimeError("file not found"))


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


def test_runtime_defaults_match_codellama_recipe():
    runtime = training_runtime_from_cfg({})
    assert runtime["per_device_train_batch_size"] == 1
    assert runtime["neftune_noise_alpha"] == 5
    assert runtime["attn_implementation"] is None


def test_holdout_validation_is_disjoint_from_train_pool():
    from datasets import Dataset

    from train_trial import holdout_validation

    raw = Dataset.from_dict({"row_id": list(range(100))})
    pool, val = holdout_validation(raw, val_size=10, seed=42)
    assert len(val) == 10
    assert len(pool) == 90
    assert set(val["row_id"]).isdisjoint(set(pool["row_id"]))
    proxy_n = max(1, int(len(pool) * 0.08))
    proxy = pool.shuffle(seed=42).select(range(proxy_n))
    assert set(proxy["row_id"]).isdisjoint(set(val["row_id"]))


def test_qwen32b_h100_config_keeps_effective_batch_range():
    cfg = load_yaml_config(ROOT / "train" / "configs" / "bo_qwen32b_h100.yaml")
    assert cfg["model_id"] == "Qwen/Qwen2.5-Coder-32B"
    assert cfg["data_path"].endswith("java_completion_train.jsonl")
    assert cfg["val_size"] == 1000
    assert cfg["max_length"] == 2048
    runtime = training_runtime_from_cfg(cfg)
    assert runtime["per_device_train_batch_size"] == 2
    assert runtime["attn_implementation"] == "sdpa"
    micro = runtime["per_device_train_batch_size"]
    effective = {micro * int(c) for c in cfg["search_space"]["gradient_accumulation_steps"]["choices"]}
    assert effective == {8, 16, 32}
    assert cfg["study_db"] != "train/studies/codellama_bo_v2.db"


def test_resolve_train_hyperparams_skips_other_model_best(tmp_path: Path):
    missing = tmp_path / "missing.json"
    hp = resolve_train_hyperparams(None, missing, allow_repo_default=False)
    assert hp["lora_r"] == DEFAULT_HYPERPARAMS["lora_r"]
    assert hp["learning_rate"] == DEFAULT_HYPERPARAMS["learning_rate"]
