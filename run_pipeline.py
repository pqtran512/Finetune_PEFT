#!/usr/bin/env python3
"""
run_pipeline.py
Automates the full PEFT fine-tuning flow:
1. Build dataset
2. Train (with smoke-test or epoch override support)
3. Merge LoRA base
4. Inference
5. Evaluate

Usage:
  python run_pipeline.py --dir finetune_lora7b_planB
  python run_pipeline.py --dir finetune_lora7b_planB --smoke-test
  python run_pipeline.py --dir finetune_qwen32b_plan --skip-build --epochs 1.5
"""

import os
import sys
import subprocess
import time
import re
import argparse
from pathlib import Path


def run_command(cmd, cwd, log_file):
    print(f"\n>>> Running: {' '.join(str(x) for x in cmd)}")
    print(f">>> Cwd: {cwd}")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- START COMMAND: {' '.join(str(x) for x in cmd)} ---\n")
        f.write(f"--- CWD: {cwd} ---\n\n")
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
                
        f.write(f"\n--- END COMMAND (Exit Code: {process.returncode}) ---\n")
        
    process.wait()
    return process.returncode


def get_inference_output_file(inference_script_path: Path) -> str:
    """Parses OUTPUT_FILE variable from the inference script."""
    if not inference_script_path.exists():
        return ""
    try:
        content = inference_script_path.read_text(encoding="utf-8")
        match = re.search(r'OUTPUT_FILE\s*=\s*(.*)', content)
        if match:
            line = match.group(1)
            parts = re.findall(r'["\']([^"\']+)["\']', line)
            if parts:
                return "/".join(parts)
    except Exception as e:
        print(f"Warning: Could not parse OUTPUT_FILE from {inference_script_path.name}: {e}")
    return ""


def main():
    parser = argparse.ArgumentParser(description="Run the full fine-tuning and evaluation pipeline.")
    parser.add_argument(
        "-d", "--dir",
        type=str,
        default=".",
        help="Target plan directory (e.g. finetune_lora7b_planB). Default is current directory."
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the dataset building step."
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the training step."
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip the merging step."
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip the inference step."
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip the evaluation step."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short training run (0.01 epochs) to verify everything works."
    )
    parser.add_argument(
        "--epochs",
        type=float,
        help="Override the number of training epochs."
    )
    
    args = parser.parse_args()
    
    plan_dir = Path(args.dir).resolve()
    if not plan_dir.exists() or not plan_dir.is_dir():
        print(f"Error: Directory {plan_dir} does not exist.")
        sys.exit(1)
        
    print(f"=== Starting pipeline execution in {plan_dir} ===")
    
    log_file = plan_dir / "pipeline_run.log"
    # Clear previous logs if running a new pipeline
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception:
            pass
            
    # Detect directory structure:
    # 1. Root structure: scripts in subdirectories (train/main.py, merge/merge_model.py, etc.)
    # 2. Plan structure: scripts directly inside plan_dir (main_qwen.py, etc.)
    is_root_structure = (plan_dir / "train").is_dir() and (plan_dir / "merge").is_dir()
    
    # 1. Dataset Build Script Resolution
    if is_root_structure:
        build_script = plan_dir / "data" / "build_evol_completion_dataset.py"
        if not build_script.exists():
            build_script = plan_dir / "data" / "build_completion_dataset.py"
    else:
        build_script = plan_dir / "build_completion_dataset.py"
        if not build_script.exists():
            # Fallback to root data folder script
            build_script = Path(__file__).parent.resolve() / "data" / "build_completion_dataset.py"
            
    # 2. Train Script Resolution
    if is_root_structure:
        train_script = plan_dir / "train" / "main.py"
        if not train_script.exists():
            train_script = plan_dir / "train" / "main_qwen.py"
    else:
        train_script = plan_dir / "main_qwen.py"
        if not train_script.exists():
            train_script = plan_dir / "main.py"
            
    # 3. Merge Script Resolution
    if is_root_structure:
        merge_script = plan_dir / "merge" / "merge_model.py"
        if not merge_script.exists():
            merge_script = plan_dir / "merge" / "merge_qwen.py"
    else:
        merge_script = plan_dir / "merge_qwen.py"
        if not merge_script.exists():
            merge_script = plan_dir / "merge_model.py"
            
    # 4. Inference Script Resolution
    if is_root_structure:
        infer_script = plan_dir / "inference" / "inference.py"
        if not infer_script.exists():
            infer_script = plan_dir / "inference" / "inference_qwen.py"
    else:
        infer_script = plan_dir / "inference_qwen.py"
        if not infer_script.exists():
            infer_script = plan_dir / "inference.py"
            
    # 5. Evaluate Script Resolution
    if is_root_structure:
        eval_script = plan_dir / "evaluate" / "evaluate.py"
    else:
        eval_script = plan_dir / "evaluate.py"

    steps = []
    
    # Add step 1: Build Dataset
    if not args.skip_build and build_script.exists():
        steps.append({
            "name": "Build Dataset",
            "cmd": [sys.executable, build_script],
            "cwd": plan_dir,
            "skip": False
        })
    else:
        steps.append({
            "name": "Build Dataset",
            "skip": True,
            "reason": "Skipped by user" if args.skip_build else "Script not found"
        })
        
    # Add step 2: Train Model
    if not args.skip_train and train_script.exists():
        epochs_override = None
        if args.smoke_test:
            epochs_override = 0.01
            print("Smoke test enabled: setting num_train_epochs = 0.01")
        elif args.epochs is not None:
            epochs_override = args.epochs
            print(f"Epochs override enabled: setting num_train_epochs = {args.epochs}")
            
        if epochs_override is not None:
            # Create a modified temporary training script in the same directory as the train script
            try:
                content = train_script.read_text(encoding="utf-8")
                # Replace num_train_epochs value
                new_content = re.sub(
                    r'(num_train_epochs\s*=\s*)[\d\.]+',
                    r'\g<1>' + str(epochs_override),
                    content
                )
                
                # Write to temp file inside the same folder as the original training script
                temp_train_script = train_script.parent / f"{train_script.stem}_temp_run.py"
                temp_train_script.write_text(new_content, encoding="utf-8")
                
                steps.append({
                    "name": "Train Model",
                    "cmd": [sys.executable, temp_train_script],
                    "cwd": plan_dir,
                    "skip": False,
                    "temp_file": temp_train_script
                })
            except Exception as e:
                print(f"Error preparing temporary training script: {e}")
                sys.exit(1)
        else:
            steps.append({
                "name": "Train Model",
                "cmd": [sys.executable, train_script],
                "cwd": plan_dir,
                "skip": False
            })
    else:
        steps.append({
            "name": "Train Model",
            "skip": True,
            "reason": "Skipped by user" if args.skip_train else "Script not found"
        })
        
    # Add step 3: Merge Model
    if not args.skip_merge and merge_script.exists():
        steps.append({
            "name": "Merge Model",
            "cmd": [sys.executable, merge_script],
            "cwd": plan_dir,
            "skip": False
        })
    else:
        steps.append({
            "name": "Merge Model",
            "skip": True,
            "reason": "Skipped by user" if args.skip_merge else "Script not found"
        })
        
    # Add step 4: Inference
    inference_output_file = ""
    if infer_script.exists():
        inference_output_file = get_inference_output_file(infer_script)
        
    if not args.skip_inference and infer_script.exists():
        steps.append({
            "name": "Inference",
            "cmd": [sys.executable, infer_script],
            "cwd": plan_dir,
            "skip": False
        })
    else:
        steps.append({
            "name": "Inference",
            "skip": True,
            "reason": "Skipped by user" if args.skip_inference else "Script not found"
        })
        
    # Add step 5: Evaluation
    if not args.skip_evaluate and eval_script.exists():
        eval_cmd = [sys.executable, eval_script]
        if inference_output_file:
            eval_cmd.append(inference_output_file)
            
        steps.append({
            "name": "Evaluation",
            "cmd": eval_cmd,
            "cwd": plan_dir,
            "skip": False
        })
    else:
        steps.append({
            "name": "Evaluation",
            "skip": True,
            "reason": "Skipped by user" if args.skip_evaluate else "Script not found"
        })
        
    # Execute steps
    overall_start = time.time()
    summary = []
    
    for step in steps:
        if step["skip"]:
            summary.append({
                "name": step["name"],
                "status": "SKIPPED",
                "duration": 0.0,
                "reason": step.get("reason", "")
            })
            continue
            
        print(f"\n==================================================")
        print(f" STEP: {step['name']}")
        print(f"==================================================")
        
        step_start = time.time()
        exit_code = 1
        
        try:
            exit_code = run_command(step["cmd"], step["cwd"], log_file)
        finally:
            # Clean up temp file if created
            if "temp_file" in step and step["temp_file"].exists():
                try:
                    step["temp_file"].unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete temporary file {step['temp_file']}: {e}")
                    
        duration = time.time() - step_start
        
        if exit_code == 0:
            summary.append({
                "name": step["name"],
                "status": "SUCCESS",
                "duration": duration
            })
        else:
            summary.append({
                "name": step["name"],
                "status": f"FAILED (Exit: {exit_code})",
                "duration": duration
            })
            print(f"\n[ERROR] Step '{step['name']}' failed. Halting pipeline execution.")
            break
            
    # Display Summary Table
    overall_duration = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"{'PIPELINE RUN SUMMARY':^60}")
    print("=" * 60)
    
    pipeline_failed = False
    for item in summary:
        duration_str = f"{item['duration']:.2f}s" if item['duration'] > 0 else "-"
        status_str = item['status']
        if "FAILED" in status_str:
            pipeline_failed = True
        reason_str = f" ({item['reason']})" if item.get("reason") else ""
        print(f"  - {item['name']:<18}: {status_str:<15} | Duration: {duration_str:>8}{reason_str}")
        
    print("-" * 60)
    total_min = overall_duration / 60
    print(f"  Total Duration: {total_min:.2f} minutes")
    print(f"  Pipeline Status: {'FAILED' if pipeline_failed else 'SUCCESS'}")
    print(f"  Pipeline Log: {log_file}")
    print("=" * 60)
    
    if pipeline_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
