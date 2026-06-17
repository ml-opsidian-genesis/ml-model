import json
import sys
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def main():
    log_file = Path('logs/training_runs.jsonl')
    if not log_file.exists():
        print("ERROR: No metrics found. Run src/train.py first.")
        sys.exit(1)
        
    try:
        # Read the latest training run metrics
        with open(log_file, 'r') as f:
            lines = f.read().strip().split('\n')
            latest_run = json.loads(lines[-1])
            
        metrics = latest_run.get('metrics', {})
        r2_score = metrics.get('r2', 0)
        rmse = metrics.get('rmse', 999)
        
        print(f"Latest Model Metrics: R2 = {r2_score:.4f}, RMSE = {rmse:.4f}")
        
        old_version = "1.0.1"
        try:
            with open("Dockerfile", "r") as df:
                for line in df:
                    if line.startswith("ENV MODEL_VERSION="):
                        old_version = line.strip().split('"')[1]
                        break
        except Exception as e:
            print(f"Warning: Could not parse Dockerfile for old version: {e}")

        old_metrics = None
        try:
            print(f"Fetching production metrics for {old_version} from Hugging Face...")
            old_metrics_path = hf_hub_download(
                repo_id="teamfalsepositives/flood-risk-model",
                filename="metrics.json",
                revision=old_version,
                token=os.environ.get("HF_TOKEN")
            )
            with open(old_metrics_path, 'r') as f:
                old_metrics = json.load(f)
        except Exception as e:
            print(f"Warning: Could not download old metrics for {old_version}. Falling back to main branch...")
            try:
                main_metrics_path = hf_hub_download(
                    repo_id="teamfalsepositives/flood-risk-model",
                    filename="metrics.json",
                    revision="main",
                    token=os.environ.get("HF_TOKEN")
                )
                with open(main_metrics_path, 'r') as f:
                    old_metrics = json.load(f)
            except Exception as ex:
                print(f"Warning: Could not download old metrics from main either: {ex}")

        if not old_metrics:
            print(f"ERROR: Could not find production metrics.json for version {old_version} on Hugging Face.")
            print("To pass the gate, you must upload a baseline metrics.json to the current live tag.")
            sys.exit(1)
            
        MIN_R2 = old_metrics.get("r2", 0.0)
        MAX_RMSE = old_metrics.get("rmse", 999.0)
        print(f"Dynamic Gate Set! New model must beat -> R2: {MIN_R2:.4f}, RMSE: {MAX_RMSE:.4f}")
        
        if r2_score < MIN_R2:
            print(f"MODEL DEGRADATION DETECTED: R2 score ({r2_score:.4f}) is worse than production ({MIN_R2:.4f})!")
            sys.exit(1)
            
        if rmse > MAX_RMSE:
            print(f"MODEL DEGRADATION DETECTED: RMSE ({rmse:.4f}) is worse than production ({MAX_RMSE:.4f})!")
            sys.exit(1)
            
        print("QUALITY GATE PASSED: The new model outperforms or matches production!")
        sys.exit(0)
        
    except Exception as e:
        print(f"Gate failed to run: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
