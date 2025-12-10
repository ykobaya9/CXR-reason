import argparse
import pandas as pd
from pathlib import Path
from typing import Any, Dict
import yaml

CHEXPERT_LABELS = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity',
    'Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax',
    'Pleural Effusion','Pleural Other','Fracture','Support Devices']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot VQA evaluation on chest X-rays."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file for this evaluation run.",
    )
    return parser.parse_args()

def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def save_config_copy(config: Dict[str, Any], out_dir: Path) -> None:
    """Save a snapshot of the config used for this run."""
    config_copy_path = out_dir / "config_used.yaml"
    with config_copy_path.open("w") as f:
        yaml.safe_dump(config, f)

def compute_metrics(config: Dict[str, Any]) -> None:

    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})

    dataset_name = dataset_cfg["name"]
    model_name   = model_cfg["name"]

    output_dir = Path(f"experiments/{dataset_name}/{model_name}/")
    output_path = output_dir / "predictions.jsonl"
    if not output_path.exists():
        print(f"[INFO] Predictions not generated")
    
    generated_df = pd.read_json(output_path, lines=True)
    

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    compute_metrics(cfg)


if __name__ == "__main__":
    main()