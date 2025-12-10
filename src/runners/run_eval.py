import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import Any, Dict
import yaml
import warnings
import logging

from tqdm import tqdm

from src.data.clean import get_cleaned
from src.models.__init__ import build_model_from_config

warnings.filterwarnings("ignore")

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("vllm.engine").setLevel(logging.ERROR)
logging.getLogger("vllm.core").setLevel(logging.ERROR)

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

def run_eval(config: Dict[str, Any]) -> None:

    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    prompt_cfg = config.get("prompt", {})

    dataset_name = dataset_cfg["name"]
    model_name   = model_cfg["name"]

    output_dir = Path(f"experiments/{dataset_name}/{model_name}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_copy(config, output_dir)

    preprocessed_path = Path(dataset_cfg["preprocessed"])
    if not preprocessed_path.exists():
        print(f"[INFO] Preprocessed path not found, running get_cleaned()")
        get_cleaned(config)
    
    chexpert_df = pd.read_json(preprocessed_path, lines=True)
    chexpert_df = chexpert_df[chexpert_df['split'] == 'test'].reset_index(drop=True)

    model = build_model_from_config(model_cfg)

    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists():
        print(f"[INFO] Skipping prediction generation — file already exists: {predictions_path}")
        return
    fout = predictions_path.open("w")

    dataset_dir = Path(dataset_cfg["base_dir"])
    batch_size = config.get("batch_size", 512)
    num_rows = len(chexpert_df)
    buffer = []

    for idx, row in tqdm(
        chexpert_df.iterrows(),
        total=num_rows,
        desc="Running VQA",
        ncols=80,
    ):
        buffer.append((idx, row))

        if len(buffer) == batch_size:
            # Process and flush
            idxs, rows = zip(*buffer)
            image_paths = [str(dataset_dir / r["image"]) for r in rows]
            prompts = [str(r["prompts"]) for r in rows]

            try:
                predictions = model.answer_vqa_batch(
                    image_paths=image_paths,
                    user_prompts=prompts,
                )
            except Exception as e:
                print(f"ERROR for batch starting at idx {idxs[0]}: {e}")
                predictions = [""] * len(rows)

            for i, (row_idx, row) in enumerate(buffer):
                record = {
                    "row_id": int(row_idx),
                    "image_path": str(dataset_dir / row["image"]),
                    "prediction": predictions[i],
                }
                fout.write(json.dumps(record) + "\n")

            buffer = []  # clear

    # Handle leftover rows in buffer
    if buffer:
        idxs, rows = zip(*buffer)
        image_paths = [str(dataset_dir / r["image"]) for r in rows]
        prompts = [prompt_cfg.get("user", "") for _ in rows]

        try:
            predictions = model.answer_vqa_batch(
                image_paths=image_paths,
                user_prompts=prompts,
            )
        except Exception as e:
            print(f"ERROR for final batch starting at idx {idxs[0]}: {e}")
            predictions = [""] * len(rows)

        for i, (row_idx, row) in enumerate(buffer):
            record = {
                "row_id": int(row_idx),
                "image_path": str(dataset_dir / row["image"]),
                "prediction": predictions[i],
            }
            fout.write(json.dumps(record) + "\n")

    fout.close()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_eval(cfg)


if __name__ == "__main__":
    main()