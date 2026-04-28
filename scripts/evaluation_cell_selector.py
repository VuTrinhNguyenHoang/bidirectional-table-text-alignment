import argparse
import json

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.evaluation.cell_metrics import average_metric_dicts, compute_cell_metrics
from src.evaluation.selector_inference import predict_cells
from src.utils.io import ensure_dir, load_yaml, save_json, save_jsonl
from src.utils.modes import add_mode_arg, add_selector_mode_arg, resolve_selector_mode

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    add_mode_arg(parser)
    add_selector_mode_arg(parser)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)

    return parser.parse_args()

def threshold_tag(threshold):
    if threshold is None:
        return None

    return f"threshold{threshold:.4f}".replace(".", "p")


def main():
    args = parse_args()
    config = load_yaml(args.config)
    selector_mode = resolve_selector_mode(args)

    if args.top_k is not None:
        config["cell_selector"]["top_k"] = args.top_k

    data_dir = f"{config['paths']['processed_dir']}/{args.mode}"
    test_dataset = load_from_disk(f"{data_dir}/test")

    checkpoint = (
        f"{config['paths']['checkpoint_dir']}/"
        f"{config['training_cell_selector']['checkpoint_subdir']}/"
        f"{selector_mode}/final"
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    records = []
    metric_records = []

    for example in tqdm(test_dataset, desc="Evaluating cell selector"):
        pred_cells, ranked_cells, selection_info = predict_cells(
            example=example,
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            threshold=args.threshold,
        )

        metrics = compute_cell_metrics(
            pred_cells=pred_cells,
            gold_cells=example["highlighted_cells"],
        )
        metric_records.append(metrics)

        records.append(
            {
                "example_id": example["example_id"],
                "totto_id": example["totto_id"],
                "target": example["target"],
                "gold_highlighted_cells": example["highlighted_cells"],
                "pred_highlighted_cells": pred_cells,
                "cell_metrics": metrics,
                "selection_info": selection_info,
                "ranked_cells": ranked_cells[:20],
            }
        )

    summary = average_metric_dicts(metric_records)
    summary["mode"] = args.mode
    summary["selector_mode"] = selector_mode
    summary["top_k"] = config["cell_selector"]["top_k"]
    summary["threshold"] = args.threshold
    summary["selection_strategy"] = (
        "top_k"
        if args.threshold is None
        else "threshold_with_top_k_fallback"
    )

    pred_dir = f"{config['paths']['prediction_dir']}/{args.mode}"
    metric_dir = f"{config['paths']['metric_dir']}/{args.mode}"

    ensure_dir(pred_dir)
    ensure_dir(metric_dir)

    top_k = config["cell_selector"]["top_k"]
    tag = f"top{top_k}"
    if args.threshold is not None:
        tag = f"{threshold_tag(args.threshold)}_{tag}"

    save_jsonl(records, f"{pred_dir}/cell_selector_predictions_{tag}.jsonl")
    save_json(summary, f"{metric_dir}/cell_selector_metrics_{tag}.json")

    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
