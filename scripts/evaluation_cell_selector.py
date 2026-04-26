import argparse
import json

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.cell_selection import build_cell_text
from src.evaluation.cell_metrics import average_metric_dicts, compute_cell_metrics
from src.utils.io import ensure_dir, load_yaml, save_json, save_jsonl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        default="small",
        choices=["debug", "small", "medium", "full"],
    )
    parser.add_argument("--top_k", type=int, default=None)

    return parser.parse_args()

@torch.no_grad()
def predict_cells(example, model, tokenizer, config, device):
    candidates = []

    for row_idx, row in enumerate(example["table"]):
        for col_idx, _ in enumerate(row):
            text = build_cell_text(example, row_idx, col_idx)

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config["cell_selector"]["max_input_length"],
            ).to(device)

            logits = model(**inputs).logits
            score = torch.softmax(logits, dim=-1)[0, 1].item()

            candidates.append(
                {
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "score": score,
                }
            )

    candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True,
    )

    top_k = config["cell_selector"]["top_k"]

    pred_cells = [
        [item["row_idx"], item["col_idx"]]
        for item in candidates[:top_k]
    ]

    return pred_cells, candidates

def main():
    args = parse_args()
    config = load_yaml(args.config)

    if args.top_k is not None:
        config["cell_selector"]["top_k"] = args.top_k

    data_dir = f"{config['paths']['processed_dir']}/{args.mode}"
    test_dataset = load_from_disk(f"{data_dir}/test")

    checkpoint = (
        f"{config['paths']['checkpoint_dir']}/"
        f"{config['training_cell_selector']['checkpoint_subdir']}/"
        f"{args.mode}/final"
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    records = []
    metric_records = []

    for example in tqdm(test_dataset, desc="Evaluating cell selector"):
        pred_cells, ranked_cells = predict_cells(
            example=example,
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
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
                "ranked_cells": ranked_cells[:20],
            }
        )

    summary = average_metric_dicts(metric_records)
    summary["mode"] = args.mode
    summary["top_k"] = config["cell_selector"]["top_k"]

    pred_dir = f"{config['paths']['prediction_dir']}/{args.mode}"
    metric_dir = f"{config['paths']['metric_dir']}/{args.mode}"

    ensure_dir(pred_dir)
    ensure_dir(metric_dir)

    top_k = config["cell_selector"]["top_k"]
    save_jsonl(records, f"{pred_dir}/cell_selector_predictions_top{top_k}.jsonl")
    save_json(summary, f"{metric_dir}/cell_selector_metrics_top{top_k}.json")

    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()