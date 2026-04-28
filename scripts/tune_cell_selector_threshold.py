import argparse
import json

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.evaluation.cell_metrics import average_metric_dicts, compute_cell_metrics
from src.evaluation.selector_inference import (
    score_table_cells,
    select_cells_from_candidates,
)
from src.utils.io import ensure_dir, load_yaml, save_json
from src.utils.modes import add_mode_arg, add_selector_mode_arg, resolve_selector_mode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    add_mode_arg(parser)
    add_selector_mode_arg(parser)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_threshold", type=float, default=0.0)
    parser.add_argument("--max_threshold", type=float, default=1.0)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--thresholds", type=float, nargs="*", default=None)

    return parser.parse_args()


def build_thresholds(args):
    if args.thresholds:
        return sorted({round(value, 6) for value in args.thresholds})

    if args.threshold_step <= 0:
        raise ValueError("threshold_step must be positive.")

    thresholds = []
    value = args.min_threshold
    while value <= args.max_threshold + 1e-12:
        thresholds.append(round(value, 6))
        value += args.threshold_step

    return thresholds


def summarize_threshold(scored_examples, threshold, top_k):
    metric_records = []
    fallback_count = 0
    empty_count = 0
    overflow_count = 0
    selected_count = 0

    for item in scored_examples:
        pred_cells, _, selection_info = select_cells_from_candidates(
            candidates=item["candidates"],
            top_k=top_k,
            threshold=threshold,
        )
        metric_records.append(
            compute_cell_metrics(
                pred_cells=pred_cells,
                gold_cells=item["gold_cells"],
            )
        )

        threshold_selected_count = selection_info["threshold_selected_count"]
        fallback_count += int(selection_info["used_top_k_fallback"])
        empty_count += int(threshold_selected_count == 0)
        overflow_count += int(threshold_selected_count > top_k)
        selected_count += selection_info["selected_count"]

    summary = average_metric_dicts(metric_records)
    total = len(scored_examples)

    summary.update(
        {
            "threshold": threshold,
            "top_k": top_k,
            "fallback_rate": fallback_count / total if total else 0.0,
            "empty_threshold_rate": empty_count / total if total else 0.0,
            "overflow_threshold_rate": overflow_count / total if total else 0.0,
            "avg_selected_count": selected_count / total if total else 0.0,
        }
    )

    return summary


def main():
    args = parse_args()
    config = load_yaml(args.config)
    selector_mode = resolve_selector_mode(args)

    if args.top_k is not None:
        config["cell_selector"]["top_k"] = args.top_k

    thresholds = build_thresholds(args)
    top_k = config["cell_selector"]["top_k"]

    data_dir = f"{config['paths']['processed_dir']}/{args.mode}"
    dataset = load_from_disk(f"{data_dir}/{args.split}")

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

    scored_examples = []
    for example in tqdm(dataset, desc=f"Scoring selector on {args.split}"):
        scored_examples.append(
            {
                "example_id": example["example_id"],
                "totto_id": example["totto_id"],
                "gold_cells": example["highlighted_cells"],
                "candidates": score_table_cells(
                    example=example,
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    device=device,
                ),
            }
        )

    results = [
        summarize_threshold(
            scored_examples=scored_examples,
            threshold=threshold,
            top_k=top_k,
        )
        for threshold in tqdm(thresholds, desc="Sweeping thresholds")
    ]

    best = max(
        results,
        key=lambda item: (
            item["cell_f1"],
            item["cell_precision"],
            item["cell_recall"],
            -item["threshold"],
        ),
    )

    output = {
        "mode": args.mode,
        "split": args.split,
        "selector_mode": selector_mode,
        "top_k": top_k,
        "selection_strategy": "threshold_with_top_k_fallback",
        "best_threshold": best["threshold"],
        "best_metrics": best,
        "results": results,
    }

    metric_dir = f"{config['paths']['metric_dir']}/{args.mode}"
    ensure_dir(metric_dir)
    output_path = (
        f"{metric_dir}/cell_selector_threshold_tuning_"
        f"{args.split}_top{top_k}.json"
    )
    save_json(output, output_path)

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
