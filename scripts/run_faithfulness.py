import argparse
import json

from datasets import load_from_disk

from src.evaluation.faithfulness import (
    get_highlighted_cell_values,
    simple_faithfulness_check,
    summarize_faithfulness,
)
from src.utils.io import load_jsonl, load_yaml, save_json, save_jsonl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument("--mode", type=str, default="small", choices=["debug", "small", "medium", "full"])
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_yaml(args.config)

    data_dir = f"{config['paths']['processed_dir']}/{args.mode}"

    test_path = f"{data_dir}/test"
    test_dataset = load_from_disk(test_path)

    example_by_id = {
        ex["example_id"]: ex
        for ex in test_dataset
    }

    pred_path = f"{config['paths']['prediction_dir']}/{args.mode}/predictions.jsonl"
    records = load_jsonl(pred_path)

    output = []

    for r in records:
        ex = example_by_id[r["example_id"]]
        evidence_values = get_highlighted_cell_values(ex)

        check = simple_faithfulness_check(
            generated_text=r["prediction"],
            evidence_values=evidence_values,
        )

        new_r = dict(r)
        new_r["evidence_values"] = evidence_values
        new_r.update(check)

        output.append(new_r)

    summary = summarize_faithfulness(output)

    out_pred_path = f"{config['paths']['prediction_dir']}/{args.mode}/faithfulness.jsonl"
    out_metric_path = f"{config['paths']['metric_dir']}/{args.mode}/faithfulness_summary.json"

    save_jsonl(output, out_pred_path)
    save_json(summary, out_metric_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
