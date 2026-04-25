import argparse
import json

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.evaluation.generation_metrics import compute_generation_metrics
from src.utils.io import ensure_dir, load_yaml, save_json, save_jsonl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument("--mode", type=str, default="small", choices=["debug", "small", "medium", "full"])
    parser.add_argument(
        "--run_name",
        type=str,
        default="linearized",
        choices=["linearized", "evidence"],
    )
    return parser.parse_args()

def generate_prediction(model, tokenizer, example, input_col, config, device):
    source = "generate: " + str(example[input_col])

    inputs = tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=config["model"]["max_input_length"],
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=config["generation"]["max_new_tokens"],
        num_beams=config["generation"]["num_beams"],
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    args = parse_args()
    config = load_yaml(args.config)

    input_col = "linearized_input" if args.run_name == "linearized" else "evidence_input"

    data_dir = f"{config['paths']['processed_dir']}/{args.mode}"
    raw_test_path = f"{data_dir}/test" if args.run_name == "linearized" else f"{data_dir}/test_evidence"

    test_dataset = load_from_disk(raw_test_path)

    checkpoint = f"{config['paths']['checkpoint_dir']}/{args.mode}/{args.run_name}/final"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    records = []

    for ex in tqdm(test_dataset, desc=f"Generating {args.run_name}"):
        pred = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            example=ex,
            input_col=input_col,
            config=config,
            device=device,
        )

        records.append({
            "example_id": ex["example_id"],
            "totto_id": ex["totto_id"],
            "input": ex[input_col],
            "target": ex["target"],
            "prediction": pred,
            "highlighted_cells": ex["highlighted_cells"],
            "overlap_subset": ex["overlap_subset"],
        })

    predictions = [r["prediction"] for r in records]
    references = [r["target"] for r in records]

    metrics = compute_generation_metrics(predictions, references)
    metrics["mode"] = args.mode
    metrics["run_name"] = args.run_name

    pred_dir = f"{config['paths']['prediction_dir']}/{args.mode}"
    metric_dir = f"{config['paths']['metric_dir']}/{args.mode}"

    ensure_dir(pred_dir)
    ensure_dir(metric_dir)

    save_jsonl(records, f"{pred_dir}/{args.run_name}_predictions.jsonl")
    save_json(metrics, f"{metric_dir}/{args.run_name}_generation_metrics.json")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
