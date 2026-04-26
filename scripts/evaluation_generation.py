import argparse
import json

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.evaluation.generation_metrics import compute_generation_metrics
from src.utils.io import ensure_dir, load_yaml, save_json, save_jsonl

INPUT_COL = "linearized_input"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument("--mode", type=str, default="small", choices=["debug", "small", "medium", "full"])
    return parser.parse_args()

def generate_prediction(model, tokenizer, example, config, device):
    source = "generate: " + str(example[INPUT_COL])

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

    data_dir = f"{config['paths']['processed_dir']}/{args.mode}"
    raw_test_path = f"{data_dir}/test"

    test_dataset = load_from_disk(raw_test_path)

    checkpoint = f"{config['paths']['checkpoint_dir']}/{args.mode}/final"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    records = []

    for ex in tqdm(test_dataset, desc="Generating"):
        pred = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            example=ex,
            config=config,
            device=device,
        )

        records.append({
            "example_id": ex["example_id"],
            "totto_id": ex["totto_id"],
            "input": ex[INPUT_COL],
            "target": ex["target"],
            "prediction": pred,
            "highlighted_cells": ex["highlighted_cells"],
            "overlap_subset": ex["overlap_subset"],
        })

    predictions = [r["prediction"] for r in records]
    references = [r["target"] for r in records]

    metrics = compute_generation_metrics(predictions, references)
    metrics["mode"] = args.mode

    pred_dir = f"{config['paths']['prediction_dir']}/{args.mode}"
    metric_dir = f"{config['paths']['metric_dir']}/{args.mode}"

    ensure_dir(pred_dir)
    ensure_dir(metric_dir)

    save_jsonl(records, f"{pred_dir}/predictions.jsonl")
    save_json(metrics, f"{metric_dir}/generation_metrics.json")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
