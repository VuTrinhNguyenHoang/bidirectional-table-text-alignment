import argparse
import json

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.data.cell_selection import build_cell_text, get_cell_values
from src.data.totto_preprocessing import linearize_from_indices
from src.evaluation.cell_metrics import average_metric_dicts, compute_cell_metrics
from src.evaluation.faithfulness import (
    simple_faithfulness_check,
    summarize_faithfulness,
)
from src.evaluation.consistency import (
    compute_pairwise_cosine_scores,
    load_sentence_embedding_model,
    summarize_cosine_scores,
)
from src.evaluation.generation_metrics import compute_generation_metrics
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

@torch.no_grad()
def generate_prediction(model, tokenizer, linearized_input, config, device):
    source = "generate: " + str(linearized_input)

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
        no_repeat_ngram_size=config["generation"]["no_repeat_ngram_size"],
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    args = parse_args()
    config = load_yaml(args.config)

    if args.top_k is not None:
        config["cell_selector"]["top_k"] = args.top_k

    data_dir = f"{config['paths']['processed_dir']}/{args.mode}"
    test_dataset = load_from_disk(f"{data_dir}/test")

    generator_checkpoint = f"{config['paths']['checkpoint_dir']}/{args.mode}/final"
    selector_checkpoint = (
        f"{config['paths']['checkpoint_dir']}/"
        f"{config['training_cell_selector']['checkpoint_subdir']}/"
        f"{args.mode}/final"
    )

    generator_tokenizer = AutoTokenizer.from_pretrained(generator_checkpoint)
    generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_checkpoint)

    selector_tokenizer = AutoTokenizer.from_pretrained(selector_checkpoint)
    selector_model = AutoModelForSequenceClassification.from_pretrained(
        selector_checkpoint
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator_model.to(device)
    selector_model.to(device)

    generator_model.eval()
    selector_model.eval()

    sentence_model = load_sentence_embedding_model(
        config["consistency"]["sentence_embedding_model"],
        device=device,
    )

    records = []
    cell_metric_records = []
    faithfulness_records = []

    for example in tqdm(test_dataset, desc="Running E2E"):
        pred_cells, ranked_cells = predict_cells(
            example=example,
            model=selector_model,
            tokenizer=selector_tokenizer,
            config=config,
            device=device,
        )

        linearized_input = linearize_from_indices(
            table=example["table"],
            cell_indices=pred_cells,
            table_page_title=example.get("table_page_title", ""),
            table_section_title=example.get("table_section_title", ""),
            with_heuristic_headers=config["e2e"]["use_heuristic_headers"],
        )

        prediction = generate_prediction(
            model=generator_model,
            tokenizer=generator_tokenizer,
            linearized_input=linearized_input,
            config=config,
            device=device,
        )

        cell_metrics = compute_cell_metrics(
            pred_cells=pred_cells,
            gold_cells=example["highlighted_cells"],
        )
        cell_metric_records.append(cell_metrics)

        evidence_values = get_cell_values(example, pred_cells)
        faithfulness = simple_faithfulness_check(
            generated_text=prediction,
            evidence_values=evidence_values,
        )
        faithfulness_records.append(faithfulness)

        records.append(
            {
                "example_id": example["example_id"],
                "totto_id": example["totto_id"],
                "input": linearized_input,
                "target": example["target"],
                "prediction": prediction,
                "gold_highlighted_cells": example["highlighted_cells"],
                "pred_highlighted_cells": pred_cells,
                "cell_metrics": cell_metrics,
                "evidence_values": evidence_values,
                "faithfulness": faithfulness,
                "ranked_cells": ranked_cells[:20],
                "overlap_subset": example["overlap_subset"],
            }
        )

    predictions = [record["prediction"] for record in records]
    references = [record["target"] for record in records]

    cosine_scores = compute_pairwise_cosine_scores(
        model=sentence_model,
        predictions=predictions,
        references=references,
    )

    for record, cosine_score in zip(records, cosine_scores):
        record["semantic_consistency"] = {
            "cosine_similarity": cosine_score
        }

    generation_metrics = compute_generation_metrics(predictions, references)
    faithfulness_metrics = summarize_faithfulness(faithfulness_records)
    semantic_metrics = summarize_cosine_scores(cosine_scores)

    metrics = {
        "mode": args.mode,
        "top_k": config["cell_selector"]["top_k"],
        "cell_selection": average_metric_dicts(cell_metric_records),

        "consistency": {
            "lexical": generation_metrics,
            "semantic": semantic_metrics,
            "number_faithfulness": faithfulness_metrics,
        },
        "generation": generation_metrics,
        "faithfulness": faithfulness_metrics,
    }

    pred_dir = f"{config['paths']['prediction_dir']}/{args.mode}"
    metric_dir = f"{config['paths']['metric_dir']}/{args.mode}"

    ensure_dir(pred_dir)
    ensure_dir(metric_dir)

    top_k = config["cell_selector"]["top_k"]
    save_jsonl(records, f"{pred_dir}/e2e_predictions_top{top_k}.jsonl")
    save_json(metrics, f"{metric_dir}/e2e_metrics_top{top_k}.json")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()