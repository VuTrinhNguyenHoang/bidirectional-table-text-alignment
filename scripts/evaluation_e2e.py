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

from src.data.cell_selection import get_cell_values
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
from src.evaluation.selector_inference import predict_cells
from src.utils.io import ensure_dir, load_yaml, save_json, save_jsonl
from src.utils.modes import (
    add_generator_mode_arg,
    add_mode_arg,
    add_selector_mode_arg,
    resolve_generator_mode,
    resolve_selector_mode,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    add_mode_arg(parser)
    add_generator_mode_arg(parser)
    add_selector_mode_arg(parser)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)

    return parser.parse_args()

def threshold_tag(threshold):
    if threshold is None:
        return None

    return f"threshold{threshold:.4f}".replace(".", "p")


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
    generator_mode = resolve_generator_mode(args)
    selector_mode = resolve_selector_mode(args)

    if args.top_k is not None:
        config["cell_selector"]["top_k"] = args.top_k

    data_dir = f"{config['paths']['processed_dir']}/{args.mode}"
    test_dataset = load_from_disk(f"{data_dir}/test")

    generator_checkpoint = f"{config['paths']['checkpoint_dir']}/{generator_mode}/final"
    selector_checkpoint = (
        f"{config['paths']['checkpoint_dir']}/"
        f"{config['training_cell_selector']['checkpoint_subdir']}/"
        f"{selector_mode}/final"
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
        pred_cells, ranked_cells, selection_info = predict_cells(
            example=example,
            model=selector_model,
            tokenizer=selector_tokenizer,
            config=config,
            device=device,
            threshold=args.threshold,
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
                "selection_info": selection_info,
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
        "generator_mode": generator_mode,
        "selector_mode": selector_mode,
        "top_k": config["cell_selector"]["top_k"],
        "threshold": args.threshold,
        "selection_strategy": (
            "top_k"
            if args.threshold is None
            else "threshold_with_top_k_fallback"
        ),
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
    tag = f"top{top_k}"
    if args.threshold is not None:
        tag = f"{threshold_tag(args.threshold)}_{tag}"

    save_jsonl(records, f"{pred_dir}/e2e_predictions_{tag}.jsonl")
    save_json(metrics, f"{metric_dir}/e2e_metrics_{tag}.json")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
