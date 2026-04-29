import re
import time

import torch

from src.data.cell_selection import build_cell_text
from src.data.totto_preprocessing import (
    _add_adjusted_col_offsets,
    _get_heuristic_col_headers,
    _get_heuristic_row_headers,
)


def sort_cells_by_table_order(cell_indices):
    return sorted(
        [[int(row_idx), int(col_idx)] for row_idx, col_idx in cell_indices],
        key=lambda cell: (cell[0], cell[1]),
    )


def count_table_cells(example):
    return sum(len(row) for row in example["table"])


def normalize_tokens(text):
    return set(re.findall(r"[a-zA-Z0-9]+", str(text).lower()))


def get_cell_value(cell):
    if isinstance(cell, dict):
        return str(cell.get("value", ""))
    return str(cell)


def build_candidate_pruning_text(example, row_idx, col_idx, adjusted_table):
    table = example["table"]
    cell = table[row_idx][col_idx]

    parts = [get_cell_value(cell)]

    for h in _get_heuristic_col_headers(adjusted_table, row_idx, col_idx):
        parts.append(str(h.get("value", "")))

    for h in _get_heuristic_row_headers(adjusted_table, row_idx, col_idx):
        parts.append(str(h.get("value", "")))

    return " ".join(parts)


def select_candidate_indices(example, config, adjusted_table):
    table = example["table"]

    all_indices = [
        (row_idx, col_idx)
        for row_idx, row in enumerate(table)
        for col_idx, _ in enumerate(row)
    ]

    max_full_table_cells = config["cell_selector"].get("max_full_table_cells", 1000)
    max_candidates = config["cell_selector"].get("max_candidates", 512)

    n_table_cells = len(all_indices)

    if n_table_cells <= max_full_table_cells:
        return set(all_indices), {
            "candidate_filter_applied": False,
            "n_table_cells": n_table_cells,
            "n_candidate_cells": n_table_cells,
            "max_full_table_cells": max_full_table_cells,
            "max_candidates": max_candidates,
        }

    claim_tokens = normalize_tokens(example["target"])
    scored = []

    for row_idx, col_idx in all_indices:
        text = build_candidate_pruning_text(
            example=example,
            row_idx=row_idx,
            col_idx=col_idx,
            adjusted_table=adjusted_table,
        )
        tokens = normalize_tokens(text)
        overlap = len(claim_tokens & tokens)

        cell = table[row_idx][col_idx]
        value = get_cell_value(cell)

        scored.append(
            {
                "row_idx": row_idx,
                "col_idx": col_idx,
                "overlap": overlap,
                "is_header": bool(cell.get("is_header", False)),
                "value_len": len(value),
            }
        )

    scored = sorted(
        scored,
        key=lambda x: (x["overlap"], x["is_header"], x["value_len"]),
        reverse=True,
    )

    selected = scored[:max_candidates]

    candidate_indices = {
        (item["row_idx"], item["col_idx"])
        for item in selected
    }

    return candidate_indices, {
        "candidate_filter_applied": True,
        "n_table_cells": n_table_cells,
        "n_candidate_cells": len(candidate_indices),
        "max_full_table_cells": max_full_table_cells,
        "max_candidates": max_candidates,
    }


def select_cells_from_candidates(candidates, top_k, threshold=None):
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    ranked_candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True,
    )

    threshold_selected_count = None
    used_top_k_fallback = False

    if threshold is None:
        selected_candidates = ranked_candidates[:top_k]
        selection_strategy = "top_k"
    else:
        threshold_candidates = [
            item for item in ranked_candidates if item["score"] >= threshold
        ]
        threshold_selected_count = len(threshold_candidates)

        used_top_k_fallback = (
            threshold_selected_count == 0
            or threshold_selected_count > top_k
        )

        selected_candidates = (
            ranked_candidates[:top_k]
            if used_top_k_fallback
            else threshold_candidates
        )
        selection_strategy = "threshold_with_top_k_fallback"

    pred_cells_by_score = [
        [item["row_idx"], item["col_idx"]]
        for item in selected_candidates
    ]

    pred_cells = sort_cells_by_table_order(pred_cells_by_score)

    selection_info = {
        "strategy": selection_strategy,
        "threshold": threshold,
        "top_k": top_k,
        "threshold_selected_count": threshold_selected_count,
        "used_top_k_fallback": used_top_k_fallback,
        "selected_count": len(pred_cells),
        "pred_cells_by_score": pred_cells_by_score,
    }

    return pred_cells, ranked_candidates, selection_info


@torch.no_grad()
def score_texts_batched(texts, model, tokenizer, config, device):
    scores = []
    batch_size = config["cell_selector"].get("inference_batch_size", 64)

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["cell_selector"]["max_input_length"],
        ).to(device)

        logits = model(**inputs).logits
        batch_scores = torch.softmax(logits, dim=-1)[:, 1]
        scores.extend(batch_scores.detach().cpu().tolist())

    return scores


def score_table_cells(example, model, tokenizer, config, device):
    start_time = time.time()

    adjusted_table = _add_adjusted_col_offsets(example["table"])

    candidate_indices, candidate_filter_info = select_candidate_indices(
        example=example,
        config=config,
        adjusted_table=adjusted_table,
    )

    items = []
    texts = []

    for row_idx, row in enumerate(example["table"]):
        for col_idx, _ in enumerate(row):
            if (row_idx, col_idx) not in candidate_indices:
                continue

            text = build_cell_text(
                example=example,
                row_idx=row_idx,
                col_idx=col_idx,
                adjusted_table=adjusted_table,
            )

            items.append(
                {
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                }
            )
            texts.append(text)

    scores = score_texts_batched(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )

    candidates = []

    for item, score in zip(items, scores):
        candidates.append(
            {
                "row_idx": item["row_idx"],
                "col_idx": item["col_idx"],
                "score": score,
            }
        )

    elapsed_sec = time.time() - start_time

    candidate_filter_info["elapsed_sec"] = round(elapsed_sec, 4)

    return candidates, candidate_filter_info


def predict_cells(example, model, tokenizer, config, device, threshold=None):
    candidates, candidate_filter_info = score_table_cells(
        example=example,
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )

    pred_cells, ranked_candidates, selection_info = select_cells_from_candidates(
        candidates=candidates,
        top_k=config["cell_selector"]["top_k"],
        threshold=threshold,
    )

    selection_info.update(candidate_filter_info)

    return pred_cells, ranked_candidates, selection_info