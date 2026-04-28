from src.data.cell_selection import build_cell_text


def sort_cells_by_table_order(cell_indices):
    return sorted(
        [[int(row_idx), int(col_idx)] for row_idx, col_idx in cell_indices],
        key=lambda cell: (cell[0], cell[1]),
    )


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


def score_table_cells(example, model, tokenizer, config, device):
    import torch

    candidates = []

    with torch.no_grad():
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

    return candidates


def predict_cells(example, model, tokenizer, config, device, threshold=None):
    candidates = score_table_cells(
        example=example,
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )

    return select_cells_from_candidates(
        candidates=candidates,
        top_k=config["cell_selector"]["top_k"],
        threshold=threshold,
    )
