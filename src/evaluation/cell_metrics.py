def compute_cell_metrics(pred_cells, gold_cells):
    pred = {(int(row_idx), int(col_idx)) for row_idx, col_idx in pred_cells}
    gold = {(int(row_idx), int(col_idx)) for row_idx, col_idx in gold_cells}

    true_positive = len(pred & gold)

    precision = true_positive / len(pred) if pred else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "cell_precision": precision,
        "cell_recall": recall,
        "cell_f1": f1,
        "cell_exact_match": 1.0 if pred == gold else 0.0,
    }

def average_metric_dicts(metric_dicts):
    if not metric_dicts:
        return {}

    keys = metric_dicts[0].keys()

    return {
        key: sum(item[key] for item in metric_dicts) / len(metric_dicts)
        for key in keys
    }