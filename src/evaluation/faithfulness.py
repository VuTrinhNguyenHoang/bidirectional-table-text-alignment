import re

from src.data.serialize import normalize_highlighted_cells


def get_highlighted_cell_values(example):
    table = example["table"]
    highlighted_cells = normalize_highlighted_cells(example["highlighted_cells"])

    values = []

    for r, c in highlighted_cells:
        if 0 <= r < len(table) and 0 <= c < len(table[r]):
            value = str(table[r][c].get("value", "")).strip()
            if value:
                values.append(value)

    return values


def extract_numbers(text):
    return re.findall(r"\d+(?:\.\d+)?", str(text))


def simple_faithfulness_check(generated_text, evidence_values):
    evidence_text = " ".join(evidence_values).lower()
    numbers = extract_numbers(generated_text)

    unsupported_numbers = [
        num for num in numbers
        if num not in evidence_text
    ]

    return {
        "is_faithful_number": len(unsupported_numbers) == 0,
        "unsupported_numbers": unsupported_numbers,
        "n_unsupported_numbers": len(unsupported_numbers),
    }


def summarize_faithfulness(records):
    n = len(records)
    faithful = sum(1 for r in records if r["is_faithful_number"])
    unsupported_total = sum(r["n_unsupported_numbers"] for r in records)

    return {
        "n_examples": n,
        "faithful_number_count": faithful,
        "faithful_number_rate": faithful / n if n else 0.0,
        "unsupported_number_total": unsupported_total,
        "unsupported_number_avg": unsupported_total / n if n else 0.0,
    }
