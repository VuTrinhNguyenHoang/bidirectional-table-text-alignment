from src.data.totto_preprocessing import (
    _add_adjusted_col_offsets,
    _get_heuristic_row_headers,
    _get_heuristic_col_headers,
)

def normalize_cell_indices(highlighted_cells):
    return {(int(r), int(c)) for r, c in highlighted_cells}

def build_cell_text(example, row_idx, col_idx, adjusted_table=None):
    table = example["table"]
    cell = table[row_idx][col_idx]

    if adjusted_table is None:
        adjusted_table = _add_adjusted_col_offsets(table)

    row_headers = _get_heuristic_row_headers(adjusted_table, row_idx, col_idx)
    col_headers = _get_heuristic_col_headers(adjusted_table, row_idx, col_idx)

    parts = []

    parts.append("<claim> " + str(example["target"]) + " </claim>")

    if example.get("table_page_title"):
        parts.append("<page_title> " + str(example["table_page_title"]) + " </page_title>")

    if example.get("table_section_title"):
        parts.append("<section_title> " + str(example["table_section_title"]) + " </section_title>")

    item = "<cell> " + str(cell.get("value", "")) + " "

    for h in col_headers:
        item += "<col_header> " + str(h.get("value", "")) + " </col_header> "

    for h in row_headers:
        item += "<row_header> " + str(h.get("value", "")) + " </row_header> "

    item += "</cell>"
    parts.append(item)

    return " ".join(parts)

def iter_cell_examples(example):
    gold = normalize_cell_indices(example["highlighted_cells"])
    table = example["table"]
    adjusted_table = _add_adjusted_col_offsets(table)

    for r_idx, row in enumerate(table):
        for c_idx, _ in enumerate(row):
            yield {
                "example_id": example["example_id"],
                "totto_id": example["totto_id"],
                "row_idx": r_idx,
                "col_idx": c_idx,
                "text": build_cell_text(
                    example,
                    r_idx,
                    c_idx,
                    adjusted_table=adjusted_table,
                ),
                "label": 1 if (r_idx, c_idx) in gold else 0,
            }

def get_cell_values(example, cell_indices):
    values = []

    for row_idx, col_idx in cell_indices:
        if 0 <= row_idx < len(example["table"]):
            row = example["table"][row_idx]

            if 0 <= col_idx < len(row):
                value = str(row[col_idx].get("value", "")).strip()

                if value:
                    values.append(value)

    return values