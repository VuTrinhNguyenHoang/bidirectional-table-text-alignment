def normalize_highlighted_cells(highlighted_cells):
    return [(int(r), int(c)) for r, c in highlighted_cells]


def is_header_cell(cell):
    return bool(cell.get("is_header", False))


def row_has_highlight(r_idx, highlighted_cells):
    return any(r == r_idx for r, _ in highlighted_cells)


def select_evidence_rows(table, highlighted_cells, window=2):
    selected = set()

    for r, c in highlighted_cells:
        if 0 <= r < len(table):
            selected.add(r)

            for nr in range(r - window, r + window + 1):
                if 0 <= nr < len(table):
                    selected.add(nr)

    for r_idx, row in enumerate(table):
        if any(is_header_cell(cell) for cell in row):
            selected.add(r_idx)

    return sorted(selected)


def serialize_cell(cell, r_idx, c_idx, highlighted_set):
    value = str(cell.get("value", "")).strip()
    if not value:
        return None

    marker = "<HIGHLIGHT> " if (r_idx, c_idx) in highlighted_set else ""
    header = " header=true" if cell.get("is_header", False) else ""

    return f"{marker}[r={r_idx}, c={c_idx}{header}] {value}"


def serialize_row(row, r_idx, highlighted_set):
    cells = []

    for c_idx, cell in enumerate(row):
        text = serialize_cell(cell, r_idx, c_idx, highlighted_set)
        if text is not None:
            cells.append(text)

    if not cells:
        return None

    return f"row {r_idx}: " + " ; ".join(cells)


def serialize_highlight_only_row(row, r_idx, highlighted_set):
    cells = []

    for c_idx, cell in enumerate(row):
        if (r_idx, c_idx) in highlighted_set or cell.get("is_header", False):
            text = serialize_cell(cell, r_idx, c_idx, highlighted_set)
            if text is not None:
                cells.append(text)

    if not cells:
        return None

    return f"row {r_idx}: " + " ; ".join(cells)


def count_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=True).input_ids)


def serialize_evidence_focused(example, tokenizer, max_tokens=512, window=2):
    table = example["table"]
    highlighted_cells = normalize_highlighted_cells(example["highlighted_cells"])
    highlighted_set = set(highlighted_cells)

    selected_rows = select_evidence_rows(
        table=table,
        highlighted_cells=highlighted_cells,
        window=window,
    )

    parts = [
        f"page_title: {example.get('table_page_title', '')}",
        f"section_title: {example.get('table_section_title', '')}",
    ]

    if example.get("table_section_text"):
        parts.append(f"section_text: {example.get('table_section_text', '')}")

    forced_parts = []
    optional_parts = []

    for r_idx in selected_rows:
        row_text = serialize_row(table[r_idx], r_idx, highlighted_set)
        if row_text is None:
            continue

        if row_has_highlight(r_idx, highlighted_cells):
            forced_parts.append((r_idx, row_text))
        else:
            optional_parts.append((r_idx, row_text))

    for r_idx, row_text in forced_parts:
        candidate = " ".join(parts + [row_text])

        if count_tokens(tokenizer, candidate) <= max_tokens:
            parts.append(row_text)
        else:
            compact = serialize_highlight_only_row(
                table[r_idx],
                r_idx,
                highlighted_set,
            )
            if compact is not None:
                parts.append(compact)

    for _, row_text in optional_parts:
        candidate = " ".join(parts + [row_text])
        if count_tokens(tokenizer, candidate) <= max_tokens:
            parts.append(row_text)

    return " ".join(parts)
