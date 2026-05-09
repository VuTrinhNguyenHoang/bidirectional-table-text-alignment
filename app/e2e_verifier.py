from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.data.totto_preprocessing import (
    _add_adjusted_col_offsets,
    _get_heuristic_col_headers,
    _get_heuristic_row_headers,
    linearize_from_indices,
)
from src.evaluation.selector_inference import predict_cells
from src.utils.io import load_yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "main.yaml"
DEFAULT_GENERATOR_DIR = ROOT_DIR / "outputs" / "checkpoints" / "t5-small"
DEFAULT_SELECTOR_DIR = ROOT_DIR / "outputs" / "checkpoints" / "electra-small-discriminator"


@dataclass
class ModelBundle:
    generator_model: AutoModelForSeq2SeqLM
    generator_tokenizer: AutoTokenizer
    selector_model: AutoModelForSequenceClassification
    selector_tokenizer: AutoTokenizer
    sentence_model: Any
    device: str


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config = load_yaml(str(config_path))

    # selector_inference reads these values from config["cell_selector"], while
    # older configs placed them under config["model"]. Keep the app tolerant.
    for key in ("max_full_table_cells", "max_candidates", "inference_batch_size"):
        if key in config.get("model", {}):
            config["cell_selector"].setdefault(key, config["model"][key])

    return config


def checkpoint_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    final = path / "final"
    return final if final.exists() else path


def load_tokenizer(path: str | Path):
    try:
        return AutoTokenizer.from_pretrained(path)
    except AttributeError as exc:
        message = str(exc)
        if "keys" not in message:
            raise
        return AutoTokenizer.from_pretrained(path, extra_special_tokens={})


def load_models(
    generator_dir: str | Path = DEFAULT_GENERATOR_DIR,
    selector_dir: str | Path = DEFAULT_SELECTOR_DIR,
    sentence_model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> ModelBundle:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator_dir = checkpoint_dir(generator_dir)
    selector_dir = checkpoint_dir(selector_dir)

    generator_tokenizer = load_tokenizer(generator_dir)
    generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_dir)
    generator_model.to(device)
    generator_model.eval()

    selector_tokenizer = load_tokenizer(selector_dir)
    selector_model = AutoModelForSequenceClassification.from_pretrained(selector_dir)
    selector_model.to(device)
    selector_model.eval()

    sentence_model = None
    if sentence_model_name:
        try:
            from sentence_transformers import SentenceTransformer

            sentence_model = SentenceTransformer(sentence_model_name, device=device)
        except Exception:
            sentence_model = None

    return ModelBundle(
        generator_model=generator_model,
        generator_tokenizer=generator_tokenizer,
        selector_model=selector_model,
        selector_tokenizer=selector_tokenizer,
        sentence_model=sentence_model,
        device=device,
    )


def normalize_cell(raw_cell: Any, row_idx: int, infer_first_row_headers: bool) -> Dict[str, Any]:
    if isinstance(raw_cell, dict):
        value = raw_cell.get("value", raw_cell.get("text", raw_cell.get("content", "")))
        is_header = raw_cell.get("is_header", infer_first_row_headers and row_idx == 0)
        column_span = raw_cell.get("column_span", raw_cell.get("colspan", 1))
        row_span = raw_cell.get("row_span", raw_cell.get("rowspan", 1))
        normalized = dict(raw_cell)
    else:
        value = raw_cell
        is_header = infer_first_row_headers and row_idx == 0
        column_span = 1
        row_span = 1
        normalized = {}

    normalized["value"] = "" if value is None else str(value)
    normalized["is_header"] = bool(is_header)
    normalized["column_span"] = max(1, int(column_span or 1))
    normalized["row_span"] = max(1, int(row_span or 1))
    return normalized


def normalize_table(table: Any, infer_first_row_headers: bool = True) -> List[List[Dict[str, Any]]]:
    if not isinstance(table, list) or not table:
        raise ValueError("`table` must be a non-empty list of rows.")

    normalized_table = []
    for row_idx, row in enumerate(table):
        if not isinstance(row, list) or not row:
            raise ValueError(f"Row {row_idx} must be a non-empty list of cells.")
        normalized_table.append(
            [
                normalize_cell(cell, row_idx, infer_first_row_headers)
                for cell in row
            ]
        )

    return normalized_table


def build_example(payload: Dict[str, Any], infer_first_row_headers: bool = True) -> Dict[str, Any]:
    claim = (
        payload.get("text_claim")
        or payload.get("claim")
        or payload.get("target")
        or payload.get("sentence")
    )
    if not claim:
        raise ValueError("JSON must include `text_claim`, `claim`, `target`, or `sentence`.")

    table = normalize_table(payload.get("table"), infer_first_row_headers)

    return {
        "example_id": payload.get("example_id", "demo"),
        "totto_id": payload.get("totto_id", "demo"),
        "table": table,
        "target": str(claim),
        "table_page_title": str(payload.get("table_page_title", "")),
        "table_section_title": str(payload.get("table_section_title", "")),
        "highlighted_cells": payload.get("highlighted_cells", []),
        "overlap_subset": payload.get("overlap_subset", False),
    }


def cell_value(example: Dict[str, Any], row_idx: int, col_idx: int) -> str:
    return str(example["table"][row_idx][col_idx].get("value", "")).strip()


def cell_headers(example: Dict[str, Any], row_idx: int, col_idx: int, adjusted_table=None) -> Dict[str, List[str]]:
    adjusted_table = adjusted_table or _add_adjusted_col_offsets(example["table"])
    row_headers = [
        str(item.get("value", "")).strip()
        for item in _get_heuristic_row_headers(adjusted_table, row_idx, col_idx)
        if str(item.get("value", "")).strip()
    ]
    col_headers = [
        str(item.get("value", "")).strip()
        for item in _get_heuristic_col_headers(adjusted_table, row_idx, col_idx)
        if str(item.get("value", "")).strip()
    ]
    return {"row_headers": row_headers, "col_headers": col_headers}


def summarize_selected_cells(
    example: Dict[str, Any],
    pred_cells: Sequence[Sequence[int]],
    ranked_cells: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    adjusted_table = _add_adjusted_col_offsets(example["table"])
    score_by_cell = {
        (int(item["row_idx"]), int(item["col_idx"])): float(item["score"])
        for item in ranked_cells
    }

    rows = []
    for row_idx, col_idx in pred_cells:
        row_idx = int(row_idx)
        col_idx = int(col_idx)
        headers = cell_headers(example, row_idx, col_idx, adjusted_table)
        rows.append(
            {
                "row_idx": row_idx,
                "col_idx": col_idx,
                "value": cell_value(example, row_idx, col_idx),
                "score": score_by_cell.get((row_idx, col_idx)),
                "row_headers": headers["row_headers"],
                "col_headers": headers["col_headers"],
            }
        )
    return rows


def summarize_ranked_cells(
    example: Dict[str, Any],
    ranked_cells: Sequence[Dict[str, Any]],
    limit: int = 25,
) -> List[Dict[str, Any]]:
    adjusted_table = _add_adjusted_col_offsets(example["table"])
    rows = []
    for item in ranked_cells[:limit]:
        row_idx = int(item["row_idx"])
        col_idx = int(item["col_idx"])
        headers = cell_headers(example, row_idx, col_idx, adjusted_table)
        rows.append(
            {
                "row_idx": row_idx,
                "col_idx": col_idx,
                "value": cell_value(example, row_idx, col_idx),
                "score": float(item["score"]),
                "row_headers": " | ".join(headers["row_headers"]),
                "col_headers": " | ".join(headers["col_headers"]),
            }
        )
    return rows


@torch.no_grad()
def generate_statement(
    linearized_input: str,
    bundle: ModelBundle,
    config: Dict[str, Any],
    max_new_tokens: Optional[int] = None,
    num_beams: Optional[int] = None,
    no_repeat_ngram_size: Optional[int] = None,
) -> str:
    source = "generate: " + str(linearized_input)
    inputs = bundle.generator_tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=config["model"]["max_input_length"],
    ).to(bundle.device)

    outputs = bundle.generator_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens or config["generation"]["max_new_tokens"],
        num_beams=num_beams or config["generation"]["num_beams"],
        no_repeat_ngram_size=(
            no_repeat_ngram_size
            if no_repeat_ngram_size is not None
            else config["generation"].get("no_repeat_ngram_size", 0)
        ),
    )
    return bundle.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)


NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?%?")
TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def normalize_number(text: str) -> Optional[float]:
    cleaned = str(text).strip().replace(",", "")
    is_percent = cleaned.endswith("%")
    cleaned = cleaned[:-1] if is_percent else cleaned
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value / 100.0 if is_percent else value


def extract_numbers(text: str) -> List[Dict[str, Any]]:
    numbers = []
    for match in NUMBER_RE.finditer(str(text)):
        raw = match.group(0)
        value = normalize_number(raw)
        if value is not None and math.isfinite(value):
            numbers.append({"raw": raw, "value": value})
    return numbers


def numbers_match(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return abs(left - right) <= tolerance


def number_support(
    claim: str,
    evidence_texts: Iterable[str],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    claim_numbers = extract_numbers(claim)
    evidence_numbers = []
    for text in evidence_texts:
        evidence_numbers.extend(extract_numbers(text))

    unsupported = []
    for claim_number in claim_numbers:
        matched = any(
            numbers_match(claim_number["value"], evidence_number["value"], tolerance)
            for evidence_number in evidence_numbers
        )
        if not matched:
            unsupported.append(claim_number["raw"])

    return {
        "claim_numbers": [item["raw"] for item in claim_numbers],
        "evidence_numbers": [item["raw"] for item in evidence_numbers],
        "unsupported_numbers": unsupported,
        "number_pass": len(unsupported) == 0,
        "has_claim_numbers": len(claim_numbers) > 0,
    }


def row_number_support(
    claim: str,
    table: Sequence[Sequence[Dict[str, Any]]],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    claim_numbers = extract_numbers(claim)
    if not claim_numbers:
        return {
            "row_number_pass": True,
            "matching_rows": [],
            "best_row": None,
            "unsupported_numbers": [],
        }

    row_summaries = []
    for row_idx, row in enumerate(table):
        row_text = " ".join(str(cell.get("value", "")) for cell in row)
        row_numbers = extract_numbers(row_text)
        supported = []
        unsupported = []

        for claim_number in claim_numbers:
            matched = any(
                numbers_match(claim_number["value"], row_number["value"], tolerance)
                for row_number in row_numbers
            )
            if matched:
                supported.append(claim_number["raw"])
            else:
                unsupported.append(claim_number["raw"])

        row_summaries.append(
            {
                "row_idx": row_idx,
                "row_text": row_text,
                "row_numbers": [item["raw"] for item in row_numbers],
                "supported_numbers": supported,
                "unsupported_numbers": unsupported,
                "support_count": len(supported),
            }
        )

    matching_rows = [
        item["row_idx"]
        for item in row_summaries
        if not item["unsupported_numbers"]
    ]
    best_row = max(row_summaries, key=lambda item: item["support_count"])

    return {
        "row_number_pass": len(matching_rows) > 0,
        "matching_rows": matching_rows,
        "best_row": best_row,
        "unsupported_numbers": [] if matching_rows else best_row["unsupported_numbers"],
    }


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(str(text).lower()))


def lexical_jaccard(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def semantic_similarity(bundle: ModelBundle, claim: str, generated_statement: str) -> Optional[float]:
    if bundle.sentence_model is None:
        return None

    embeddings = bundle.sentence_model.encode(
        [claim, generated_statement],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return float((embeddings[0] * embeddings[1]).sum())


def verify_claim_with_rules(
    claim: str,
    generated_statement: str,
    evidence_cells: Sequence[Dict[str, Any]],
    bundle: ModelBundle,
    example: Optional[Dict[str, Any]] = None,
    semantic_threshold: float = 0.72,
    lexical_threshold: float = 0.35,
    require_number: bool = True,
) -> Dict[str, Any]:
    evidence_texts = []
    for cell in evidence_cells:
        evidence_texts.append(str(cell.get("value", "")))
        evidence_texts.extend(cell.get("row_headers", []))
        evidence_texts.extend(cell.get("col_headers", []))

    evidence_number_check = number_support(claim, evidence_texts)
    row_check = (
        row_number_support(claim, example["table"])
        if example is not None
        else {
            "row_number_pass": True,
            "matching_rows": [],
            "best_row": None,
            "unsupported_numbers": [],
        }
    )
    number_pass = (
        evidence_number_check["number_pass"]
        and row_check["row_number_pass"]
    )
    number_check = {
        "claim_numbers": evidence_number_check["claim_numbers"],
        "evidence_numbers": evidence_number_check["evidence_numbers"],
        "unsupported_numbers": (
            evidence_number_check["unsupported_numbers"]
            or row_check["unsupported_numbers"]
        ),
        "number_pass": number_pass,
        "has_claim_numbers": evidence_number_check["has_claim_numbers"],
        "evidence_number_pass": evidence_number_check["number_pass"],
        "row_number_pass": row_check["row_number_pass"],
        "matching_rows": row_check["matching_rows"],
        "best_row": row_check["best_row"],
    }
    cosine = semantic_similarity(bundle, claim, generated_statement)
    lexical = lexical_jaccard(claim, generated_statement)

    if cosine is None:
        similarity_pass = lexical >= lexical_threshold
        similarity_mode = "lexical_jaccard"
        similarity_score = lexical
        similarity_threshold = lexical_threshold
    else:
        similarity_pass = cosine >= semantic_threshold
        similarity_mode = "semantic_cosine"
        similarity_score = cosine
        similarity_threshold = semantic_threshold

    if require_number and not number_check["has_claim_numbers"]:
        status = "uncertain"
        verdict = "Không đủ căn cứ"
        reason = "Claim không có số để kiểm chứng bằng rule số; cần dựa vào semantic/logic mạnh hơn."
    elif not number_pass:
        status = "refuted"
        verdict = "Sai / không được bảng hỗ trợ"
        if not number_check["evidence_number_pass"]:
            reason = "Có số trong claim không xuất hiện trong evidence cells đã chọn."
        else:
            reason = "Các số trong claim không cùng xuất hiện trong một hàng bảng nhất quán."
    elif not similarity_pass:
        status = "uncertain"
        verdict = "Không đủ căn cứ"
        reason = "Số liệu khớp, nhưng claim chưa đủ gần với câu sinh ra từ evidence."
    else:
        status = "supported"
        verdict = "Đúng / được bảng hỗ trợ"
        reason = "Số liệu trong claim được evidence hỗ trợ và claim đủ gần với câu sinh từ bảng."

    return {
        "status": status,
        "verdict": verdict,
        "reason": reason,
        "number_check": number_check,
        "similarity": {
            "mode": similarity_mode,
            "score": similarity_score,
            "semantic_cosine": cosine,
            "lexical_jaccard": lexical,
            "threshold": similarity_threshold,
            "pass": similarity_pass,
        },
    }


def run_e2e_demo(
    payload: Dict[str, Any],
    bundle: ModelBundle,
    config: Dict[str, Any],
    top_k: int = 3,
    selector_threshold: Optional[float] = None,
    semantic_threshold: float = 0.72,
    lexical_threshold: float = 0.35,
    require_number: bool = True,
    infer_first_row_headers: bool = True,
    use_heuristic_headers: bool = True,
    max_new_tokens: Optional[int] = None,
    num_beams: Optional[int] = None,
) -> Dict[str, Any]:
    example = build_example(payload, infer_first_row_headers=infer_first_row_headers)
    config = dict(config)
    config["cell_selector"] = dict(config["cell_selector"])
    config["cell_selector"]["top_k"] = int(top_k)

    pred_cells, ranked_cells, selection_info = predict_cells(
        example=example,
        model=bundle.selector_model,
        tokenizer=bundle.selector_tokenizer,
        config=config,
        device=bundle.device,
        threshold=selector_threshold,
    )

    linearized_input = linearize_from_indices(
        table=example["table"],
        cell_indices=pred_cells,
        table_page_title=example.get("table_page_title", ""),
        table_section_title=example.get("table_section_title", ""),
        with_heuristic_headers=use_heuristic_headers,
    )

    generated_statement = generate_statement(
        linearized_input=linearized_input,
        bundle=bundle,
        config=config,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )

    selected_cells = summarize_selected_cells(example, pred_cells, ranked_cells)
    ranked_cell_rows = summarize_ranked_cells(example, ranked_cells)
    rule = verify_claim_with_rules(
        claim=example["target"],
        generated_statement=generated_statement,
        evidence_cells=selected_cells,
        bundle=bundle,
        example=example,
        semantic_threshold=semantic_threshold,
        lexical_threshold=lexical_threshold,
        require_number=require_number,
    )

    return {
        "claim": example["target"],
        "example": example,
        "pred_cells": pred_cells,
        "selected_cells": selected_cells,
        "ranked_cells": ranked_cell_rows,
        "selection_info": selection_info,
        "linearized_input": linearized_input,
        "generated_statement": generated_statement,
        "rule": rule,
    }
