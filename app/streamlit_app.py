from __future__ import annotations

import json
import html
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.e2e_verifier import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_GENERATOR_DIR,
    DEFAULT_SELECTOR_DIR,
    build_example,
    load_config,
    load_models,
    run_e2e_demo,
)


SAMPLE_PATH = Path(__file__).with_name("sample_claim.json")
DEFAULT_TOP_K = 3
DEFAULT_SEMANTIC_THRESHOLD = 0.72
DEFAULT_LEXICAL_THRESHOLD = 0.35


@st.cache_resource(show_spinner=False)
def cached_models(
    generator_dir: str,
    selector_dir: str,
    sentence_model_name: str | None,
    device: str | None,
):
    return load_models(
        generator_dir=generator_dir,
        selector_dir=selector_dir,
        sentence_model_name=sentence_model_name,
        device=device,
    )


def read_sample(altered_number: bool = False) -> str:
    payload = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    if altered_number:
        payload["text_claim"] = "Brazil won the 2002 FIFA World Cup final 3-0 against Germany."
    return json.dumps(payload, ensure_ascii=False, indent=2)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #f8fafc;
          --surface: #ffffff;
          --ink: #172033;
          --muted: #64748b;
          --line: #dbe3ef;
          --green: #16815f;
          --green-bg: #e8f7f1;
          --red: #b42318;
          --red-bg: #fff0ed;
          --amber: #a15c07;
          --amber-bg: #fff7e6;
          --blue: #2458d3;
          --blue-bg: #eef4ff;
          --shadow: 0 18px 45px rgba(15, 23, 42, .08);
        }
        .stApp { background: var(--bg); color: var(--ink); }
        header[data-testid="stHeader"] {
          background: transparent;
          height: 0rem;
        }
        #MainMenu, footer { visibility: hidden; }
        .block-container {
          padding-top: 1.2rem;
          padding-bottom: 2.8rem;
          max-width: 1180px;
        }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
        h1 { font-size: 2rem; line-height: 1.12; margin-bottom: .2rem; }
        h2 { font-size: 1.08rem; margin-top: .2rem; }
        h3 { font-size: .98rem; }
        div[data-testid="stMetric"] {
          background: var(--surface);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: .8rem .9rem;
        }
        .hero-row {
          display: flex;
          align-items: flex-end;
          justify-content: space-between;
          gap: 1rem;
          padding: .2rem 0 1rem 0;
          border-bottom: 1px solid var(--line);
          margin-bottom: 1.1rem;
        }
        .title-note {
          margin: .25rem 0 0 0;
          color: var(--muted);
          font-size: .95rem;
        }
        .panel {
          background: var(--surface);
          border: 1px solid var(--line);
          border-radius: 8px;
          box-shadow: var(--shadow);
          padding: 1rem;
        }
        .panel-tight {
          background: var(--surface);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: .85rem .95rem;
        }
        .section-kicker {
          color: var(--muted);
          font-size: .75rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: .05em;
          margin-bottom: .4rem;
        }
        .status-card {
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 1rem 1.1rem;
          background: var(--surface);
          margin: .3rem 0 1rem 0;
        }
        .status-supported { border-color: #a6dfc8; background: var(--green-bg); }
        .status-refuted { border-color: #ffc1b8; background: var(--red-bg); }
        .status-uncertain { border-color: #f3d38b; background: var(--amber-bg); }
        .status-title {
          margin: 0;
          font-size: 1.15rem;
          font-weight: 750;
        }
        .status-supported .status-title { color: var(--green); }
        .status-refuted .status-title { color: var(--red); }
        .status-uncertain .status-title { color: var(--amber); }
        .status-reason {
          margin: .35rem 0 0 0;
          color: #334155;
          font-size: .94rem;
        }
        .soft-panel {
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: .9rem 1rem;
          background: var(--surface);
        }
        .claim-text {
          font-size: 1rem;
          line-height: 1.5;
          margin: 0;
          color: var(--ink);
        }
        .muted-label {
          color: var(--muted);
          font-size: .78rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: .04em;
          margin-bottom: .25rem;
        }
        div[data-testid="stButton"] button {
          border-radius: 8px;
          border: 1px solid var(--line);
          font-weight: 650;
          background: #ffffff;
          color: var(--ink);
        }
        div[data-testid="stButton"] button[kind="primary"] {
          background: var(--blue);
          border-color: var(--blue);
          color: #ffffff;
        }
        textarea, input, div[data-baseweb="select"] > div {
          border-radius: 8px !important;
        }
        .stTextArea textarea {
          background: #ffffff !important;
          color: #111827 !important;
          border: 1px solid var(--line) !important;
          font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
          font-size: .9rem;
          line-height: 1.55;
        }
        div[data-testid="stAlert"] {
          border-radius: 8px;
        }
        .html-table {
          width: 100%;
          border-collapse: separate;
          border-spacing: 0;
          overflow: hidden;
          border: 1px solid var(--line);
          border-radius: 8px;
          background: #ffffff;
          font-size: .88rem;
        }
        .html-table th {
          background: #f1f5f9;
          color: #334155;
          text-align: left;
          padding: .62rem .72rem;
          border-bottom: 1px solid var(--line);
          font-weight: 750;
        }
        .html-table td {
          color: #111827;
          padding: .62rem .72rem;
          border-bottom: 1px solid #eef2f7;
          border-right: 1px solid #eef2f7;
          vertical-align: top;
        }
        .html-table tr:last-child td { border-bottom: none; }
        .html-table td:last-child, .html-table th:last-child { border-right: none; }
        .html-table .cell-selected {
          background: #e8f7f1;
          color: #07543f;
          font-weight: 750;
        }
        .claim-box {
          background: #eef4ff;
          border: 1px solid #cfe0ff;
          border-radius: 8px;
          padding: .85rem .95rem;
          color: #17366f;
          line-height: 1.48;
          font-size: .95rem;
        }
        .caption-line {
          color: var(--muted);
          font-size: .86rem;
          margin-top: .55rem;
        }
        div[data-testid="stExpander"] {
          border: 1px solid var(--line);
          border-radius: 8px;
          background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_json(text: str) -> Dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        st.error(f"JSON không hợp lệ: {exc}")
        return None

    if not isinstance(payload, dict):
        st.error("JSON gốc phải là object.")
        return None
    return payload


def table_dataframe(table: Sequence[Sequence[Dict[str, Any]]]) -> pd.DataFrame:
    width = max(len(row) for row in table)
    rows = []
    for row in table:
        values = [str(cell.get("value", "")) for cell in row]
        values.extend([""] * (width - len(values)))
        rows.append(values)
    return pd.DataFrame(rows, columns=[f"C{idx}" for idx in range(width)])


def table_html(
    df: pd.DataFrame,
    selected_cells: Iterable[Sequence[int]] = (),
    max_rows: int | None = None,
) -> str:
    selected = {(int(row), int(col)) for row, col in selected_cells}
    visible_df = df if max_rows is None else df.head(max_rows)

    header = "".join(f"<th>{html.escape(str(col))}</th>" for col in visible_df.columns)
    body_rows = []
    for row_idx, row in visible_df.iterrows():
        cells = []
        for col_idx, value in enumerate(row):
            cls = " class=\"cell-selected\"" if (int(row_idx), col_idx) in selected else ""
            cells.append(f"<td{cls}>{html.escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<table class=\"html-table\">"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def style_table(df: pd.DataFrame, selected_cells: Iterable[Sequence[int]] = ()):
    selected = {(int(row), int(col)) for row, col in selected_cells}

    def apply_styles(_: pd.DataFrame):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for row_idx, col_idx in selected:
            if row_idx in styles.index and 0 <= col_idx < len(styles.columns):
                styles.iat[row_idx, col_idx] = (
                    "background-color: #e8f7f1; color: #07543f; font-weight: 700;"
                )
        return styles

    return df.style.apply(apply_styles, axis=None)


def compact_cell_rows(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("row_headers", "col_headers"):
        if col in df.columns:
            df[col] = df[col].apply(lambda value: " | ".join(value) if isinstance(value, list) else value)
    if "score" in df.columns:
        df["score"] = df["score"].map(lambda value: None if value is None else round(float(value), 4))
    return df


def status_class(status: str) -> str:
    if status == "supported":
        return "status-supported"
    if status == "refuted":
        return "status-refuted"
    return "status-uncertain"


def render_status(rule: Dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="status-card {status_class(rule["status"])}">
          <p class="status-title">{html.escape(rule["verdict"])}</p>
          <p class="status-reason">{html.escape(rule["reason"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_claim_pair(claim: str, generated: str) -> None:
    claim = html.escape(claim)
    generated = html.escape(generated)
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
            <div class="soft-panel">
              <div class="muted-label">Claim đưa vào</div>
              <p class="claim-text">{claim}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="soft-panel">
              <div class="muted-label">Câu sinh từ evidence</div>
              <p class="claim-text">{generated}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Table-Text Claim Verifier",
        page_icon="✓",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_theme()

    if "json_input" not in st.session_state:
        st.session_state["json_input"] = read_sample()

    config = load_config(DEFAULT_CONFIG_PATH)
    generator_dir = str(DEFAULT_GENERATOR_DIR)
    selector_dir = str(DEFAULT_SELECTOR_DIR)
    device = None
    sentence_model_name = config.get("consistency", {}).get("sentence_embedding_model")
    selector_threshold = None
    infer_headers = True
    use_heuristic_headers = True
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    num_beams = int(config["generation"]["num_beams"])

    st.markdown(
        """
        <div class="hero-row">
          <div>
            <h1>Table-Text Claim Verifier</h1>
            <p class="title-note">Demo kiểm chứng claim từ bảng: chọn evidence, sinh lại câu, rồi áp rule số liệu.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Rule threshold", expanded=False):
        rule_col_a, rule_col_b, rule_col_c, rule_col_d = st.columns([1, 1, 1, 1.2])
        with rule_col_a:
            top_k = st.number_input(
                "Top-k cells",
                min_value=1,
                max_value=20,
                value=DEFAULT_TOP_K,
                step=1,
            )
        with rule_col_b:
            semantic_threshold = st.slider(
                "Semantic threshold",
                0.0,
                1.0,
                DEFAULT_SEMANTIC_THRESHOLD,
                0.01,
            )
        with rule_col_c:
            lexical_threshold = st.slider(
                "Lexical fallback",
                0.0,
                1.0,
                DEFAULT_LEXICAL_THRESHOLD,
                0.01,
            )
        with rule_col_d:
            require_number = st.checkbox(
                "Yêu cầu claim có số",
                value=True,
                help="Nếu claim không có số, app trả về trạng thái không đủ căn cứ.",
            )

    input_col, preview_col = st.columns([1.02, 0.98], gap="large")

    with input_col:
        st.markdown('<div class="section-kicker">Input</div>', unsafe_allow_html=True)
        button_col_a, button_col_b = st.columns([1, 1])
        with button_col_a:
            if st.button("Sample đúng", use_container_width=True):
                st.session_state["json_input"] = read_sample(False)
                st.rerun()
        with button_col_b:
            if st.button("Sample sai số", use_container_width=True):
                st.session_state["json_input"] = read_sample(True)
                st.rerun()

        json_input = st.text_area(
            "JSON",
            key="json_input",
            height=380,
            label_visibility="collapsed",
        )

    payload = parse_json(json_input)

    with preview_col:
        st.markdown('<div class="section-kicker">Preview</div>', unsafe_allow_html=True)
        if payload:
            try:
                preview_example = build_example(payload, infer_first_row_headers=infer_headers)
                preview_df = table_dataframe(preview_example["table"])
                st.markdown(table_html(preview_df), unsafe_allow_html=True)
                st.markdown("<div style='height:.85rem'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="panel-tight">
                      <div class="muted-label">Claim</div>
                      <div class="claim-box">{html.escape(preview_example["target"])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if preview_example.get("table_page_title") or preview_example.get("table_section_title"):
                    meta = " / ".join(
                            item
                            for item in [
                                preview_example.get("table_page_title"),
                                preview_example.get("table_section_title"),
                            ]
                            if item
                    )
                    st.markdown(
                        f"<div class='caption-line'>{html.escape(meta)}</div>",
                        unsafe_allow_html=True,
                    )
            except Exception as exc:
                st.error(str(exc))
        run_clicked = st.button("Chạy verification", type="primary", use_container_width=True)

    if run_clicked and payload:
        try:
            with st.spinner("Đang load checkpoint và chạy selector → generator → rule..."):
                bundle = cached_models(
                    generator_dir=generator_dir,
                    selector_dir=selector_dir,
                    sentence_model_name=sentence_model_name,
                    device=device,
                )
                result = run_e2e_demo(
                    payload=payload,
                    bundle=bundle,
                    config=config,
                    top_k=int(top_k),
                    selector_threshold=selector_threshold,
                    semantic_threshold=semantic_threshold,
                    lexical_threshold=lexical_threshold,
                    require_number=require_number,
                    infer_first_row_headers=infer_headers,
                    use_heuristic_headers=use_heuristic_headers,
                    max_new_tokens=int(max_new_tokens),
                    num_beams=int(num_beams),
                )
            st.session_state["last_result"] = result
        except Exception as exc:
            st.error("Không chạy được demo với input/model hiện tại.")
            st.exception(exc)

    result = st.session_state.get("last_result")
    if not result:
        return

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Result</div>', unsafe_allow_html=True)

    rule = result["rule"]
    render_status(rule)
    render_claim_pair(result["claim"], result["generated_statement"])

    similarity = rule["similarity"]
    number_check = rule["number_check"]
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Similarity", f"{similarity['score']:.3f}")
    metric_b.metric("Threshold", f"{similarity['threshold']:.2f}")
    metric_c.metric("Claim numbers", len(number_check["claim_numbers"]))
    metric_d.metric("Unsupported", len(number_check["unsupported_numbers"]))

    evidence_col, table_col = st.columns([1, 1], gap="large")
    with evidence_col:
        st.markdown('<div class="section-kicker">Selected evidence</div>', unsafe_allow_html=True)
        st.markdown(
            table_html(compact_cell_rows(result["selected_cells"])),
            unsafe_allow_html=True,
        )

    with table_col:
        st.markdown('<div class="section-kicker">Highlighted table</div>', unsafe_allow_html=True)
        df = table_dataframe(result["example"]["table"])
        st.markdown(table_html(df, result["pred_cells"]), unsafe_allow_html=True)

    with st.expander("Chi tiết kỹ thuật", expanded=False):
        detail_tab_a, detail_tab_b, detail_tab_c = st.tabs(
            ["Ranked cells", "Linearized input", "Rule JSON"]
        )
        with detail_tab_a:
            st.dataframe(
                pd.DataFrame(result["ranked_cells"]),
                use_container_width=True,
                hide_index=True,
            )
            st.json(result["selection_info"])
        with detail_tab_b:
            st.code(result["linearized_input"], language="text")
        with detail_tab_c:
            st.json(rule)


if __name__ == "__main__":
    main()
