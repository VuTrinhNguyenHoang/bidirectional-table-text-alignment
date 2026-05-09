# Streamlit Demo

Local demo cho pipeline table-text verification:

1. Nhập JSON gồm `table` chưa linearize và `text_claim`.
2. Cell selector chọn evidence cells từ claim.
3. Generator sinh một câu từ evidence đã chọn.
4. Rule kiểm tra claim bằng numeric support và semantic/lexical similarity.

Chạy từ repo root:

```bash
streamlit run app/streamlit_app.py
```

Hai checkpoint mặc định:

- `outputs/checkpoints/t5-small`
- `outputs/checkpoints/electra-small-discriminator`

JSON tối thiểu:

```json
{
  "table_page_title": "FIFA World Cup Finals",
  "table_section_title": "Selected finals",
  "text_claim": "Brazil won the 2002 FIFA World Cup final 2-0 against Germany.",
  "table": [
    ["Year", "Winner", "Score", "Runner-up"],
    ["1998", "France", "3-0", "Brazil"],
    ["2002", "Brazil", "2-0", "Germany"]
  ]
}
```
