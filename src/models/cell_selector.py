from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_cell_selector(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    return model, tokenizer
