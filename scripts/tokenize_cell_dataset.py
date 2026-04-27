import argparse

from datasets import load_from_disk
from transformers import AutoTokenizer

from src.utils.io import load_yaml
from src.utils.modes import add_mode_arg, add_selector_mode_arg, resolve_selector_mode

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    add_mode_arg(parser)
    add_selector_mode_arg(parser)
    return parser.parse_args()

def tokenize_split(dataset, tokenizer, max_input_length):
    def preprocess(batch):
        model_inputs = tokenizer(
            batch["text"],
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = batch["label"]

        return model_inputs

    return dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing cell selection split",
    )

def main():
    args = parse_args()
    config = load_yaml(args.config)
    selector_mode = resolve_selector_mode(args)

    tokenizer = AutoTokenizer.from_pretrained(config["cell_selector"]["name"])

    data_dir = f"{config['paths']['processed_dir']}/{selector_mode}"

    cell_train = load_from_disk(f"{data_dir}/cell_train")
    cell_valid = load_from_disk(f"{data_dir}/cell_valid")
    cell_test = load_from_disk(f"{data_dir}/cell_test")

    max_input_length = config["cell_selector"]["max_input_length"]

    cell_train_tok = tokenize_split(cell_train, tokenizer, max_input_length)
    cell_valid_tok = tokenize_split(cell_valid, tokenizer, max_input_length)
    cell_test_tok = tokenize_split(cell_test, tokenizer, max_input_length)

    cell_train_tok.save_to_disk(f"{data_dir}/cell_train_tokenized")
    cell_valid_tok.save_to_disk(f"{data_dir}/cell_valid_tokenized")
    cell_test_tok.save_to_disk(f"{data_dir}/cell_test_tokenized")

    print(cell_train_tok)
    print(cell_valid_tok)
    print(cell_test_tok)
    print(f"Selector mode: {selector_mode}")
    print(f"Saved tokenized cell datasets to: {data_dir}")


if __name__ == "__main__":
    main()
