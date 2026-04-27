import argparse

from datasets import load_from_disk
from transformers import AutoTokenizer

from src.utils.io import load_yaml
from src.utils.modes import add_generator_mode_arg, add_mode_arg, resolve_generator_mode

INPUT_COL = "linearized_input"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    add_mode_arg(parser)
    add_generator_mode_arg(parser)
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_yaml(args.config)
    generator_mode = resolve_generator_mode(args)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    data_dir = f"{config['paths']['processed_dir']}/{generator_mode}"

    train_path = f"{data_dir}/train"
    valid_path = f"{data_dir}/valid"
    test_path = f"{data_dir}/test"

    train_dataset = load_from_disk(train_path)
    valid_dataset = load_from_disk(valid_path)
    test_dataset = load_from_disk(test_path)

    max_input_length = config["model"]["max_input_length"]
    max_target_length = config["model"]["max_target_length"]

    def preprocess(batch):
        sources = [
            "generate: " + str(x)
            for x in batch[INPUT_COL]
        ]

        targets = [
            str(x)
            for x in batch["target"]
        ]

        model_inputs = tokenizer(
            sources,
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )

    valid_tok = valid_dataset.map(
        preprocess,
        batched=True,
        remove_columns=valid_dataset.column_names,
        desc="Tokenizing valid"
    )

    test_tok = test_dataset.map(
        preprocess,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test"
    )

    out_train = f"{data_dir}/train_tokenized"
    out_valid = f"{data_dir}/valid_tokenized"
    out_test = f"{data_dir}/test_tokenized"

    train_tok.save_to_disk(out_train)
    valid_tok.save_to_disk(out_valid)
    test_tok.save_to_disk(out_test)

    print(train_tok)
    print(valid_tok)
    print(test_tok)
    print(f"Generator mode: {generator_mode}")
    print(f"Saved tokenized train to: {out_train}")
    print(f"Saved tokenized valid to: {out_valid}")
    print(f"Saved tokenized test to: {out_test}")

if __name__ == "__main__":
    main()
