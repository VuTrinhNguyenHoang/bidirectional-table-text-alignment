import argparse

from datasets import load_from_disk
from transformers import AutoTokenizer

from src.utils.io import load_yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument("--mode", type=str, default="small", choices=["debug", "small", "medium"])
    parser.add_argument(
        "--input_col",
        type=str,
        default="linearized_input",
        choices=["linearized_input", "evidence_input"],
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_yaml(args.config)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    data_dir = f"{config['paths']['processed_dir']}/{args.mode}"

    if args.input_col == "linearized_input":
        train_path = f"{data_dir}/train"
        valid_path = f"{data_dir}/valid"
        test_path = f"{data_dir}/test"
        run_name = "linearized"
    else:
        train_path = f"{data_dir}/train_evidence"
        valid_path = f"{data_dir}/valid_evidence"
        test_path = f"{data_dir}/test_evidence"
        run_name = "evidence"

    train_dataset = load_from_disk(train_path)
    valid_dataset = load_from_disk(valid_path)
    test_dataset = load_from_disk(test_path)

    max_input_length = config["model"]["max_input_length"]
    max_target_length = config["model"]["max_target_length"]

    def preprocess(batch):
        sources = [
            "generate: " + str(x)
            for x in batch[args.input_col]
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
        desc=f"Tokenizing train {run_name}"
    )

    valid_tok = valid_dataset.map(
        preprocess,
        batched=True,
        remove_columns=valid_dataset.column_names,
        desc=f"Tokenizing valid {run_name}"
    )

    test_tok = test_dataset.map(
        preprocess,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc=f"Tokenizing test {run_name}"
    )

    out_train = f"{data_dir}/train_{run_name}_tokenized"
    out_valid = f"{data_dir}/valid_{run_name}_tokenized"
    out_test = f"{data_dir}/test_{run_name}_tokenized"

    train_tok.save_to_disk(out_train)
    valid_tok.save_to_disk(out_valid)
    test_tok.save_to_disk(out_test)

    print(train_tok)
    print(valid_tok)
    print(test_tok)
    print(f"Saved tokenized train to: {out_train}")
    print(f"Saved tokenized valid to: {out_valid}")
    print(f"Saved tokenized test to: {out_test}")

if __name__ == "__main__":
    main()
