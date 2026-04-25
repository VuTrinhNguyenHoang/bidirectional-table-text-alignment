import argparse

from datasets import load_from_disk
from transformers import AutoTokenizer

from src.data.serialize import serialize_evidence_focused
from src.utils.io import load_yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument("--mode", type=str, default="small", choices=["debug", "small", "medium", "full"])
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_yaml(args.config)

    data_dir = f"{config['paths']['processed_dir']}/{args.mode}"

    train_dataset = load_from_disk(f"{data_dir}/train")
    valid_dataset = load_from_disk(f"{data_dir}/valid")
    test_dataset = load_from_disk(f"{data_dir}/test")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    def add_evidence_input(example):
        example["evidence_input"] = serialize_evidence_focused(
            example=example,
            tokenizer=tokenizer,
            max_tokens=config["serialization"]["max_input_tokens"],
            window=config["serialization"]["evidence_window"],
        )
        return example
    
    train_evidence = train_dataset.map(add_evidence_input, desc="Building train evidence input")
    valid_evidence = valid_dataset.map(add_evidence_input, desc="Building valid evidence input")
    test_evidence = test_dataset.map(add_evidence_input, desc="Building test evidence input")

    train_evidence.save_to_disk(f"{data_dir}/train_evidence")
    valid_evidence.save_to_disk(f"{data_dir}/valid_evidence")
    test_evidence.save_to_disk(f"{data_dir}/test_evidence")

    print(train_evidence)
    print(valid_evidence)
    print(test_evidence)
    print(train_evidence[0]["evidence_input"])

if __name__ == "__main__":
    main()
