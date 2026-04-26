import argparse

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer

from src.models.cell_selector import load_cell_selector
from src.training.train_seq2cls import build_training_args
from src.utils.io import load_yaml
from src.utils.seed import set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument("--mode", type=str, default="small", choices=["debug", "small", "medium", "full"],)
    return parser.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def main():
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(config["project"]["seed"])

    model, tokenizer = load_cell_selector(config["cell_selector"]["name"])

    data_dir = f"{config['paths']['processed_dir']}/{args.mode}"

    train_dataset = load_from_disk(f"{data_dir}/cell_train_tokenized")
    valid_dataset = load_from_disk(f"{data_dir}/cell_valid_tokenized")

    output_dir = (
        f"{config['paths']['checkpoint_dir']}/"
        f"{config['training_cell_selector']['checkpoint_subdir']}/"
        f"{args.mode}"
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    training_args = build_training_args(config, output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"Saved final cell selector checkpoint to: {final_dir}")

if __name__ == "__main__":
    main()
