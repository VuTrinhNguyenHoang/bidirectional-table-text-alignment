import argparse

from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer

from src.models.generator import load_generator
from src.training.train_seq2seq import build_training_args
from src.utils.io import load_yaml
from src.utils.seed import set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument("--mode", type=str, default="small", choices=["debug", "small", "medium", "full"])
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(config["project"]["seed"])

    model, tokenizer = load_generator(config["model"]["name"])

    data_dir = f"{config['paths']['processed_dir']}/{args.mode}"

    train_dataset = load_from_disk(f"{data_dir}/train_tokenized")
    valid_dataset = load_from_disk(f"{data_dir}/valid_tokenized")

    output_dir = f"{config['paths']['checkpoint_dir']}/{args.mode}"

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    training_args = build_training_args(config, output_dir)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"Saved final checkpoint to: {final_dir}")

if __name__ == "__main__":
    main()
