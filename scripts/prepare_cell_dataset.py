import argparse
import random

from datasets import Dataset, load_from_disk

from src.data.cell_selection import iter_cell_examples
from src.utils.io import load_yaml
from src.utils.modes import add_mode_arg, add_selector_mode_arg, resolve_selector_mode

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    add_mode_arg(parser)
    add_selector_mode_arg(parser)
    return parser.parse_args()

def downsample_negatives(rows, negative_ratio, seed):
    positives = [row for row in rows if row["label"] == 1]
    negatives = [row for row in rows if row["label"] == 0]

    rng = random.Random(seed)
    rng.shuffle(negatives)

    max_negatives = len(positives) * negative_ratio
    sampled_negatives = negatives[:max_negatives]

    output = positives + sampled_negatives
    rng.shuffle(output)

    return output

def build_cell_split(dataset, config):
    rows = []

    for ex in dataset:
        ex_rows = list(iter_cell_examples(ex))
        ex_rows = downsample_negatives(
            ex_rows,
            negative_ratio=config["training_cell_selector"]["negative_ratio"],
            seed=config["project"]["seed"],
        )
        rows.extend(ex_rows)

    return Dataset.from_list(rows)

def main():
    args = parse_args()
    config = load_yaml(args.config)
    selector_mode = resolve_selector_mode(args)

    data_dir = f"{config['paths']['processed_dir']}/{selector_mode}"

    train = load_from_disk(f"{data_dir}/train")
    valid = load_from_disk(f"{data_dir}/valid")
    test = load_from_disk(f"{data_dir}/test")

    cell_train = build_cell_split(train, config)
    cell_valid = build_cell_split(valid, config)
    cell_test = build_cell_split(test, config)

    cell_train.save_to_disk(f"{data_dir}/cell_train")
    cell_valid.save_to_disk(f"{data_dir}/cell_valid")
    cell_test.save_to_disk(f"{data_dir}/cell_test")

    print(cell_train)
    print(cell_valid)
    print(cell_test)
    print(f"Selector mode: {selector_mode}")

if __name__ == "__main__":
    main()
