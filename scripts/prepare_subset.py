import argparse

from src.data.load_totto import load_totto_from_parquet
from src.data.subset import build_subset
from src.utils.io import ensure_dir, load_yaml
from src.utils.modes import (
    add_generator_mode_arg,
    add_mode_arg,
    add_selector_mode_arg,
    unique_modes,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    add_mode_arg(parser)
    add_generator_mode_arg(parser)
    add_selector_mode_arg(parser)
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_yaml(args.config)

    dataset = load_totto_from_parquet()

    for mode in unique_modes(args.mode, args.generator_mode, args.selector_mode):
        train_subset, valid_subset, test_subset = build_subset(dataset, mode, config)

        out_dir = f"{config['paths']['processed_dir']}/{mode}"
        ensure_dir(out_dir)

        train_subset.save_to_disk(f"{out_dir}/train")
        valid_subset.save_to_disk(f"{out_dir}/valid")
        test_subset.save_to_disk(f"{out_dir}/test")

        print(f"[mode={mode}]")
        print(train_subset)
        print(valid_subset)
        print(test_subset)
        print(f"Saved subsets to: {out_dir}")

if __name__ == "__main__":
    main()
