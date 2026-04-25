from datasets import DatasetDict

def _select_split(dataset, size, seed: int):
    shuffled = dataset.shuffle(seed=seed)
    if size is None:
        return shuffled
    return shuffled.select(range(size))

def build_subset(dataset: DatasetDict, mode: str, config: dict):
    seed = config["project"]["seed"]

    n_train = config["data"][mode]["train"]
    n_valid = config["data"][mode]["valid"]

    train_subset = _select_split(dataset["train"], n_train, seed)
    valid_subset = _select_split(dataset["validation"], n_valid, seed)
    test_subset = valid_subset

    return train_subset, valid_subset, test_subset
