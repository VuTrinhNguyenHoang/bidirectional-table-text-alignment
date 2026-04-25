from datasets import DatasetDict

def build_subset(dataset: DatasetDict, mode: str, config: dict):
    seed = config["project"]["seed"]

    n_train = config["data"][mode]["train"]
    n_valid = config["data"][mode]["valid"]

    train_subset = (
        dataset["train"]
        .shuffle(seed=seed)
        .select(range(n_train))
    )

    valid_subset = (
        dataset["validation"]
        .shuffle(seed=seed)
        .select(range(n_valid))
    )

    test_subset = (
        dataset["validation"]
        .shuffle(seed=seed)
        .select(range(n_valid))
    )

    return train_subset, valid_subset, test_subset
