from datasets import DatasetDict

def build_subset(dataset: DatasetDict, mode: str, config: dict):
    seed = config["project"]["seed"]

    n_train = config["data"][mode]["train"]
    n_valid = config["data"][mode]["valid"]
    n_test = config["data"][mode]["test"]

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
        dataset["test"]
        .shuffle(seed=seed)
        .select(range(n_test))
    )

    return train_subset, valid_subset, test_subset
