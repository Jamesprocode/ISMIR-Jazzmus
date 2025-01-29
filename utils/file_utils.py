import os

import yaml


def check_folders():
    if not os.path.exists("debug"):
        os.makedirs("debug")

    if not os.path.exists("logs"):
        os.makedirs("logs")


def load_config(file_path: str) -> dict:
    """Load a yaml configuration file.

    Args:
        file_path (str): Path to the yaml file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(file_path) as file:
        config = yaml.safe_load(file)
    return config


def create_folds(files):
    # given a list of files, create folds
    from sklearn.model_selection import KFold

    train, val, test = [], [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(files):
        train.append(files[train_idx])
        val, test = (
            files[val_idx][: len(val_idx) // 2],
            files[val_idx][len(val_idx) // 2 :],
        )
    return train, val, test


if __name__ == "__main__":
    # Example usage:
    config_dict = load_config("config.yaml")
    print(config_dict)
