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


def list_files_recursively(directory):
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return sorted(files)


if __name__ == "__main__":
    # Example usage:
    config_dict = load_config("config.yaml")
    print(config_dict)
