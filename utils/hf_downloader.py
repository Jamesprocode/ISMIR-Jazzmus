from datasets import load_dataset

from utils.file_utils import create_folds


def hf_downloader(hf_name, folder: str = "data", name: str = ""):
    # Get dataset from the hub, store it in the folder, create folds and save them
    dataset = load_dataset(hf_name, split="train").save_to_disk(f"{folder}/{name}")

    # Create folds
    train, test, val = create_folds(dataset["train"]["file"])
