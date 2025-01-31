import os

import numpy as np

from datasets import load_dataset
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm


def save_regions(image, regions, folder, name, idx):
    images = []
    annotations = []

    if "both" not in regions:
        return

    for r_idx, region in enumerate(regions["both"]):
        if "bounding_box" not in region or "symbols" not in region:
            continue

        fromx, tox, fromy, toy = (
            region["bounding_box"]["fromX"],
            region["bounding_box"]["toX"],
            region["bounding_box"]["fromY"],
            region["bounding_box"]["toY"],
        )
        cropped = image.crop((fromx, fromy, tox, toy))
        cropped.save(f"{folder}/{name}/jpg/img_{idx}_{r_idx}.jpg", "JPEG")

        with open(f"{folder}/{name}/gt/img_{idx}_{r_idx}.txt", "w") as f:
            # write symbols one per line
            for symbol in region["symbols"]:
                if "agnostic_symbol_type" not in symbol:
                    continue
                f.write(symbol["agnostic_symbol_type"] + "\n")

        images.append(f"data/{name}/jpg/img_{idx}_{r_idx}.jpg")
        annotations.append(f"data/{name}/gt/img_{idx}_{r_idx}.txt")
    return images, annotations


def prepare_hf_dataset(hf_name, name, folder: str = "data", folds: int = 5):
    images_paths = []
    annotations_paths = []

    # Get dataset from the hub, store it in the folder, create folds and save them
    dataset = load_dataset(hf_name, split="train", num_proc=4)

    os.makedirs(f"{folder}/{name}/jpg", exist_ok=True)
    os.makedirs(f"{folder}/{name}/gt", exist_ok=True)
    os.makedirs(f"{folder}/{name}/splits", exist_ok=True)

    # Wrap the range iterator with tqdm to track progress
    for idx in tqdm(range(len(dataset)), desc="Processing images"):
        image = dataset[idx]["image"]
        regions = dataset[idx]["regions"]
        images, annotations = save_regions(image, regions, folder, name, idx)

        images_paths.extend(images)
        annotations_paths.extend(annotations)

    # create folds from the images and annotations paths, store them in the folder
    create_kfold_splits(
        images_paths,
        annotations_paths,
        n_folds=folds,
        val_pct=0.1,
        name=name,
    )


def create_kfold_splits(
    image_paths: list[str],
    annotation_paths: list[str],
    n_folds: int,
    val_pct: float,
    name: str,
) -> None:
    assert len(image_paths) == len(annotation_paths), (
        "Image and annotation lists must be of the same length."
    )

    # Convert lists to numpy array for indexing
    data_indices = np.arange(len(image_paths))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_val_indices, test_indices) in enumerate(
        kf.split(data_indices)
    ):
        # Get 10% of the training set for validation
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_pct,
            random_state=42,
        )

        # save folds in files in
        # /data/{name}/splits/train_{fold_idx}.dat, val_{fold_idx}.dat, test_{fold_idx}.dat
        with open(f"data/{name}/splits/train_{fold_idx}.dat", "w") as f:
            for idx in train_indices:
                f.write(f"{image_paths[idx]} {annotation_paths[idx]}\n")

        with open(f"data/{name}/splits/val_{fold_idx}.dat", "w") as f:
            for idx in val_indices:
                f.write(f"{image_paths[idx]} {annotation_paths[idx]}\n")

        with open(f"data/{name}/splits/test_{fold_idx}.dat", "w") as f:
            for idx in test_indices:
                f.write(f"{image_paths[idx]} {annotation_paths[idx]}\n")


if __name__ == "__main__":
    prepare_hf_dataset("PRAIG/JAZZMUS", name="jazzmus", folds=5)
