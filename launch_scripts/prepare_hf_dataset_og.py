import os
import json
import ast
import argparse

import numpy as np

from datasets import load_dataset
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm


def save_regions(image, regions, folder, name, idx):
    images = []
    annotations = []

    # Check for 'systems' instead of 'both'
    if "systems" not in regions:
        print(f"Warning: No 'systems' key found in regions for image {idx}")
        print(f"Available keys: {regions.keys()}")
        return images, annotations

    for r_idx, system in enumerate(regions["systems"]):
        if "bounding_box" not in system:
            print(f"Warning: No 'bounding_box' in system {r_idx} for image {idx}")
            continue

        # Extract bounding box coordinates
        fromx, tox, fromy, toy = (
            system["bounding_box"]["fromX"],
            system["bounding_box"]["toX"],
            system["bounding_box"]["fromY"],
            system["bounding_box"]["toY"],
        )
        
        # Crop and save the image
        cropped = image.crop((fromx, fromy, tox, toy))

        cropped.save(f"{folder}/{name}/jpg/img_{idx}_{r_idx}_syn.png", "PNG")

        # Save the **kern encoding as the annotation
        with open(f"{folder}/{name}/gt/img_{idx}_{r_idx}_syn.txt", "w") as f:
            if "**kern" in system:
                f.write(system["**kern"])
            else:
                print(f"Warning: No '**kern' in system {r_idx} for image {idx}")
                # Write empty file or skip
                f.write("")

        images.append(f"data/{name}/jpg/img_{idx}_{r_idx}_syn.jpg")
        annotations.append(f"data/{name}/gt/img_{idx}_{r_idx}_syn.txt")
        
    return images, annotations


def prepare_hf_dataset(hf_name, name, folder: str = "data", folds: int = 5):
    images_paths = []
    annotations_paths = []

    # Get dataset from the hub
    dataset = load_dataset(hf_name, split="train", num_proc=4, revision="d127980")

    os.makedirs(f"{folder}/{name}/jpg", exist_ok=True)
    os.makedirs(f"{folder}/{name}/gt", exist_ok=True)
    os.makedirs(f"{folder}/{name}/splits", exist_ok=True)

    # Process each image
    for idx in tqdm(range(len(dataset)), desc="Processing images"):
        image = dataset[idx]["image"]
        
        # Get the annotation - it might already be a dict or a string
        annotation_data = dataset[idx]["annotation"]
        
        # Debug: print type and sample of first annotation
        if idx == 0:
            print(f"\nFirst annotation type: {type(annotation_data)}")
            print(f"First annotation sample: {str(annotation_data)[:200]}...")
        
        # Parse the annotation if it's a string
        if isinstance(annotation_data, str):
            try:
                # Try JSON first
                regions = json.loads(annotation_data)
            except json.JSONDecodeError:
                try:
                    # If JSON fails, try Python literal eval (handles single quotes)
                    regions = ast.literal_eval(annotation_data)
                except Exception as e:
                    print(f"Error parsing annotation for image {idx}: {e}")
                    continue
        else:
            # It's already a dictionary
            regions = annotation_data
        
        images, annotations = save_regions(image, regions, folder, name, idx)
        
        images_paths.extend(images)
        annotations_paths.extend(annotations)

    print(f"\nTotal images processed: {len(images_paths)}")
    print(f"Total annotations processed: {len(annotations_paths)}")
    
    if len(images_paths) == 0:
        print("ERROR: No images were processed! Check the warnings above.")
        return

    # create folds from the images and annotations paths
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

        # save folds in files
        with open(f"data/{name}/splits/train_{fold_idx}_syn.txt", "w") as f:
            for idx in train_indices:
                f.write(f"{image_paths[idx]} {annotation_paths[idx]}\n")

        with open(f"data/{name}/splits/val_{fold_idx}_syn.txt", "w") as f:
            for idx in val_indices:
                f.write(f"{image_paths[idx]} {annotation_paths[idx]}\n")

        with open(f"data/{name}/splits/test_{fold_idx}_syn.txt", "w") as f:
            for idx in test_indices:
                f.write(f"{image_paths[idx]} {annotation_paths[idx]}\n")


if __name__ == "__main__":
    prepare_hf_dataset("PRAIG/JAZZMUS_Synthetic", name="jazzmus_dataset_synthetic_regions_og", folds=5)