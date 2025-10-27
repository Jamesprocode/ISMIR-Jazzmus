"""
Prepare HuggingFace dataset with PIECE-LEVEL splits (70/10/20) to match the paper.

This script:
1. Downloads data from HuggingFace
2. Groups regions by piece ID (from image index in HF dataset)
3. Splits pieces (not regions) into 70% train / 10% val / 20% test
4. Ensures all regions from the same piece go to the same split

This replicates the paper's methodology where pieces are split, not individual regions.
"""

import os
import json
import ast
import argparse
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def save_regions(image, regions, folder, name, idx):
    """Save cropped regions and annotations."""
    images = []
    annotations = []

    if "systems" not in regions:
        print(f"Warning: No 'systems' key found in regions for image {idx}")
        return images, annotations

    for r_idx, system in enumerate(regions["systems"]):
        if "bounding_box" not in system:
            print(f"Warning: No 'bounding_box' in system {r_idx} for image {idx}")
            continue

        fromx, tox, fromy, toy = (
            system["bounding_box"]["fromX"],
            system["bounding_box"]["toX"],
            system["bounding_box"]["fromY"],
            system["bounding_box"]["toY"],
        )

        cropped = image.crop((fromx, fromy, tox, toy))
        cropped.convert("RGB").save(f"{folder}/{name}/jpg/img_{idx}_{r_idx}.jpg", "JPEG")

        with open(f"{folder}/{name}/gt/img_{idx}_{r_idx}.txt", "w") as f:
            if "**kern" in system:
                f.write(system["**kern"])
            else:
                print(f"Warning: No '**kern' in system {r_idx} for image {idx}")
                f.write("")

        images.append(f"data/{name}/jpg/img_{idx}_{r_idx}.jpg")
        annotations.append(f"data/{name}/gt/img_{idx}_{r_idx}.txt")

    return images, annotations


def prepare_hf_dataset_piece_split(
    hf_name,
    name,
    folder: str = "data",
    train_pct: float = 0.70,
    val_pct: float = 0.10,
    test_pct: float = 0.20,
    max_images: int = None,
    random_seed: int = 42,
):
    """
    Prepare HuggingFace dataset with PIECE-LEVEL splits.

    Args:
        hf_name: HuggingFace dataset name
        name: output dataset name
        folder: output folder
        train_pct: percentage of pieces for training (default 0.70)
        val_pct: percentage of pieces for validation (default 0.10)
        test_pct: percentage of pieces for testing (default 0.20)
        max_images: limit number of images to process (for testing)
        random_seed: random seed for reproducibility
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 0.01, "Percentages must sum to 1.0"

    # Get dataset from the hub
    print("="*60)
    print("Loading HuggingFace dataset...")
    print("="*60)
    dataset = load_dataset(hf_name, split="train", num_proc=4)
    print(f"✓ Dataset loaded: {len(dataset)} images\n")

    os.makedirs(f"{folder}/{name}/jpg", exist_ok=True)
    os.makedirs(f"{folder}/{name}/gt", exist_ok=True)
    os.makedirs(f"{folder}/{name}/splits", exist_ok=True)

    # Limit number of images if specified
    num_images = min(len(dataset), max_images) if max_images else len(dataset)

    # Group regions by piece ID
    print("="*60)
    print("Processing images and grouping by piece ID...")
    print("="*60)

    piece_data = defaultdict(lambda: {"images": [], "annotations": []})

    for idx in tqdm(range(num_images), desc="Processing images"):
        image = dataset[idx]["image"]
        annotation_data = dataset[idx]["annotation"]

        if idx == 0:
            print(f"\nFirst annotation type: {type(annotation_data)}")

        # Parse annotation if it's a string
        if isinstance(annotation_data, str):
            try:
                regions = json.loads(annotation_data)
            except json.JSONDecodeError:
                try:
                    regions = ast.literal_eval(annotation_data)
                except Exception as e:
                    print(f"Error parsing annotation for image {idx}: {e}")
                    continue
        else:
            regions = annotation_data

        # Save regions for this piece
        images, annotations = save_regions(image, regions, folder, name, idx)

        # Group by piece ID (the image index in HF dataset represents a piece)
        piece_id = idx
        piece_data[piece_id]["images"].extend(images)
        piece_data[piece_id]["annotations"].extend(annotations)

    print(f"\n✓ Total pieces: {len(piece_data)}")
    print(f"✓ Total regions: {sum(len(p['images']) for p in piece_data.values())}")

    # Split pieces into train/val/test
    print("\n" + "="*60)
    print("Creating piece-level splits...")
    print("="*60)

    # NOTE: The HuggingFace dataset doesn't preserve version information
    # (e.g., piece_version_1, piece_version_2) in the metadata.
    # Each image has a unique index, so we treat each HF image as a unique piece.
    #
    # In the original dataset with JSON files, pieces with multiple handwritten
    # versions were identified by filename (e.g., "TakeFive_version_1.json").
    # The paper's approach was:
    # 1. Filter to unique pieces (only version_1 or no version)
    # 2. Split these unique pieces 70/10/20
    # 3. Put ALL versions of training pieces into train
    #
    # Since HF dataset has already lost this version metadata, we do a
    # simpler piece-level split where each HF image = one piece.
    # This is still better than region-level splitting!

    piece_ids = list(piece_data.keys())
    np.random.seed(random_seed)
    np.random.shuffle(piece_ids)

    n_pieces = len(piece_ids)
    n_test = int(n_pieces * test_pct)
    n_val = int(n_pieces * val_pct)
    n_train = n_pieces - n_test - n_val

    test_pieces = piece_ids[:n_test]
    val_pieces = piece_ids[n_test:n_test + n_val]
    train_pieces = piece_ids[n_test + n_val:]

    print(f"\nPiece distribution:")
    print(f"  Train: {n_train} pieces ({train_pct*100:.0f}%)")
    print(f"  Val:   {n_val} pieces ({val_pct*100:.0f}%)")
    print(f"  Test:  {n_test} pieces ({test_pct*100:.0f}%)")

    # Collect all regions for each split
    train_images = []
    train_annotations = []
    val_images = []
    val_annotations = []
    test_images = []
    test_annotations = []

    for piece_id in train_pieces:
        train_images.extend(piece_data[piece_id]["images"])
        train_annotations.extend(piece_data[piece_id]["annotations"])

    for piece_id in val_pieces:
        val_images.extend(piece_data[piece_id]["images"])
        val_annotations.extend(piece_data[piece_id]["annotations"])

    for piece_id in test_pieces:
        test_images.extend(piece_data[piece_id]["images"])
        test_annotations.extend(piece_data[piece_id]["annotations"])

    print(f"\nRegion distribution:")
    print(f"  Train: {len(train_images)} regions")
    print(f"  Val:   {len(val_images)} regions")
    print(f"  Test:  {len(test_images)} regions")
    print(f"  Total: {len(train_images) + len(val_images) + len(test_images)} regions")

    # Save splits to files
    with open(f"{folder}/{name}/splits/train_0.txt", "w") as f:
        for img, ann in zip(train_images, train_annotations):
            f.write(f"{img} {ann}\n")

    with open(f"{folder}/{name}/splits/val_0.txt", "w") as f:
        for img, ann in zip(val_images, val_annotations):
            f.write(f"{img} {ann}\n")

    with open(f"{folder}/{name}/splits/test_0.txt", "w") as f:
        for img, ann in zip(test_images, test_annotations):
            f.write(f"{img} {ann}\n")

    print("\n" + "="*60)
    print("✓ Dataset preparation complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HuggingFace dataset with piece-level splits")
    parser.add_argument("--hf_name", type=str, default="PRAIG/JAZZMUS",
                        help="HuggingFace dataset name")
    parser.add_argument("--name", type=str, default="jazzmus_piece_split_701020",
                        help="Output dataset name")
    parser.add_argument("--folder", type=str, default="data",
                        help="Output folder")
    parser.add_argument("--train_pct", type=float, default=0.70,
                        help="Percentage of pieces for training")
    parser.add_argument("--val_pct", type=float, default=0.10,
                        help="Percentage of pieces for validation")
    parser.add_argument("--test_pct", type=float, default=0.20,
                        help="Percentage of pieces for testing")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit number of images (for testing)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    prepare_hf_dataset_piece_split(
        hf_name=args.hf_name,
        name=args.name,
        folder=args.folder,
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        max_images=args.max_images,
        random_seed=args.random_seed,
    )
