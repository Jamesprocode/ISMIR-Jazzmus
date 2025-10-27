"""
Prepare HuggingFace dataset with piece-level splits to replicate the paper.

Paper's methodology:
- 163 unique pieces total
- Split unique pieces: 115 train / 16 val / 32 test (70/10/20)
- All versions of training pieces go to train
- Results in: 245 scores (train) / 16 scores (val) / 32 scores (test)
- Region counts: 1696 (train) / 102 (val) / 220 (test)

This script:
1. Extracts piece titles from MusicXML (Title - Composer)
2. Groups pieces by title (identifies duplicates/versions)
3. Keeps first occurrence of each unique piece for splitting
4. Splits unique pieces 70/10/20
5. Adds ALL versions of training pieces back to training
"""

import os
import json
import ast
import argparse
import re
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def extract_title_from_musicxml(musicxml_str):
    """Extract title from MusicXML string."""
    match = re.search(r'<movement-title>(.*?)</movement-title>', musicxml_str)
    if match:
        return match.group(1).strip()
    match = re.search(r'<work-title>(.*?)</work-title>', musicxml_str)
    if match:
        return match.group(1).strip()
    return None


def extract_composer_from_musicxml(musicxml_str):
    """Extract composer from MusicXML string."""
    match = re.search(r'<creator type="composer">(.*?)</creator>', musicxml_str)
    if match:
        return match.group(1).strip()
    return None


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


def prepare_hf_dataset_paper_split(
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
    Prepare HuggingFace dataset with piece-level splits like the paper.
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

    num_images = min(len(dataset), max_images) if max_images else len(dataset)

    # First pass: Extract titles and group by piece
    print("="*60)
    print("Pass 1: Extracting piece titles from MusicXML...")
    print("="*60)

    piece_titles = {}  # idx -> "Title - Composer"
    title_to_indices = defaultdict(list)  # "Title - Composer" -> [idx1, idx2, ...]

    for idx in tqdm(range(num_images), desc="Extracting titles"):
        annotation_data = dataset[idx]["annotation"]

        if isinstance(annotation_data, str):
            try:
                annotation = json.loads(annotation_data)
            except json.JSONDecodeError:
                annotation = ast.literal_eval(annotation_data)
        else:
            annotation = annotation_data

        # Extract title from MusicXML in encodings
        if 'encodings' in annotation and 'musicxml' in annotation['encodings']:
            musicxml = annotation['encodings']['musicxml']
            title = extract_title_from_musicxml(musicxml)
            composer = extract_composer_from_musicxml(musicxml)

            if title:
                piece_id = f"{title} - {composer}" if composer else title
                piece_titles[idx] = piece_id
                title_to_indices[piece_id].append(idx)
            else:
                # Fallback to index if no title
                piece_id = f"Unknown_{idx}"
                piece_titles[idx] = piece_id
                title_to_indices[piece_id].append(idx)
        else:
            piece_id = f"Unknown_{idx}"
            piece_titles[idx] = piece_id
            title_to_indices[piece_id].append(idx)

    print(f"\n✓ Total images: {num_images}")
    print(f"✓ Unique pieces: {len(title_to_indices)}")

    duplicates = {title: indices for title, indices in title_to_indices.items() if len(indices) > 1}
    print(f"✓ Pieces with duplicates: {len(duplicates)}")
    print(f"✓ Total duplicate scores: {sum(len(indices) for indices in duplicates.values())}")

    # Create list of unique titles (like filtered_scores in create_splits.py)
    # For each title, we'll use the first occurrence to represent it
    unique_titles = list(title_to_indices.keys())
    print(f"\n✓ Total unique pieces (like paper's 163): {len(unique_titles)}")

    # Split unique TITLES (not indices) - this is the key difference!
    print("\n" + "="*60)
    print("Pass 2: Splitting unique piece titles...")
    print("="*60)

    np.random.seed(random_seed)
    shuffled_titles = np.array(unique_titles)
    np.random.shuffle(shuffled_titles)

    n_unique = len(shuffled_titles)
    n_test = int(n_unique * test_pct)
    n_val = int(n_unique * val_pct)
    n_train = n_unique - n_test - n_val

    # Split by TITLES
    test_titles = shuffled_titles[:n_test].tolist()
    val_titles = shuffled_titles[n_test:n_test + n_val].tolist()
    train_titles = shuffled_titles[n_test + n_val:].tolist()

    # Now convert titles to indices
    # The key logic from create_splits.py line 105:
    # train gets everything NOT in test/val

    # First, collect which indices are assigned to test/val (only first occurrence)
    test_indices = [title_to_indices[title][0] for title in test_titles]
    val_indices = [title_to_indices[title][0] for title in val_titles]

    assigned_to_test_or_val = set(test_indices + val_indices)

    # Train: ALL indices that are NOT in test or val
    # This automatically includes:
    # - ALL versions of training pieces
    # - Additional versions of test/val pieces (since only first was assigned)
    train_all_indices = [idx for idx in range(num_images) if idx not in assigned_to_test_or_val]

    print(f"\nUnique piece distribution:")
    print(f"  Train: {n_train} unique pieces")
    print(f"  Val:   {n_val} unique pieces")
    print(f"  Test:  {n_test} unique pieces")

    print(f"\nScore distribution (with duplicates):")
    print(f"  Train: {len(train_all_indices)} scores (includes all versions)")
    print(f"  Val:   {len(val_indices)} scores")
    print(f"  Test:  {len(test_indices)} scores")

    # Process and save images
    print("\n" + "="*60)
    print("Pass 3: Processing and saving images...")
    print("="*60)

    piece_data = {}
    for idx in tqdm(range(num_images), desc="Processing images"):
        image = dataset[idx]["image"]
        annotation_data = dataset[idx]["annotation"]

        if isinstance(annotation_data, str):
            try:
                annotation = json.loads(annotation_data)
            except json.JSONDecodeError:
                annotation = ast.literal_eval(annotation_data)
        else:
            annotation = annotation_data

        images, annotations = save_regions(image, annotation, folder, name, idx)
        piece_data[idx] = {"images": images, "annotations": annotations}

    # Collect regions for each split
    train_images = []
    train_annotations = []
    val_images = []
    val_annotations = []
    test_images = []
    test_annotations = []

    for idx in train_all_indices:
        train_images.extend(piece_data[idx]["images"])
        train_annotations.extend(piece_data[idx]["annotations"])

    for idx in val_indices:
        val_images.extend(piece_data[idx]["images"])
        val_annotations.extend(piece_data[idx]["annotations"])

    for idx in test_indices:
        test_images.extend(piece_data[idx]["images"])
        test_annotations.extend(piece_data[idx]["annotations"])

    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Region distribution:")
    print(f"  Train: {len(train_images)} regions")
    print(f"  Val:   {len(val_images)} regions")
    print(f"  Test:  {len(test_images)} regions")
    print(f"  Total: {len(train_images) + len(val_images) + len(test_images)} regions")

    print(f"\nPaper target: 1696 (train) / 102 (val) / 220 (test)")

    # Save splits
    with open(f"{folder}/{name}/splits/train_0.txt", "w") as f:
        for img, ann in zip(train_images, train_annotations):
            f.write(f"{img} {ann}\n")

    with open(f"{folder}/{name}/splits/val_0.txt", "w") as f:
        for img, ann in zip(val_images, val_annotations):
            f.write(f"{img} {ann}\n")

    with open(f"{folder}/{name}/splits/test_0.txt", "w") as f:
        for img, ann in zip(test_images, test_annotations):
            f.write(f"{img} {ann}\n")

    print("\n✓ Dataset preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HuggingFace dataset with paper-style piece-level splits")
    parser.add_argument("--hf_name", type=str, default="PRAIG/JAZZMUS",
                        help="HuggingFace dataset name")
    parser.add_argument("--name", type=str, default="jazzmus_paper_split",
                        help="Output dataset name")
    parser.add_argument("--folder", type=str, default="data",
                        help="Output folder")
    parser.add_argument("--train_pct", type=float, default=0.70,
                        help="Percentage of unique pieces for training")
    parser.add_argument("--val_pct", type=float, default=0.10,
                        help="Percentage of unique pieces for validation")
    parser.add_argument("--test_pct", type=float, default=0.20,
                        help="Percentage of unique pieces for testing")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit number of images (for testing)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    prepare_hf_dataset_paper_split(
        hf_name=args.hf_name,
        name=args.name,
        folder=args.folder,
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        max_images=args.max_images,
        random_seed=args.random_seed,
    )
