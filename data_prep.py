"""
Prepare HuggingFace dataset with TWO VERSIONS:
1. Full-page images (complete pages, no cropping)
2. System-level crops (individual staffs from same pages)

Both use piece-level splits (70/10/20) and put all duplicates in training.

Output structure:
  data/
    jazzmus_fullpage/
      jpg/, gt/, splits/
    jazzmus_systems/
      jpg/, gt/, splits/
"""

import os
import json
import ast
import re
from collections import defaultdict
from pathlib import Path

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


def save_fullpage_image(image, idx, output_dir):
    """Save full-page image with naming: img_{page_idx}.jpg"""
    output_path = f"{output_dir}/jpg/img_{idx}.jpg"
    image.convert("RGB").save(output_path, "JPEG")
    return output_path


def save_system_regions(image, regions, output_dir, idx):
    """Save individual system-level crops with naming: img_{page_idx}_{system_idx}.jpg"""
    image_paths = []

    if "systems" not in regions:
        return image_paths

    for r_idx, system in enumerate(regions["systems"]):
        if "bounding_box" not in system:
            continue

        fromx, tox, fromy, toy = (
            system["bounding_box"]["fromX"],
            system["bounding_box"]["toX"],
            system["bounding_box"]["fromY"],
            system["bounding_box"]["toY"],
        )

        cropped = image.crop((fromx, fromy, tox, toy))
        output_path = f"{output_dir}/jpg/img_{idx}_{r_idx}.jpg"
        cropped.convert("RGB").save(output_path, "JPEG")
        image_paths.append((output_path, idx, r_idx))

    return image_paths


def save_ground_truth(regions, output_dir, idx, file_prefix="img"):
    """Save ground truth **kern annotations."""
    kern_content = ""

    if "systems" in regions:
        for r_idx, system in enumerate(regions["systems"]):
            if "**kern" in system:
                kern_content += system["**kern"]
                if r_idx < len(regions["systems"]) - 1:
                    kern_content += "\n"

    output_path = f"{output_dir}/gt/{file_prefix}_{idx}.txt"
    with open(output_path, "w") as f:
        f.write(kern_content)

    return output_path, kern_content


def prepare_dual_dataset(
    hf_name="PRAIG/JAZZMUS",
    fullpage_name="jazzmus_fullpage",
    systems_name="jazzmus_systems",
    folder="data",
    train_pct=0.70,
    val_pct=0.10,
    test_pct=0.20,
    max_images=None,
    random_seed=42,
):
    """Prepare dual dataset (full-page and system-level) with piece-level splits."""

    assert abs(train_pct + val_pct + test_pct - 1.0) < 0.01, "Percentages must sum to 1.0"

    # Load dataset
    print("="*70)
    print("Loading HuggingFace dataset...")
    print("="*70)
    dataset = load_dataset(hf_name, split="train", num_proc=4)
    print(f"✓ Dataset loaded: {len(dataset)} images\n")

    num_images = min(len(dataset), max_images) if max_images else len(dataset)

    # Create directories for both versions
    for name in [fullpage_name, systems_name]:
        os.makedirs(f"{folder}/{name}/jpg", exist_ok=True)
        os.makedirs(f"{folder}/{name}/gt", exist_ok=True)
        os.makedirs(f"{folder}/{name}/splits", exist_ok=True)

    # Pass 1: Extract piece titles and group
    print("="*70)
    print("Pass 1: Extracting piece titles from MusicXML...")
    print("="*70)

    piece_titles = {}
    title_to_indices = defaultdict(list)

    for idx in tqdm(range(num_images), desc="Extracting titles"):
        annotation_data = dataset[idx]["annotation"]

        if isinstance(annotation_data, str):
            try:
                annotation = json.loads(annotation_data)
            except json.JSONDecodeError:
                annotation = ast.literal_eval(annotation_data)
        else:
            annotation = annotation_data

        # Extract title from MusicXML
        if 'encodings' in annotation and 'musicxml' in annotation['encodings']:
            musicxml = annotation['encodings']['musicxml']
            title = extract_title_from_musicxml(musicxml)
            composer = extract_composer_from_musicxml(musicxml)

            if title:
                piece_id = f"{title} - {composer}" if composer else title
                piece_titles[idx] = piece_id
                title_to_indices[piece_id].append(idx)
            else:
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
    print(f"✓ Total duplicate images: {sum(len(indices) for indices in duplicates.values())}")

    # Pass 2: Split unique pieces
    print("\n" + "="*70)
    print("Pass 2: Splitting unique piece titles...")
    print("="*70)

    unique_titles = list(title_to_indices.keys())

    np.random.seed(random_seed)
    shuffled_titles = np.array(unique_titles)
    np.random.shuffle(shuffled_titles)

    n_unique = len(shuffled_titles)
    n_test = int(n_unique * test_pct)
    n_val = int(n_unique * val_pct)
    n_train = n_unique - n_test - n_val

    test_titles = shuffled_titles[:n_test].tolist()
    val_titles = shuffled_titles[n_test:n_test + n_val].tolist()
    train_titles = shuffled_titles[n_test + n_val:].tolist()

    # Assign indices based on piece titles
    test_indices = [title_to_indices[title][0] for title in test_titles]
    val_indices = [title_to_indices[title][0] for title in val_titles]

    assigned_to_test_or_val = set(test_indices + val_indices)
    train_all_indices = [idx for idx in range(num_images) if idx not in assigned_to_test_or_val]

    print(f"\n✓ Unique piece distribution:")
    print(f"  Train: {n_train} unique pieces")
    print(f"  Val:   {n_val} unique pieces")
    print(f"  Test:  {n_test} unique pieces")

    print(f"\n✓ Image distribution (with duplicates):")
    print(f"  Train: {len(train_all_indices)} images")
    print(f"  Val:   {len(val_indices)} images")
    print(f"  Test:  {len(test_indices)} images")

    # Pass 3: Process and save images
    print("\n" + "="*70)
    print("Pass 3: Processing and saving images...")
    print("="*70)

    fullpage_data = defaultdict(lambda: {"images": [], "annotations": []})
    systems_data = defaultdict(lambda: {"images": [], "annotations": []})

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

        # Determine split for this image's piece
        piece_title = piece_titles[idx]
        if idx in train_all_indices:
            split = "train"
        elif idx in val_indices:
            split = "val"
        elif idx in test_indices:
            split = "test"
        else:
            continue

        # Save full-page image
        fp_img_path = save_fullpage_image(image, idx, f"{folder}/{fullpage_name}")
        fp_gt_path, _ = save_ground_truth(annotation, f"{folder}/{fullpage_name}", idx, "img")

        fullpage_data[split]["images"].append(f"data/{fullpage_name}/jpg/img_{idx}.jpg")
        fullpage_data[split]["annotations"].append(f"data/{fullpage_name}/gt/img_{idx}.txt")

        # Save system-level crops
        system_paths = save_system_regions(image, annotation, f"{folder}/{systems_name}", idx)

        for img_path, img_idx, region_idx in system_paths:
            systems_data[split]["images"].append(img_path.replace(f"{folder}/", "data/"))

            # Save individual system ground truth
            if "systems" in annotation and region_idx < len(annotation["systems"]):
                system = annotation["systems"][region_idx]
                sys_gt_path = f"{folder}/{systems_name}/gt/img_{img_idx}_{region_idx}.txt"
                with open(sys_gt_path, "w") as f:
                    if "**kern" in system:
                        f.write(system["**kern"])
                systems_data[split]["annotations"].append(f"data/{systems_name}/gt/img_{img_idx}_{region_idx}.txt")

    # Save split files for both datasets
    print("\n" + "="*70)
    print("Pass 4: Saving split files...")
    print("="*70)

    for split in ["train", "val", "test"]:
        # Full-page splits
        with open(f"{folder}/{fullpage_name}/splits/{split}_0.txt", "w") as f:
            for img, ann in zip(fullpage_data[split]["images"], fullpage_data[split]["annotations"]):
                f.write(f"{img} {ann}\n")

        # System-level splits
        with open(f"{folder}/{systems_name}/splits/{split}_0.txt", "w") as f:
            for img, ann in zip(systems_data[split]["images"], systems_data[split]["annotations"]):
                f.write(f"{img} {ann}\n")

    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    print(f"\nFull-Page Dataset:")
    for split in ["train", "val", "test"]:
        print(f"  {split.upper()}: {len(fullpage_data[split]['images'])} images")

    print(f"\nSystem-Level Dataset:")
    for split in ["train", "val", "test"]:
        print(f"  {split.upper()}: {len(systems_data[split]['images'])} systems")

    print(f"\n✓ Full-page dataset: {folder}/{fullpage_name}/")
    print(f"✓ System-level dataset: {folder}/{systems_name}/")
    print("\n✓ Dataset preparation complete!")


if __name__ == "__main__":
    # Edit these values directly in the code
    prepare_dual_dataset(
        hf_name="PRAIG/JAZZMUS",
        fullpage_name="jazzmus_fullpage",
        systems_name="jazzmus_systems",
        folder="data",
        train_pct=0.70,
        val_pct=0.10,
        test_pct=0.20,
        max_images=None,  # Set to number to limit (e.g., 500 for testing)
        random_seed=42,
    )
