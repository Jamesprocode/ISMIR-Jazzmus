import os
import json
import ast
from ultralytics import YOLO

import numpy as np

from datasets import load_dataset
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm


def detect_staff_boxes_with_yolo(image, yolo_model):
    """
    Use YOLO to detect staff bounding boxes in the image.

    Returns:
        list of tuples: [(x1, y1, x2, y2), ...] sorted top to bottom
    """
    # Save image temporarily for YOLO
    temp_path = "temp_yolo_image.jpg"
    image.convert("RGB").save(temp_path)

    # Run YOLO inference
    results = yolo_model([temp_path])
    result = results[0]

    boxes = result.boxes
    names = result.names

    # Extract staff boxes
    staff_boxes = []
    for box, cls in zip(boxes.xyxy, boxes.cls):
        class_name = names[int(cls)]
        if class_name.lower() == "staff":
            x1, y1, x2, y2 = map(int, box)
            staff_boxes.append((x1, y1, x2, y2))

    # Sort by y position (top to bottom)
    staff_boxes.sort(key=lambda b: b[1])

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return staff_boxes


def match_yolo_boxes_to_systems(yolo_boxes, systems):
    """
    Match YOLO-detected boxes to system annotations by vertical position.

    Returns:
        list of tuples: [(yolo_box, system), ...]
    """
    matched = []

    # Sort systems by their Y position
    systems_with_y = []
    for system in systems:
        if 'bounding_box' in system:
            bbox = system['bounding_box']
            y_center = (bbox['fromY'] + bbox['toY']) / 2
            systems_with_y.append((y_center, system))

    systems_with_y.sort(key=lambda x: x[0])

    # Match YOLO boxes to systems
    if len(yolo_boxes) == len(systems_with_y):
        # Same number - match in order
        for yolo_box, (_, system) in zip(yolo_boxes, systems_with_y):
            matched.append((yolo_box, system))
    else:
        # Different number - match what we can
        min_len = min(len(yolo_boxes), len(systems_with_y))
        for i in range(min_len):
            matched.append((yolo_boxes[i], systems_with_y[i][1]))

    return matched


def save_regions(image, regions, folder, name, idx, use_yolo=False, yolo_model=None):
    """
    Save cropped regions and annotations.

    Args:
        image: PIL Image
        regions: annotation dict with 'systems' key
        folder: output folder
        name: dataset name
        idx: image index
        use_yolo: if True, use YOLO to detect bounding boxes
        yolo_model: loaded YOLO model (required if use_yolo=True)
    """
    images = []
    annotations = []

    # Check for 'systems' instead of 'both'
    if "systems" not in regions:
        print(f"Warning: No 'systems' key found in regions for image {idx}")
        print(f"Available keys: {regions.keys()}")
        return images, annotations

    # Update bounding boxes with YOLO if requested
    if use_yolo:
        if yolo_model is None:
            raise ValueError("YOLO model must be provided when use_yolo=True")

        # Detect staff boxes with YOLO
        yolo_boxes = detect_staff_boxes_with_yolo(image, yolo_model)

        # Match YOLO boxes to systems
        matched_systems = match_yolo_boxes_to_systems(yolo_boxes, regions["systems"])

        if len(yolo_boxes) != len(regions["systems"]):
            print(f"Warning: YOLO found {len(yolo_boxes)} boxes but dataset has {len(regions['systems'])} systems for image {idx}")
    else:
        # Use original bounding boxes from dataset
        matched_systems = [(None, system) for system in regions["systems"]]

    for r_idx, (yolo_box, system) in enumerate(matched_systems):
        if "bounding_box" not in system and yolo_box is None:
            print(f"Warning: No 'bounding_box' in system {r_idx} for image {idx}")
            continue

        # Get bounding box coordinates
        if use_yolo and yolo_box is not None:
            # Use YOLO-detected box
            fromx, fromy, tox, toy = yolo_box
        else:
            # Use original box from dataset
            fromx, tox, fromy, toy = (
                system["bounding_box"]["fromX"],
                system["bounding_box"]["toX"],
                system["bounding_box"]["fromY"],
                system["bounding_box"]["toY"],
            )

        # Crop and save the image
        cropped = image.crop((fromx, fromy, tox, toy))

        cropped.convert("RGB").save(f"{folder}/{name}/jpg/img_{idx}_{r_idx}_syn.jpg", "JPEG")

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


def prepare_hf_dataset(hf_name, name, folder: str = "data", folds: int = 5,
                       use_yolo: bool = False, yolo_model_path: str = None,
                       max_images: int = None):
    """
    Prepare HuggingFace dataset with optional YOLO bbox detection.

    Args:
        hf_name: HuggingFace dataset name
        name: output dataset name
        folder: output folder
        folds: number of k-folds
        use_yolo: if True, use YOLO to detect and update bounding boxes
        yolo_model_path: path to YOLO model (required if use_yolo=True)
        max_images: limit number of images to process (for testing)
    """
    images_paths = []
    annotations_paths = []

    # Load YOLO model if requested
    yolo_model = None
    if use_yolo:
        if yolo_model_path is None:
            raise ValueError("yolo_model_path must be provided when use_yolo=True")

        print("="*60)
        print("Loading YOLO model...")
        print("="*60)

        yolo_model = YOLO(yolo_model_path)
        print(f"✓ YOLO model loaded from: {yolo_model_path}")
        print(f"  Model classes: {yolo_model.names}\n")

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

    if use_yolo:
        print(f"Processing {num_images} images with YOLO bbox detection...")
    else:
        print(f"Processing {num_images} images with original bounding boxes...")

    # Process each image
    for idx in tqdm(range(num_images), desc="Processing images"):
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

        images, annotations = save_regions(
            image, regions, folder, name, idx,
            use_yolo=use_yolo, yolo_model=yolo_model
        )

        images_paths.extend(images)
        annotations_paths.extend(annotations)

    print(f"\nTotal images processed: {len(images_paths)}")
    print(f"Total annotations processed: {len(annotations_paths)}")

    if len(images_paths) == 0:
        print("ERROR: No images were processed! Check the warnings above.")
        return

    # create folds from the images and annotations paths
    print("\nCreating k-fold splits...")
    create_kfold_splits(
        images_paths,
        annotations_paths,
        n_folds=folds,
        val_pct=0.1,
        name=name,
    )
    print("✓ Dataset preparation complete!")


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
    # Set use_yolo=True to detect bounding boxes with YOLO
    # Set use_yolo=False to use original bounding boxes from dataset
    prepare_hf_dataset(
        hf_name="PRAIG/JAZZMUS_Synthetic",
        name="jazzmus_dataset_synthetic_regions",
        folder="data",
        folds=5,
        use_yolo=True,  # Change to False to use original bboxes
        yolo_model_path="/Users/jameswang/workspace/OMR-Jazz/ismiromrjazz/ISMIR-Jazzmus/weigths/yolov11s_20241108.pt",
        max_images=None  # Set to a number (e.g., 10) to test on fewer images
    )