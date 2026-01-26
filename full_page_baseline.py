"""
YOLO-based Staff Segmentation for Jazz Lead Sheets

Simple function to detect and crop staff systems from full-page images.
"""

from typing import List, Tuple
from PIL import Image
from jazzmus.dataset.data_preprocessing import convert_img_to_tensor
from ultralytics import YOLO
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Union
import sys
import os
from jazzmus.smt_trainer import SMT_Trainer
from jazzmus.dataset.tokenizer import untokenize
from inference import FullPageInference
import cv2
import numpy as np


def segment_staves(
    image_path: str,
    yolo_model_path: str,
    confidence_threshold: float = 0.5
) -> List[Image.Image]:
    """
    Detect staff systems in a full-page image and return cropped regions.

    Args:
        image_path: Path to the full-page jazz lead sheet image
        yolo_model_path: Path to YOLO model weights (.pt file)
        confidence_threshold: Minimum confidence for staff detection (default: 0.5)

    Returns:
        List of PIL Images, one per detected staff system, sorted top-to-bottom
    """
    # Load YOLO model
    model = YOLO(yolo_model_path)

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Run detection
    results = model(image_path, conf=confidence_threshold, verbose=False)
    result = results[0]

    # Extract staff bounding boxes
    staff_boxes = []
    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
        class_name = result.names[int(cls)]
        if class_name.lower() == "staff":
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            y_center = (y1 + y2) / 2
            staff_boxes.append((y_center, (x1, y1, x2, y2)))

    #extend top and bottom of each box by 10 pixels
    extended_staff_boxes = []
    for y_center, (x1, y1, x2, y2) in staff_boxes:
        extended_y1 = max(0, y1 - 30)
        extended_y2 = min(image.height, y2 + 30)
        extended_staff_boxes.append((y_center, (x1, extended_y1, x2, extended_y2)))
    staff_boxes = extended_staff_boxes

    # Sort by vertical position (top to bottom)
    staff_boxes.sort(key=lambda x: x[0])
    

    # Crop each system
    cropped_systems = []
    for _, (x1, y1, x2, y2) in staff_boxes:
        cropped = image.crop((x1, y1, x2, y2))
        cropped_systems.append(cropped)

    print(f"Detected and cropped {len(cropped_systems)} staff systems")
    return cropped_systems

def load_model(checkpoint_path: str, device: str = "cpu") -> SMT_Trainer:
    """
    Load a trained SMT model from checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file
        device: Device to load model on ('cpu', 'cuda', 'mps')

    Returns:
        Loaded SMT_Trainer model
    """
    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    model = SMT_Trainer.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    model.eval()
    model.to(device)

    print(f"Model loaded successfully on {device}")
    return model


def preprocess_image(image: Image.Image, fixed_img_height: int = 128, max_fix_img_width: int = 1000) -> torch.Tensor:
    """
    Preprocess a staff system image for model input.

    Args:
        image: PIL Image of a staff system
        max_height: Maximum height for the model
        max_width: Maximum width for the model

    Returns:
        Preprocessed tensor (1, 1, H, W)
    """
    # Convert to grayscale
    image = np.array(image.convert('L'))

    # Get original dimensions
    original_height, original_width = image.shape

    # Resize with aspect ratio preservation (matching inference.py logic)
    new_height = fixed_img_height
    new_width = int(np.ceil(original_width * fixed_img_height / original_height))

    # Cap width at max
    if new_width > max_fix_img_width:
        new_width = max_fix_img_width


    image = cv2.resize(image, (new_width, new_height))

    # Convert to tensor using the same pipeline as training
    # This applies: ToPILImage → Grayscale → ToTensor
    img_tensor = convert_img_to_tensor(image)  # Returns (C, H, W) = (1, H, W)
    img_tensor = img_tensor.unsqueeze(0)    # Add batch dimension: (1, 1, H, W)

    # Pad to minimum dimensions using batch_preparation_img2seq logic (lines 105-106)
    # This matches what happens during training when batch_size=1
    pad_height = max(32, new_height)      # At least 32 (from batch_preparation_img2seq)
    pad_width = max(1000, new_width)      # At least 1000 (from batch_preparation_img2seq)

    padded = torch.ones(1, 1, pad_height, pad_width)
    padded[:, :, :new_height, :new_width] = img_tensor


    # # Pad to exact dimensions
    # padded = Image.new('L', (max_width, max_height), color=255)  # white background
    # padded.paste(image, (0, 0))

    # # Convert to tensor and normalize
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # Invert: black text on white -> white text on black
    #     transforms.Lambda(lambda x: 1 - x),
    # ])

    # tensor = transform(padded)
    # tensor = tensor.unsqueeze(0)  # Add batch dimension: (1, 1, H, W)

    return padded

def recognize_system(
    image: Image.Image,
    model: SMT_Trainer,
    device: str = "cuda"
) -> str:
    """
    Recognize a single staff system and return **kern prediction.

    Args:
        image: PIL Image of a staff system
        model: Loaded SMT_Trainer model
        device: Device for inference

    Returns:
        Predicted **kern string for this system
    """
    # Preprocess
    input_tensor = preprocess_image(image).to(device)

    # Run inference
    with torch.no_grad():
        predicted_sequence, _ = model.model.predict(input=input_tensor[0])

        predicted_tokens = predicted_sequence[0]  # Get the first (and only) sequence

        if isinstance(predicted_tokens[0], int):
            # If tokens are integers, look them up in i2w
            token_strs = [model.model.i2w.get(int(t), "<unk>") for t in predicted_tokens]
        else:
            # If tokens are already strings, use them directly
            token_strs = [str(t) for t in predicted_tokens]     

    # Decode to **kern
    kern_prediction = untokenize(token_strs)

    return kern_prediction

def recognize_systems(
    images: List[Image.Image],
    model: SMT_Trainer,
    device: str = "cpu"
) -> List[str]:
    """
    Recognize multiple staff systems.

    Args:
        images: List of PIL Images (staff systems)
        model: Loaded SMT_Trainer model
        device: Device for inference

    Returns:
        List of **kern predictions, one per system
    """
    predictions = []

    print(f"\nRecognizing {len(images)} systems...")
    for i, image in enumerate(images):
        print(f"  System {i+1}/{len(images)}...", end=" ")
        kern = recognize_system(image, model, device)
        predictions.append(kern)
        print("✓")

    print(f"Recognition complete!")
    return predictions

"""
Kern concatenation logic to merge system-level predictions into full-page **kern.

This module handles merging individual system **kern predictions into a complete
full-page **kern representation, handling headers and linebreaks appropriately.
"""

from typing import List


def concatenate_systems(system_kerns: List[str]) -> str:
    """
    Concatenate system-level **kern predictions into full-page **kern.

    Args:
        system_kerns: List of **kern strings, one per system (top to bottom)

    Returns:
        Full-page **kern string with linebreak markers

    Strategy:
        1. Use headers from the first system
        2. For subsequent systems, strip headers and add linebreak markers
        3. Remove all "b:none" lines
    """
    if not system_kerns:
        return ""

    if len(system_kerns) == 1:
        # Remove b:none lines from single system
        lines = [l for l in system_kerns[0].strip().split('\n') if 'b:none' not in l]
        return '\n'.join(lines)

    # Parse first system (keep all headers, remove b:none)
    lines = system_kerns[0].strip().split('\n')
    full_kern_lines = []

    # Add all lines from first system (except b:none)
    for line in lines:
        if 'b:none' not in line:
            full_kern_lines.append(line)

    # Add linebreak marker after first system
    full_kern_lines.append("!!linebreak:original")

    # Process remaining systems
    for system_kern in system_kerns[1:]:
        lines = system_kern.strip().split('\n')

        # Skip header lines (lines starting with *, !, or **)
        content_started = False
        for line in lines:
            # Skip b:none lines
            if 'b:none' in line:
                continue

            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Skip headers (keep only musical content)
            if stripped.startswith('**') or stripped.startswith('*-'):
                # Skip spine definitions and terminations
                continue
            elif stripped.startswith('*') and not content_started:
                # Skip initial metadata lines
                continue
            elif stripped.startswith('!') and not content_started:
                # Skip initial comments
                continue
            else:
                # This is content
                content_started = True
                full_kern_lines.append(line)

        # Add linebreak marker after this system
        full_kern_lines.append("!!linebreak:original")

    # Remove the last linebreak marker and add proper ending
    if full_kern_lines and full_kern_lines[-1] == "!!linebreak:original":
        full_kern_lines.pop()

    # Add spine terminators at the end
    full_kern_lines.append("*-\t*-")

    return '\n'.join(full_kern_lines)


if __name__ == "__main__":
    from tqdm import tqdm
    from jazzmus.dataset.eval_functions import compute_poliphony_metrics

    checkpint_path = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/weights/smt_sys_best/smt_pre_syn_medium.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model_path = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/yolo_weigths/yolov11s_20241108.pt"
    test_split_file = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/data/jazzmus_fullpage/splits/test_0.txt"

    # Load test split
    with open(test_split_file, 'r') as f:
        test_pairs = [line.strip().split() for line in f.readlines()]

    print(f"Loaded {len(test_pairs)} test samples")

    # Load model once
    print("Loading model...")
    inference_model = FullPageInference(checkpint_path, device=device)
    print("✓ Model loaded\n")

    # Collect all predictions and ground truths
    all_predictions = []
    all_ground_truths = []
    per_sample_metrics = []

    for img_path, gt_path in tqdm(test_pairs, desc="Processing test set"):
        try:
            # Step 1: Segment staves with YOLO
            cropped_systems = segment_staves(
                image_path=img_path,
                yolo_model_path=yolo_model_path,
                confidence_threshold=0.5
            )

            # Step 2: Recognize each system
            system_kerns = []
            for system_image in cropped_systems:
                # Convert PIL Image to numpy array (grayscale) for inference_model.predict()
                system_array = np.array(system_image.convert('L'))
                result = inference_model.predict(system_array)
                system_kerns.append(result['prediction'])

            # Step 3: Concatenate into full-page **kern
            full_page_kern = concatenate_systems(system_kerns)

            # Step 4: Load ground truth
            with open(gt_path, 'r') as f:
                ground_truth = f.read()

            all_predictions.append(full_page_kern)
            all_ground_truths.append(ground_truth)

            # Compute per-sample metrics
            sample_cer, sample_ser, sample_ler = compute_poliphony_metrics([full_page_kern], [ground_truth])
            per_sample_metrics.append({
                'cer': sample_cer,
                'ser': sample_ser,
                'ler': sample_ler,
                'image': img_path
            })

        except Exception as e:
            print(f"\n✗ Failed on {img_path}: {e}")
            continue

    # Compute aggregate metrics (all predictions concatenated)
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (All predictions concatenated)")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(all_predictions)}/{len(test_pairs)}")

    cer_agg, ser_agg, ler_agg = compute_poliphony_metrics(all_predictions, all_ground_truths)

    print(f"\nCER (Character Error Rate): {cer_agg:.2f}%")
    print(f"SER (Symbol Error Rate):    {ser_agg:.2f}%")
    print(f"LER (Line Error Rate):      {ler_agg:.2f}%")
    print(f"{'='*60}\n")

    # Compute average of per-sample metrics
    print(f"{'='*60}")
    print("AVERAGE PER-SAMPLE RESULTS")
    print(f"{'='*60}")

    cer_mean = np.mean([m['cer'] for m in per_sample_metrics])
    ser_mean = np.mean([m['ser'] for m in per_sample_metrics])
    ler_mean = np.mean([m['ler'] for m in per_sample_metrics])

    cer_std = np.std([m['cer'] for m in per_sample_metrics])
    ser_std = np.std([m['ser'] for m in per_sample_metrics])
    ler_std = np.std([m['ler'] for m in per_sample_metrics])

    print(f"\nCER: {cer_mean:.2f}% (±{cer_std:.2f}%)")
    print(f"SER: {ser_mean:.2f}% (±{ser_std:.2f}%)")
    print(f"LER: {ler_mean:.2f}% (±{ler_std:.2f}%)")
    print(f"{'='*60}\n")



