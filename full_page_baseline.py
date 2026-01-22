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


def preprocess_image(image: Image.Image, max_height: int = 128, max_width: int = 1000) -> torch.Tensor:
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
    image = image.convert('L')

    # Resize to fit max dimensions while maintaining aspect ratio
    width, height = image.size
    aspect_ratio = width / height

    if height > max_height:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_height = height
        new_width = width

    if new_width > max_width:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)

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
        Full-page **kern string with proper linebreak markers

    Strategy:
        1. Use headers from the first system
        2. For subsequent systems, strip headers and add linebreak markers
        3. Concatenate all content
    """
    if not system_kerns:
        return ""

    if len(system_kerns) == 1:
        return system_kerns[0]

    # Parse first system (keep all headers)
    lines = system_kerns[0].strip().split('\n')
    full_kern_lines = []

    # Add all lines from first system
    for line in lines:
        full_kern_lines.append(line)

    # Add linebreak marker after first system
    full_kern_lines.append("!!linebreak:original")

    # Process remaining systems
    for system_kern in system_kerns[1:]:
        lines = system_kern.strip().split('\n')

        # Skip header lines (lines starting with *, !, or **)
        content_started = False
        for line in lines:
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

        # Add linebreak marker after this system (except for last)
        full_kern_lines.append("!!linebreak:original")

    # Remove the last linebreak marker and add proper ending
    if full_kern_lines and full_kern_lines[-1] == "!!linebreak:original":
        full_kern_lines.pop()

    # Add spine terminators at the end
    full_kern_lines.append("*-\t*-")

    return '\n'.join(full_kern_lines)


checkpint_path = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/weights/smt_sys_best/smt_pre_syn_medium.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(checkpint_path, device=device)
image_path = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/data/jazzmus_fullpage/jpg/img_0.jpg"
yolo_model_path = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/yolo_weigths/yolov11s_20241108.pt"

# Step 1: Segment staves with YOLO
print("Step 1: YOLO Staff Segmentation")
print("-" * 60)
cropped_systems = segment_staves(
    image_path=image_path,
    yolo_model_path=yolo_model_path,
    confidence_threshold=0.5
    )
print(f"✓ Segmented into {len(cropped_systems)} systems\n")

# Step 2: Load SMT inference model
print("Step 2: Load Recognition Model")
print("-" * 60)
inference_model = FullPageInference(checkpint_path, device=device)
print(f"✓ Model loaded\n")

# Step 3: Recognize each system
print("Step 3: System-by-System Recognition")
print("-" * 60)
predictions = recognize_systems(cropped_systems, model, device=device)
# system_kerns = []
# for i, system_image in enumerate(cropped_systems):
#     print(f"  System {i+1}/{len(cropped_systems)}...", end=" ")
#     result = inference_model.predict(system_image)
#     system_kerns.append(result['prediction'])
#     print("✓")
print(f"✓ Recognized {len(predictions)} systems\n")

# Step 4: Concatenate into full-page **kern
print("Step 4: Concatenate Systems")
print("-" * 60)
full_page_kern = concatenate_systems(predictions)
print(f"✓ Concatenated into full-page **kern\n")
print(full_page_kern)


# predictions = recognize_systems(cropped_systems, model, device=device)
# print("testing Predictions:")
# print(predictions[0])

# full_page_kern = concatenate_systems(predictions)
# print("\nFull-page **kern prediction:")
# print(full_page_kern)


