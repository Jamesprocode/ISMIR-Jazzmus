from ultralytics import YOLO
from typing import List
import os
import cv2

from PIL import Image


def run_la_inference(files: List[str], save_path: str):
    os.makedirs(save_path, exist_ok=True)

    # Load the YOLO model
    model = YOLO("../yolo_weigths/yolov11s_20241108.pt")

    print("Model loaded")

    # Run inference on images
    results = model(files)  # List of Results objects

    # Process each image result
    for img_idx, result in enumerate(results):
        image_path = files[img_idx]
        image = Image.open(image_path)  # Load image using Pillow

        # Ensure image is loaded
        if image is None:
            print(f"Warning: Unable to load image {image_path}")
            continue

        boxes = result.boxes  # Bounding box outputs
        names = result.names  # Class names dictionary

        # Ensure names are present
        if not names:
            print("Warning: No class names found in model.")
            continue

        # Collect "staff" boxes
        staff_boxes = []
        for box, cls in zip(boxes.xyxy, boxes.cls):
            class_name = names[int(cls)]
            if class_name.lower() == "staff":
                x1, y1, x2, y2 = map(int, box)  # Convert box to integers
                staff_boxes.append((y1, x1, y2, x2))  # Store with y1 for sorting

        # Sort staff boxes by top-to-bottom position (y1)
        staff_boxes.sort(key=lambda b: b[0])

        # Process and save sorted staff images
        for sorted_idx, (y1, x1, y2, x2) in enumerate(staff_boxes):
            cropped_staff = image.crop((x1, y1, x2, y2))

            # Ensure crop is valid
            if cropped_staff.size == 0:
                print(f"Warning: Empty crop for {image_path} at index {sorted_idx}")
                continue

            # Save cropped staff image with ordered index and original image path
            staff_filename = f"{save_path}/{image_path.stem.replace('jpg','')}_staff_{sorted_idx}.jpg"
            # cv2.imwrite(staff_filename, cropped_staff)
            # write Pillow
            cropped_staff.save(staff_filename)
            print(f"Saved: {staff_filename}")

        # Optionally save the full result image
        result.save(filename=f"{save_path}/{image_path.stem.replace('jpg','')}_result.jpg")


# Example usage:
# run_la_inference(["path/to/image1.jpg", "path/to/image2.jpg"], "output/directory")
