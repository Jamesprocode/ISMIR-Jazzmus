"""
Debug Cropping / GT Mismatch Tool

Runs YOLO detection + GT system extraction WITHOUT model inference.
Compares YOLO crops vs GT systems and generates visual diagnostics.

Usage:
    python debug_cropping.py --test_split <path> --yolo_model <path> [options]
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2

from full_page_baseline import segment_staves, compute_dynamic_boundaries, deskew_image
from system_level_evaluation import extract_gt_systems
from ultralytics import YOLO


def run_yolo_detection(
    image: Image.Image,
    yolo_model: YOLO,
    confidence_threshold: float = 0.6,
) -> Tuple[List[Tuple[float, Tuple[int, int, int, int]]], dict]:
    """
    Run YOLO detection and return raw bounding boxes + class info.

    Returns:
        Tuple of:
            - staff_boxes: List of (y_center, (x1, y1, x2, y2)) for 'staff' class
            - info: Dict with detection details (all_classes, filtered_counts, etc.)
    """
    results = yolo_model(image, conf=confidence_threshold, verbose=False)
    result = results[0]

    staff_boxes = []
    filtered_counts = {}
    all_detections = []

    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        class_name = result.names[int(cls)].lower()
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        confidence = float(conf.cpu().numpy())
        y_center = (y1 + y2) / 2

        all_detections.append({
            'class': class_name,
            'bbox': (x1, y1, x2, y2),
            'y_center': y_center,
            'confidence': confidence,
        })

        if class_name == "staff":
            staff_boxes.append((y_center, (x1, y1, x2, y2)))
        else:
            filtered_counts[class_name] = filtered_counts.get(class_name, 0) + 1

    # Sort by vertical position
    staff_boxes.sort(key=lambda x: x[0])

    info = {
        'all_detections': all_detections,
        'filtered_counts': filtered_counts,
        'n_staff': len(staff_boxes),
        'available_classes': list(result.names.values()),
    }

    return staff_boxes, info


def draw_annotated_page(
    image: Image.Image,
    staff_boxes: List[Tuple[float, Tuple[int, int, int, int]]],
    crop_boundaries: List[Tuple[int, int, int, int]],
    n_gt_systems: int,
    gt_system_indices: List[int],
    info: dict,
    title: str = "",
) -> Image.Image:
    """
    Draw annotated image showing YOLO bboxes, crop boundaries, and GT count.
    """
    # Work on a copy
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    w, h = annotated.size

    # Draw crop boundaries (green dashed regions)
    for i, (x1, crop_top, x2, crop_bottom) in enumerate(crop_boundaries):
        # Draw crop region outline
        draw.rectangle(
            [x1, crop_top, x2, crop_bottom],
            outline='green', width=2
        )
        # Label crop index
        draw.text(
            (x1 + 5, crop_top + 5),
            f"Crop {i}",
            fill='green',
        )

    # Draw YOLO staff bboxes (red)
    for i, (y_center, (x1, y1, x2, y2)) in enumerate(staff_boxes):
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text(
            (x2 - 80, y1 + 5),
            f"YOLO {i}",
            fill='red',
        )

    # Draw info panel at top
    panel_height = 80
    panel = Image.new('RGB', (w, panel_height), 'white')
    panel_draw = ImageDraw.Draw(panel)

    match_status = "MATCH" if len(staff_boxes) == n_gt_systems else "MISMATCH"
    color = 'green' if match_status == "MATCH" else 'red'

    lines = [
        f"{title}",
        f"YOLO: {len(staff_boxes)} staff detections | GT: {n_gt_systems} systems | {match_status}",
        f"GT indices (after !LO filter): {gt_system_indices}",
    ]
    if info['filtered_counts']:
        filtered_str = ", ".join(f"{k}={v}" for k, v in info['filtered_counts'].items())
        lines.append(f"Filtered out: {filtered_str}")

    for i, line in enumerate(lines):
        panel_draw.text((10, 5 + i * 18), line, fill=color if i == 1 else 'black')

    # Combine panel + annotated image
    combined = Image.new('RGB', (w, h + panel_height), 'white')
    combined.paste(panel, (0, 0))
    combined.paste(annotated, (0, panel_height))

    return combined


def save_individual_crops(
    image: Image.Image,
    crop_boundaries: List[Tuple[int, int, int, int]],
    gt_systems: List[str],
    output_dir: Path,
    image_name: str,
):
    """Save individual crop images side-by-side with their GT text."""
    n_crops = len(crop_boundaries)
    n_gt = len(gt_systems)
    n_eval = min(n_crops, n_gt)

    for i in range(max(n_crops, n_gt)):
        # Save crop image if available
        if i < n_crops:
            x1, crop_top, x2, crop_bottom = crop_boundaries[i]
            crop = image.crop((x1, crop_top, x2, crop_bottom))
            crop.save(output_dir / f"{image_name}_crop{i}.jpg")

        # Save GT text if available
        if i < n_gt:
            with open(output_dir / f"{image_name}_gt_system{i}.txt", 'w') as f:
                f.write(gt_systems[i])

        # Mark unmatched
        if i >= n_crops:
            with open(output_dir / f"{image_name}_MISSING_crop{i}.txt", 'w') as f:
                f.write(f"No YOLO crop for GT system {i}\n\nGT content:\n{gt_systems[i]}")
        elif i >= n_gt:
            # Extra crop with no GT
            x1, crop_top, x2, crop_bottom = crop_boundaries[i]
            crop = image.crop((x1, crop_top, x2, crop_bottom))
            crop.save(output_dir / f"{image_name}_EXTRA_crop{i}.jpg")


def process_single_image(
    img_path: str,
    gt_path: str,
    yolo_model: YOLO,
    confidence_threshold: float = 0.6,
    deskew: bool = True,
    top_ratio: float = 0.7,
    bottom_ratio: float = 0.3,
):
    """Process a single image and return diagnostic info."""
    # Load image
    image = Image.open(img_path).convert("RGB")

    # Optional deskew
    if deskew:
        image = deskew_image(image)

    # Run YOLO
    staff_boxes, detection_info = run_yolo_detection(image, yolo_model, confidence_threshold)

    # Compute crop boundaries
    crop_boundaries = compute_dynamic_boundaries(
        staff_boxes, image.height, top_ratio, bottom_ratio
    )

    # Extract GT systems
    with open(gt_path, 'r') as f:
        full_page_gt = f.read()
    gt_systems, gt_system_indices = extract_gt_systems(full_page_gt)

    return {
        'image': image,
        'img_path': img_path,
        'gt_path': gt_path,
        'staff_boxes': staff_boxes,
        'crop_boundaries': crop_boundaries,
        'detection_info': detection_info,
        'gt_systems': gt_systems,
        'gt_system_indices': gt_system_indices,
        'n_yolo': len(staff_boxes),
        'n_gt': len(gt_systems),
        'match': len(staff_boxes) == len(gt_systems),
    }


def main():
    parser = argparse.ArgumentParser(description="Debug YOLO cropping vs GT system alignment")
    parser.add_argument("--test_split", required=True, help="Path to test split file")
    parser.add_argument("--yolo_model", required=True, help="Path to YOLO model weights")
    parser.add_argument("--output_dir", default="./debug_crops", help="Output directory for diagnostics")
    parser.add_argument("--confidence", type=float, default=0.6, help="YOLO confidence threshold")
    parser.add_argument("--deskew", action="store_true", default=True, help="Apply deskew correction")
    parser.add_argument("--no_deskew", action="store_true", help="Disable deskew correction")
    parser.add_argument("--save_all", action="store_true", help="Save annotated images for ALL samples (not just mismatches)")
    parser.add_argument("--save_crops", action="store_true", help="Save individual crop images")
    parser.add_argument("--single", type=str, default=None, help="Process only a single image (by name, e.g. 'img_172')")
    args = parser.parse_args()

    if args.no_deskew:
        args.deskew = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test split
    with open(args.test_split, 'r') as f:
        test_pairs = [line.strip().split() for line in f.readlines() if line.strip()]

    # Filter to single image if requested
    if args.single:
        test_pairs = [p for p in test_pairs if args.single in p[0]]
        if not test_pairs:
            print(f"No images matching '{args.single}' found in test split")
            return

    print(f"Processing {len(test_pairs)} test samples")
    print(f"YOLO model: {args.yolo_model}")
    print(f"Confidence: {args.confidence}")
    print(f"Deskew: {args.deskew}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load YOLO model once
    yolo_model = YOLO(args.yolo_model)

    # Process all images
    results = []
    mismatches = []

    for img_path, gt_path in test_pairs:
        img_name = Path(img_path).stem
        print(f"\n{img_name}:", end=" ")

        try:
            result = process_single_image(
                img_path, gt_path, yolo_model,
                confidence_threshold=args.confidence,
                deskew=args.deskew,
            )
            results.append(result)

            status = "OK" if result['match'] else f"MISMATCH (YOLO={result['n_yolo']}, GT={result['n_gt']})"
            print(status)

            if result['detection_info']['filtered_counts']:
                filtered_str = ", ".join(
                    f"{k}={v}" for k, v in result['detection_info']['filtered_counts'].items()
                )
                print(f"  Filtered: {filtered_str}")

            if not result['match']:
                mismatches.append(result)

            # Save annotated image for mismatches (or all if --save_all)
            if not result['match'] or args.save_all:
                subdir = output_dir / ("all" if args.save_all else "mismatches")
                subdir.mkdir(parents=True, exist_ok=True)

                annotated = draw_annotated_page(
                    result['image'],
                    result['staff_boxes'],
                    result['crop_boundaries'],
                    result['n_gt'],
                    result['gt_system_indices'],
                    result['detection_info'],
                    title=img_name,
                )
                annotated.save(subdir / f"{img_name}_annotated.jpg")

            # Save individual crops if requested
            if args.save_crops and (not result['match'] or args.save_all):
                crop_dir = output_dir / "individual_crops" / img_name
                crop_dir.mkdir(parents=True, exist_ok=True)
                save_individual_crops(
                    result['image'],
                    result['crop_boundaries'],
                    result['gt_systems'],
                    crop_dir,
                    img_name,
                )

        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary report
    print("\n" + "=" * 60)
    print("CROPPING DEBUG SUMMARY")
    print("=" * 60)

    total = len(results)
    matched = sum(1 for r in results if r['match'])
    print(f"\nTotal samples: {total}")
    print(f"Matched (YOLO == GT): {matched}/{total}")
    print(f"Mismatched: {total - matched}/{total}")

    total_yolo = sum(r['n_yolo'] for r in results)
    total_gt = sum(r['n_gt'] for r in results)
    print(f"\nTotal YOLO crops: {total_yolo}")
    print(f"Total GT systems: {total_gt}")
    print(f"Difference: {total_yolo - total_gt}")

    if mismatches:
        print(f"\nMismatch details:")
        print(f"  {'Image':<20} {'YOLO':>6} {'GT':>6} {'Diff':>6}")
        print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6}")
        for r in sorted(mismatches, key=lambda x: abs(x['n_yolo'] - x['n_gt']), reverse=True):
            img_name = Path(r['img_path']).stem
            diff = r['n_yolo'] - r['n_gt']
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            print(f"  {img_name:<20} {r['n_yolo']:>6} {r['n_gt']:>6} {diff_str:>6}")

    # Per-image breakdown table
    print(f"\nFull breakdown:")
    print(f"  {'Image':<20} {'YOLO':>6} {'GT':>6} {'Status':<12} {'Filtered'}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*12} {'-'*20}")
    for r in results:
        img_name = Path(r['img_path']).stem
        status = "OK" if r['match'] else "MISMATCH"
        filtered = ", ".join(f"{k}={v}" for k, v in r['detection_info']['filtered_counts'].items()) or "-"
        print(f"  {img_name:<20} {r['n_yolo']:>6} {r['n_gt']:>6} {status:<12} {filtered}")

    # Save summary as JSON
    summary = {
        'total_samples': total,
        'matched': matched,
        'mismatched': total - matched,
        'total_yolo_crops': total_yolo,
        'total_gt_systems': total_gt,
        'confidence_threshold': args.confidence,
        'deskew': args.deskew,
        'per_image': [
            {
                'image': Path(r['img_path']).name,
                'n_yolo': r['n_yolo'],
                'n_gt': r['n_gt'],
                'match': r['match'],
                'gt_indices': r['gt_system_indices'],
                'filtered': r['detection_info']['filtered_counts'],
            }
            for r in results
        ]
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'summary.json'}")

    if not results[0]['match'] or args.save_all:
        print(f"Annotated images saved to {output_dir}/")


if __name__ == "__main__":
    main()
