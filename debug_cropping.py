"""
Debug Cropping / GT Mismatch Tool

Runs YOLO detection + GT system extraction WITHOUT model inference.
Compares YOLO crops vs GT systems and generates visual diagnostics
for ALL images, highlighting mismatches.

All paths hardcoded — no config needed. Just run:
    python debug_cropping.py
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw
import traceback

from full_page_baseline import compute_dynamic_boundaries, deskew_image
from system_level_evaluation import extract_gt_systems
from ultralytics import YOLO

# ── Hardcoded paths ──────────────────────────────────────────────────────────
YOLO_MODEL_PATH = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/yolo_weigths/yolov11s_20241108.pt"
SPLIT_FILES = [
    "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/data/jazzmus_fullpage/splits/train_0.txt",
    "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/data/jazzmus_fullpage/splits/val_0.txt",
    "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/data/jazzmus_fullpage/splits/test_0.txt",
]
DATA_ROOT       = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus"
OUTPUT_DIR      = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/debug/crop_debug"

CONFIDENCE_THRESHOLD = 0.2
DESKEW = True
TOP_RATIO = 0.7
BOTTOM_RATIO = 0.3
# ─────────────────────────────────────────────────────────────────────────────


def run_yolo_detection(
    image: Image.Image,
    yolo_model: YOLO,
    confidence_threshold: float = 0.6,
) -> Tuple[List[Tuple[float, Tuple[int, int, int, int]]], dict]:
    """Run YOLO and return staff bboxes + detection info."""
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

    staff_boxes.sort(key=lambda x: x[0])

    # Merge duplicate detections on the same line (partial + full width)
    n_before = len(staff_boxes)
    staff_boxes = merge_overlapping_staff_boxes(staff_boxes)
    n_merged = n_before - len(staff_boxes)

    # Interpolate virtual boxes where large gaps suggest missed systems
    staff_boxes, interp_info = interpolate_missing_systems(staff_boxes)

    info = {
        'all_detections': all_detections,
        'filtered_counts': filtered_counts,
        'n_staff': len(staff_boxes),
        'n_staff_before_merge': n_before,
        'n_merged': n_merged,
        'n_virtual': interp_info['n_virtual'],
        'virtual_indices': interp_info['virtual_indices'],
        'interpolation': interp_info,
        'available_classes': list(result.names.values()),
    }
    return staff_boxes, info


def merge_overlapping_staff_boxes(
    staff_boxes: List[Tuple[float, Tuple[int, int, int, int]]],
    y_overlap_threshold: float = 0.6,
) -> List[Tuple[float, Tuple[int, int, int, int]]]:
    """
    Deduplicate staff detections that overlap vertically (same line detected twice).

    When YOLO detects a partial-width and full-width bbox for the same staff,
    keep the narrower one to be consistent with training data.

    Args:
        staff_boxes: Sorted list of (y_center, (x1, y1, x2, y2))
        y_overlap_threshold: Min vertical IoU to consider same line (default 0.5)

    Returns:
        Deduplicated list of staff boxes (narrower kept)
    """
    if len(staff_boxes) <= 1:
        return staff_boxes

    keep = []
    used = set()

    for i in range(len(staff_boxes)):
        if i in used:
            continue

        yc_i, (x1_i, y1_i, x2_i, y2_i) = staff_boxes[i]
        width_i = x2_i - x1_i

        # Find all boxes overlapping vertically with this one
        best_idx = i
        best_width = width_i

        for j in range(i + 1, len(staff_boxes)):
            if j in used:
                continue
            yc_j, (x1_j, y1_j, x2_j, y2_j) = staff_boxes[j]

            v_iou = iou_1d(y1_i, y2_i, y1_j, y2_j)
            if v_iou >= y_overlap_threshold:
                # Same line — keep the narrower one (matches training data)
                width_j = x2_j - x1_j
                if width_j < best_width:
                    used.add(best_idx)
                    best_idx = j
                    best_width = width_j
                else:
                    used.add(j)

        keep.append(staff_boxes[best_idx])
        used.add(best_idx)

    keep.sort(key=lambda x: x[0])
    return keep


def iou_1d(a_top, a_bot, b_top, b_bot):
    """1-D IoU of two vertical intervals."""
    inter = max(0, min(a_bot, b_bot) - max(a_top, b_top))
    union = max(a_bot, b_bot) - min(a_top, b_top)
    return inter / union if union > 0 else 0.0


def interpolate_missing_systems(
    staff_boxes: List[Tuple[float, Tuple[int, int, int, int]]],
    gap_factor: float = 2.5,
) -> Tuple[List[Tuple[float, Tuple[int, int, int, int]]], dict]:
    """
    Insert virtual staff boxes where suspiciously large vertical gaps exist.

    After YOLO detection + merge, some systems may be missed entirely.
    A gap triggers interpolation only when BOTH conditions are met:
      1) gap > gap_factor × median_gap   (relatively unusual for this page)
      2) gap > median_height + median_gap (physically large enough to hold a system)

    All statistics (median_gap, median_height) are computed per-page so pages
    with naturally tight or wide spacing each get their own threshold.

    Args:
        staff_boxes: Sorted list of (y_center, (x1, y1, x2, y2))
        gap_factor: Multiplier on median gap for the relative test (default 2.5)

    Returns:
        (updated_staff_boxes, interpolation_info)
    """
    if len(staff_boxes) < 3:
        # Need at least 3 real boxes to have a reliable median gap
        return staff_boxes, {'n_virtual': 0, 'virtual_indices': []}

    # ── Per-page statistics from real detections ──
    heights = [y2 - y1 for _, (_, y1, _, y2) in staff_boxes]
    median_h = int(np.median(heights))

    # Gaps = space between bottom of box i and top of box i+1
    gaps = []
    for i in range(len(staff_boxes) - 1):
        _, (_, _, _, y2_cur) = staff_boxes[i]
        _, (_, y1_next, _, _) = staff_boxes[i + 1]
        gaps.append(y1_next - y2_cur)
    median_gap = float(np.median(gaps))

    if median_gap <= 0:
        return staff_boxes, {'n_virtual': 0, 'virtual_indices': []}

    # Two thresholds — BOTH must be exceeded
    relative_thresh = gap_factor * median_gap         # unusual for this page
    absolute_thresh = median_h + median_gap            # room for ≥1 system
    threshold = max(relative_thresh, absolute_thresh)  # whichever is stricter

    # ── Build new list, inserting virtual boxes only in truly huge gaps ──
    new_boxes = [staff_boxes[0]]
    virtual_ycenters = set()

    for i in range(len(staff_boxes) - 1):
        _, (x1_cur, _, x2_cur, y2_cur) = staff_boxes[i]
        _, (x1_next, y1_next, x2_next, _) = staff_boxes[i + 1]

        gap = y1_next - y2_cur

        if gap > threshold:
            # How many systems fit?  stride = one system height + normal gap
            stride = median_h + median_gap
            n_insert = max(1, int(round(gap / stride)) - 1)

            # Spread evenly: divide the gap into (n_insert + 1) equal sub-gaps
            spacing = gap / (n_insert + 1)

            # Width = average of the two real neighbours
            avg_x1 = (x1_cur + x1_next) // 2
            avg_x2 = (x2_cur + x2_next) // 2

            for k in range(1, n_insert + 1):
                vc_y = y2_cur + k * spacing
                vy1 = max(y2_cur + 1, int(vc_y - median_h / 2))
                vy2 = vy1 + median_h
                y_center = (vy1 + vy2) / 2
                virtual_ycenters.add(y_center)
                new_boxes.append((y_center, (avg_x1, vy1, avg_x2, vy2)))

        new_boxes.append(staff_boxes[i + 1])

    new_boxes.sort(key=lambda x: x[0])

    virtual_indices = [
        i for i, (yc, _) in enumerate(new_boxes) if yc in virtual_ycenters
    ]

    info = {
        'n_virtual': len(virtual_indices),
        'virtual_indices': virtual_indices,
        'median_gap': median_gap,
        'median_height': float(median_h),
        'relative_thresh': relative_thresh,
        'absolute_thresh': absolute_thresh,
        'effective_thresh': threshold,
    }

    return new_boxes, info


def match_crops_to_gt(
    crop_boundaries: List[Tuple[int, int, int, int]],
    gt_systems: List[str],
    image_height: int,
) -> dict:
    """
    Match YOLO crops to GT systems by sequential order (top-to-bottom).

    Both YOLO crops and GT systems are already sorted top-to-bottom,
    so we match them 1-to-1 in order. Extra crops or GT systems at the
    end are flagged as unmatched.
    """
    n_crops = len(crop_boundaries)
    n_gt = len(gt_systems)
    n_matched = min(n_crops, n_gt)

    matched = []  # (crop_idx, gt_idx, overlap_ratio)
    for i in range(n_matched):
        matched.append((i, i, 1.0))

    unmatched_crops = list(range(n_matched, n_crops))  # extra YOLO detections
    unmatched_gt = list(range(n_matched, n_gt))         # missing GT systems

    # Build GT slots for visualization (evenly divided, just for drawing)
    gt_slots = []
    slot_height = image_height / max(n_gt, 1)
    for i in range(n_gt):
        gt_slots.append((int(i * slot_height), int((i + 1) * slot_height)))

    return {
        'matched': matched,
        'unmatched_crops': unmatched_crops,
        'unmatched_gt': unmatched_gt,
        'gt_slots': gt_slots,
    }


def draw_debug_page(
    image: Image.Image,
    staff_boxes: List[Tuple[float, Tuple[int, int, int, int]]],
    crop_boundaries: List[Tuple[int, int, int, int]],
    match_info: dict,
    n_gt: int,
    gt_system_indices: List[int],
    detection_info: dict,
    title: str = "",
) -> Image.Image:
    """
    Draw a richly annotated debug image:
      - Red boxes = YOLO staff bboxes (with confidence)
      - Green boxes = crop boundaries (with index)
      - Blue dashed lines = estimated GT slots
      - Yellow highlight on unmatched crops / missing GT
      - Info panel at top with counts + match score
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    unmatched_crop_set = set(match_info['unmatched_crops'])
    unmatched_gt_set = set(match_info['unmatched_gt'])
    match_dict = {ci: (gi, iou) for ci, gi, iou in match_info['matched']}

    # 1) Draw estimated GT slots (blue dashed horizontal lines)
    for gi, (gt_top, gt_bot) in enumerate(match_info['gt_slots']):
        color = 'orange' if gi in unmatched_gt_set else 'blue'
        # Top line
        for x in range(0, w, 20):
            draw.line([(x, gt_top), (min(x + 10, w), gt_top)], fill=color, width=2)
        # Bottom line
        for x in range(0, w, 20):
            draw.line([(x, gt_bot), (min(x + 10, w), gt_bot)], fill=color, width=2)
        label = f"GT slot {gi}"
        if gi in unmatched_gt_set:
            label += " [MISSING CROP]"
        draw.text((10, gt_top + 4), label, fill=color)

    # 2) Draw crop boundaries
    for ci, (x1, ct, x2, cb) in enumerate(crop_boundaries):
        is_unmatched = ci in unmatched_crop_set
        outline_color = 'yellow' if is_unmatched else 'green'
        lw = 4 if is_unmatched else 2

        # Semi-transparent fill for unmatched
        if is_unmatched:
            overlay = Image.new('RGBA', annotated.size, (0, 0, 0, 0))
            ov_draw = ImageDraw.Draw(overlay)
            ov_draw.rectangle([x1, ct, x2, cb], fill=(255, 255, 0, 40))
            annotated = Image.alpha_composite(annotated.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(annotated)

        draw.rectangle([x1, ct, x2, cb], outline=outline_color, width=lw)

        tag = f"Crop {ci}"
        if ci in match_dict:
            gi, iou = match_dict[ci]
            tag += f" -> GT {gi}  IoU={iou:.2f}"
        else:
            tag += " [EXTRA - NO GT]"
        draw.text((x1 + 5, ct + 5), tag, fill=outline_color)

    # 3) Draw YOLO staff bboxes (red) and virtual boxes (cyan dashed)
    staff_dets = [d for d in detection_info['all_detections'] if d['class'] == 'staff']
    virtual_set = set(detection_info.get('virtual_indices', []))
    for si, (y_center, (x1, y1, x2, y2)) in enumerate(staff_boxes):
        if si in virtual_set:
            # Virtual box — dashed cyan outline
            for seg_x in range(x1, x2, 20):
                draw.line([(seg_x, y1), (min(seg_x + 10, x2), y1)], fill='cyan', width=3)
                draw.line([(seg_x, y2), (min(seg_x + 10, x2), y2)], fill='cyan', width=3)
            for seg_y in range(y1, y2, 20):
                draw.line([(x1, seg_y), (x1, min(seg_y + 10, y2))], fill='cyan', width=3)
                draw.line([(x2, seg_y), (x2, min(seg_y + 10, y2))], fill='cyan', width=3)
            draw.text((x2 - 180, y1 + 5), f"VIRTUAL {si} (interpolated)", fill='cyan')
        else:
            # Real YOLO detection
            # Find matching real detection for confidence
            real_idx = si - sum(1 for v in virtual_set if v < si)
            conf = staff_dets[real_idx]['confidence'] if real_idx < len(staff_dets) else 0.0
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            draw.text((x2 - 140, y1 + 5), f"YOLO {si} ({conf:.2f})", fill='red')

    # 4) Info panel
    n_yolo = len(staff_boxes)
    n_virtual = detection_info.get('n_virtual', 0)
    is_match = n_yolo == n_gt
    match_str = "MATCH" if is_match else f"MISMATCH  (YOLO {n_yolo} vs GT {n_gt})"
    panel_lines = [
        f"{title}  --  {match_str}",
        f"YOLO staff: {n_yolo} (real: {n_yolo - n_virtual}, virtual: {n_virtual})  |  GT systems: {n_gt}  |  GT indices kept: {gt_system_indices}",
        f"Unmatched crops: {match_info['unmatched_crops']}  |  Missing GT: {match_info['unmatched_gt']}",
    ]
    if detection_info['filtered_counts']:
        filt = ", ".join(f"{k}={v}" for k, v in detection_info['filtered_counts'].items())
        panel_lines.append(f"Filtered detections: {filt}")

    panel_h = 22 * len(panel_lines) + 10
    panel = Image.new('RGB', (w, panel_h), 'white')
    pdraw = ImageDraw.Draw(panel)
    for i, line in enumerate(panel_lines):
        color = 'red' if not is_match and i == 0 else 'black'
        pdraw.text((10, 5 + i * 22), line, fill=color)

    combined = Image.new('RGB', (w, h + panel_h), 'white')
    combined.paste(panel, (0, 0))
    combined.paste(annotated, (0, panel_h))
    return combined


def save_crop_details(
    image: Image.Image,
    crop_boundaries: List[Tuple[int, int, int, int]],
    gt_systems: List[str],
    match_info: dict,
    detection_info: dict,
    staff_boxes: List[Tuple[float, Tuple[int, int, int, int]]],
    out_dir: Path,
    img_name: str,
):
    """Save every crop image + GT text, clearly marking unmatched ones.
    Also saves all YOLO detections with confidence and class labels."""
    match_dict = {ci: (gi, iou) for ci, gi, iou in match_info['matched']}

    # Save a summary of ALL YOLO detections for this image
    all_dets = detection_info['all_detections']
    with open(out_dir / f"{img_name}_yolo_detections.txt", 'w') as f:
        f.write(f"All YOLO detections for {img_name}\n")
        f.write(f"Total detections: {len(all_dets)}\n")
        f.write(f"Staff detections: {len(staff_boxes)}\n")
        f.write(f"Filtered: {detection_info['filtered_counts']}\n")
        f.write(f"{'Idx':<5} {'Class':<15} {'Conf':>6} {'x1':>6} {'y1':>6} {'x2':>6} {'y2':>6}\n")
        f.write("-" * 55 + "\n")
        for i, d in enumerate(sorted(all_dets, key=lambda x: x['y_center'])):
            x1, y1, x2, y2 = d['bbox']
            f.write(f"{i:<5} {d['class']:<15} {d['confidence']:>6.3f} {x1:>6} {y1:>6} {x2:>6} {y2:>6}\n")

    # Save each YOLO crop with confidence label drawn on it
    staff_dets = sorted(
        [d for d in all_dets if d['class'] == 'staff'],
        key=lambda x: x['y_center'],
    )
    virtual_set = set(detection_info.get('virtual_indices', []))
    for si, (y_center, (sx1, sy1, sx2, sy2)) in enumerate(staff_boxes):
        if si in virtual_set:
            # Virtual box — label it clearly
            x1, ct, x2, cb = crop_boundaries[si]
            crop = image.crop((x1, ct, x2, cb))
            labeled = crop.copy()
            draw = ImageDraw.Draw(labeled)
            label = f"VIRTUAL crop {si} | interpolated | bbox=({sx1},{sy1},{sx2},{sy2})"
            draw.rectangle([0, 0, labeled.width, 22], fill='darkblue')
            draw.text((4, 3), label, fill='cyan')
            labeled.save(out_dir / f"{img_name}_virtual_crop{si}.jpg")
        else:
            # Real detection
            real_idx = si - sum(1 for v in virtual_set if v < si)
            conf = staff_dets[real_idx]['confidence'] if real_idx < len(staff_dets) else 0.0
            x1, ct, x2, cb = crop_boundaries[si]
            crop = image.crop((x1, ct, x2, cb))

            # Draw confidence + label banner on the crop
            labeled = crop.copy()
            draw = ImageDraw.Draw(labeled)
            label = f"YOLO crop {si} | staff | conf={conf:.3f}"
            draw.rectangle([0, 0, labeled.width, 22], fill='black')
            draw.text((4, 3), label, fill='yellow')

            labeled.save(out_dir / f"{img_name}_yolo_crop{si}_conf{conf:.3f}.jpg")

    # Save matched crops
    for ci, gi, iou in match_info['matched']:
        x1, ct, x2, cb = crop_boundaries[ci]
        crop = image.crop((x1, ct, x2, cb))
        crop.save(out_dir / f"{img_name}_crop{ci}_gt{gi}_iou{iou:.2f}.jpg")
        with open(out_dir / f"{img_name}_crop{ci}_gt{gi}.txt", 'w') as f:
            f.write(gt_systems[gi])

    # Save unmatched crops (extra YOLO detections)
    for ci in match_info['unmatched_crops']:
        x1, ct, x2, cb = crop_boundaries[ci]
        crop = image.crop((x1, ct, x2, cb))
        if ci in virtual_set:
            crop.save(out_dir / f"{img_name}_EXTRA_virtual_crop{ci}.jpg")
        else:
            real_idx = ci - sum(1 for v in virtual_set if v < ci)
            conf = staff_dets[real_idx]['confidence'] if real_idx < len(staff_dets) else 0.0
            crop.save(out_dir / f"{img_name}_EXTRA_crop{ci}_conf{conf:.3f}.jpg")
        with open(out_dir / f"{img_name}_EXTRA_crop{ci}_info.txt", 'w') as f:
            is_virt = ci in virtual_set
            f.write(f"Extra {'virtual' if is_virt else 'YOLO'} crop {ci} with no matching GT system\n")
            if not is_virt:
                real_idx = ci - sum(1 for v in virtual_set if v < ci)
                conf = staff_dets[real_idx]['confidence'] if real_idx < len(staff_dets) else 0.0
                f.write(f"Confidence: {conf:.3f}\n")
            f.write(f"Crop bbox: x1={x1}, top={ct}, x2={x2}, bottom={cb}\n")

    # Save missing GT systems (GT with no crop)
    for gi in match_info['unmatched_gt']:
        with open(out_dir / f"{img_name}_MISSING_gt{gi}.txt", 'w') as f:
            f.write(f"GT system {gi} has no matching YOLO crop\n\n")
            f.write(f"GT content:\n{gt_systems[gi]}\n")

    # ── Save raw YOLO bbox crops (no dynamic boundary) in separate folder ──
    # Only for real detections — virtual boxes have no raw YOLO crop
    raw_dir = out_dir / "raw_yolo_crops"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for si, (y_center, (sx1, sy1, sx2, sy2)) in enumerate(staff_boxes):
        if si in virtual_set:
            continue  # Skip virtual boxes (no actual YOLO detection)
        real_idx = si - sum(1 for v in virtual_set if v < si)
        conf = staff_dets[real_idx]['confidence'] if real_idx < len(staff_dets) else 0.0
        raw_crop = image.crop((sx1, sy1, sx2, sy2))

        # Draw label
        labeled = raw_crop.copy()
        draw = ImageDraw.Draw(labeled)
        label = f"RAW YOLO {si} | conf={conf:.3f} | bbox=({sx1},{sy1},{sx2},{sy2})"
        draw.rectangle([0, 0, labeled.width, 22], fill='black')
        draw.text((4, 3), label, fill='cyan')
        labeled.save(raw_dir / f"{img_name}_raw_yolo{si}_conf{conf:.3f}.jpg")

    # Also save raw crops for ALL detections (including non-staff)
    for i, d in enumerate(sorted(all_dets, key=lambda x: x['y_center'])):
        x1, y1, x2, y2 = d['bbox']
        raw_crop = image.crop((x1, y1, x2, y2))
        labeled = raw_crop.copy()
        draw = ImageDraw.Draw(labeled)
        label = f"det {i} | {d['class']} | conf={d['confidence']:.3f}"
        draw.rectangle([0, 0, labeled.width, 22], fill='black')
        draw.text((4, 3), label, fill='cyan')
        labeled.save(raw_dir / f"{img_name}_alldet{i}_{d['class']}_conf{d['confidence']:.3f}.jpg")


def process_image(img_path, gt_path, yolo_model):
    """Process one image: YOLO detect -> crop -> match with GT."""
    image = Image.open(img_path).convert("RGB")

    if DESKEW:
        image = deskew_image(image)

    staff_boxes, detection_info = run_yolo_detection(image, yolo_model, CONFIDENCE_THRESHOLD)

    crop_boundaries = compute_dynamic_boundaries(
        staff_boxes, image.height, TOP_RATIO, BOTTOM_RATIO
    )

    with open(gt_path, 'r') as f:
        full_page_gt = f.read()
    gt_systems, gt_system_indices = extract_gt_systems(full_page_gt)

    match_info = match_crops_to_gt(crop_boundaries, gt_systems, image.height)

    return {
        'image': image,
        'img_path': str(img_path),
        'gt_path': str(gt_path),
        'staff_boxes': staff_boxes,
        'crop_boundaries': crop_boundaries,
        'detection_info': detection_info,
        'gt_systems': gt_systems,
        'gt_system_indices': gt_system_indices,
        'match_info': match_info,
        'n_yolo': len(staff_boxes),
        'n_gt': len(gt_systems),
        'is_match': len(staff_boxes) == len(gt_systems),
    }


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "all_annotated").mkdir(exist_ok=True)
    (output_dir / "mismatches").mkdir(exist_ok=True)
    (output_dir / "mismatch_crops").mkdir(exist_ok=True)

    # Load ALL splits (train + val + test)
    test_pairs = []
    for split_file in SPLIT_FILES:
        with open(split_file, 'r') as f:
            pairs = [line.strip().split() for line in f if line.strip()]
        split_name = Path(split_file).stem
        print(f"  Loaded {split_name}: {len(pairs)} samples")
        test_pairs.extend(pairs)

    # Deduplicate (in case splits overlap)
    seen = set()
    unique_pairs = []
    for ip, gp in test_pairs:
        if ip not in seen:
            seen.add(ip)
            unique_pairs.append((ip, gp))
    test_pairs = unique_pairs

    # Resolve relative paths
    test_pairs = [
        (str(Path(DATA_ROOT) / ip), str(Path(DATA_ROOT) / gp))
        for ip, gp in test_pairs
    ]

    # ── Focus on a single image (set to None to process all) ──
    SINGLE_IMAGE = None
    if SINGLE_IMAGE:
        test_pairs = [p for p in test_pairs if SINGLE_IMAGE in p[0]]

    print(f"Total samples: {len(test_pairs)}  (from {len(SPLIT_FILES)} splits)")
    print(f"YOLO model   : {YOLO_MODEL_PATH}")
    print(f"Confidence   : {CONFIDENCE_THRESHOLD}")
    print(f"Deskew       : {DESKEW}")
    print(f"Output dir   : {output_dir}")
    print("=" * 70)

    yolo_model = YOLO(YOLO_MODEL_PATH)

    results = []
    mismatches = []

    for img_path, gt_path in test_pairs:
        img_name = Path(img_path).stem
        print(f"\n{img_name}: ", end="", flush=True)

        try:
            r = process_image(img_path, gt_path, yolo_model)
            results.append(r)

            mi = r['match_info']
            status = "OK" if r['is_match'] else f"MISMATCH (YOLO={r['n_yolo']}, GT={r['n_gt']})"
            n_virt = r['detection_info'].get('n_virtual', 0)
            virt_str = f"  virtual={n_virt}" if n_virt > 0 else ""
            extra = f"  unmatched_crops={mi['unmatched_crops']}" if mi['unmatched_crops'] else ""
            missing = f"  missing_gt={mi['unmatched_gt']}" if mi['unmatched_gt'] else ""
            print(f"{status}{virt_str}{extra}{missing}")

            if r['detection_info']['filtered_counts']:
                filt = ", ".join(f"{k}={v}" for k, v in r['detection_info']['filtered_counts'].items())
                print(f"  Filtered: {filt}")

            if not r['is_match']:
                mismatches.append(r)

            # ── Save annotated image for EVERY sample ──
            annotated = draw_debug_page(
                r['image'],
                r['staff_boxes'],
                r['crop_boundaries'],
                r['match_info'],
                r['n_gt'],
                r['gt_system_indices'],
                r['detection_info'],
                title=img_name,
            )
            annotated.save(output_dir / "all_annotated" / f"{img_name}.jpg")

            # ── Extra saves for mismatches ──
            if not r['is_match']:
                annotated.save(output_dir / "mismatches" / f"{img_name}.jpg")

            # ── Always save all crop details when focusing on single image ──
            crop_dir = output_dir / "mismatch_crops" / img_name
            crop_dir.mkdir(parents=True, exist_ok=True)
            save_crop_details(
                r['image'],
                r['crop_boundaries'],
                r['gt_systems'],
                r['match_info'],
                r['detection_info'],
                r['staff_boxes'],
                crop_dir,
                img_name,
            )

        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROP DEBUG SUMMARY")
    print("=" * 70)

    total = len(results)
    matched = sum(1 for r in results if r['is_match'])
    print(f"\nTotal samples        : {total}")
    print(f"Matched (YOLO == GT) : {matched}/{total}")
    print(f"Mismatched           : {total - matched}/{total}")

    total_yolo = sum(r['n_yolo'] for r in results)
    total_gt   = sum(r['n_gt']   for r in results)
    total_virtual = sum(r['detection_info'].get('n_virtual', 0) for r in results)
    total_matched_systems = sum(len(r['match_info']['matched']) for r in results)
    total_extra_crops = sum(len(r['match_info']['unmatched_crops']) for r in results)
    total_missing_gt  = sum(len(r['match_info']['unmatched_gt']) for r in results)

    # System-level detection rate: how many GT systems got a crop
    system_detect_rate = total_matched_systems / total_gt * 100 if total_gt > 0 else 0.0
    # Page-level perfect match rate
    page_match_rate = matched / total * 100 if total > 0 else 0.0

    print(f"\nTotal YOLO crops     : {total_yolo} (real: {total_yolo - total_virtual}, virtual: {total_virtual})")
    print(f"Total GT systems     : {total_gt}")
    print(f"Difference           : {total_yolo - total_gt}")
    print(f"\n── Detection Metrics ──")
    print(f"Matched systems      : {total_matched_systems}/{total_gt}  ({system_detect_rate:.1f}%)")
    print(f"Missing GT (no crop) : {total_missing_gt}/{total_gt}  ({total_missing_gt/total_gt*100:.1f}%)" if total_gt else "")
    print(f"Extra crops (no GT)  : {total_extra_crops}/{total_yolo}  ({total_extra_crops/total_yolo*100:.1f}%)" if total_yolo else "")
    print(f"Page-level match     : {matched}/{total}  ({page_match_rate:.1f}%)")

    # Mismatch table
    if mismatches:
        print(f"\n{'Image':<25} {'YOLO':>5} {'GT':>5} {'Diff':>6} {'Extra crops':<18} {'Missing GT'}")
        print(f"{'-'*25} {'-'*5} {'-'*5} {'-'*6} {'-'*18} {'-'*18}")
        for r in sorted(mismatches, key=lambda x: abs(x['n_yolo'] - x['n_gt']), reverse=True):
            name = Path(r['img_path']).stem
            diff = r['n_yolo'] - r['n_gt']
            diff_s = f"+{diff}" if diff > 0 else str(diff)
            extras = str(r['match_info']['unmatched_crops']) if r['match_info']['unmatched_crops'] else "-"
            miss   = str(r['match_info']['unmatched_gt'])    if r['match_info']['unmatched_gt']    else "-"
            print(f"{name:<25} {r['n_yolo']:>5} {r['n_gt']:>5} {diff_s:>6} {extras:<18} {miss}")

            # Show all YOLO detections for this mismatch
            all_dets = sorted(r['detection_info']['all_detections'], key=lambda x: x['y_center'])
            for i, d in enumerate(all_dets):
                x1, y1, x2, y2 = d['bbox']
                print(f"    det {i}: {d['class']:<12} conf={d['confidence']:.3f}  bbox=({x1},{y1},{x2},{y2})")

    # Full breakdown
    print(f"\n{'Image':<25} {'YOLO':>5} {'GT':>5} {'Status':<10} {'Filtered'}")
    print(f"{'-'*25} {'-'*5} {'-'*5} {'-'*10} {'-'*30}")
    for r in results:
        name = Path(r['img_path']).stem
        status = "OK" if r['is_match'] else "MISMATCH"
        filt = ", ".join(f"{k}={v}" for k, v in r['detection_info']['filtered_counts'].items()) or "-"
        print(f"{name:<25} {r['n_yolo']:>5} {r['n_gt']:>5} {status:<10} {filt}")

    # Save JSON summary
    summary = {
        'total_samples': total,
        'matched': matched,
        'mismatched': total - matched,
        'total_yolo_crops': total_yolo,
        'total_gt_systems': total_gt,
        'total_virtual_boxes': total_virtual,
        'metrics': {
            'matched_systems': total_matched_systems,
            'missing_gt': total_missing_gt,
            'extra_crops': total_extra_crops,
            'system_detect_rate_pct': round(system_detect_rate, 2),
            'page_match_rate_pct': round(page_match_rate, 2),
        },
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'deskew': DESKEW,
        'mismatch_details': [
            {
                'image': Path(r['img_path']).name,
                'n_yolo': r['n_yolo'],
                'n_gt': r['n_gt'],
                'diff': r['n_yolo'] - r['n_gt'],
                'unmatched_crops': r['match_info']['unmatched_crops'],
                'unmatched_gt': r['match_info']['unmatched_gt'],
                'matched_pairs': [
                    {'crop': ci, 'gt': gi, 'iou': round(iou, 3)}
                    for ci, gi, iou in r['match_info']['matched']
                ],
            }
            for r in mismatches
        ],
        'per_image': [
            {
                'image': Path(r['img_path']).name,
                'n_yolo': r['n_yolo'],
                'n_gt': r['n_gt'],
                'is_match': r['is_match'],                'n_virtual': r['detection_info'].get('n_virtual', 0),
                'interpolation': r['detection_info'].get('interpolation', {}),                'gt_indices': r['gt_system_indices'],
                'filtered': r['detection_info']['filtered_counts'],
                'matched_pairs': [
                    {'crop': ci, 'gt': gi, 'iou': round(iou, 3)}
                    for ci, gi, iou in r['match_info']['matched']
                ],
                'unmatched_crops': r['match_info']['unmatched_crops'],
                'unmatched_gt': r['match_info']['unmatched_gt'],
            }
            for r in results
        ],
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON -> {output_dir / 'summary.json'}")
    print(f"Annotated pages -> {output_dir / 'all_annotated'}/")
    print(f"Mismatch pages  -> {output_dir / 'mismatches'}/")
    print(f"Mismatch crops  -> {output_dir / 'mismatch_crops'}/")


if __name__ == "__main__":
    main()
