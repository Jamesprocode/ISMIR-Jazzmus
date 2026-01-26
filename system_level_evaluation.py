"""
System-Level Evaluation for Jazz Lead Sheet Recognition

Evaluates recognition performance system-by-system (after YOLO cropping)
to identify outliers and isolate YOLO detection vs recognition errors.
"""

import numpy as np
from typing import List, Tuple, Dict
from PIL import Image
import torch
from tqdm import tqdm
import cv2
from pathlib import Path

from full_page_baseline import segment_staves
from inference import FullPageInference
from jazzmus.dataset.eval_functions import compute_poliphony_metrics


def extract_gt_systems(full_page_gt: str) -> List[str]:
    """
    Extract system-level ground truth from full-page **kern by splitting on linebreak markers.

    Args:
        full_page_gt: Full-page **kern string with !!linebreak:original markers

    Returns:
        List of **kern strings, one per system
    """
    lines = full_page_gt.strip().split('\n')

    systems = []
    current_system_lines = []
    header_lines = []
    in_header = True

    for line in lines:
        # Collect header lines (before first content)
        if in_header and (line.startswith('**') or line.startswith('*') or line.startswith('!')):
            if not line.startswith('*-'):  # Don't include spine terminators in header
                header_lines.append(line)
            continue
        else:
            in_header = False

        # Check for linebreak marker
        if line.strip() == '!!linebreak:original':
            if current_system_lines:
                # Build complete system with headers
                system = '\n'.join(header_lines + current_system_lines + ['*-\t*-'])
                systems.append(system)
                current_system_lines = []
        elif line.strip() and not line.startswith('*-'):
            # Add content line (skip spine terminators and empty lines)
            current_system_lines.append(line)

    # Add final system if any content remains
    if current_system_lines:
        system = '\n'.join(header_lines + current_system_lines + ['*-\t*-'])
        systems.append(system)

    return systems


def evaluate_systems(
    yolo_crops: List[Image.Image],
    gt_systems: List[str],
    inference_model: FullPageInference,
    image_path: str
) -> Dict:
    """
    Evaluate recognition on each system and compute per-system metrics.

    Args:
        yolo_crops: List of cropped system images from YOLO
        gt_systems: List of ground truth **kern strings (one per system)
        inference_model: Loaded FullPageInference model
        image_path: Path to original image (for logging)

    Returns:
        Dictionary with per-system results and diagnostics
    """
    n_crops = len(yolo_crops)
    n_gt = len(gt_systems)

    result = {
        'image': image_path,
        'n_yolo_crops': n_crops,
        'n_gt_systems': n_gt,
        'count_mismatch': n_crops != n_gt,
        'system_metrics': [],
        'crops': yolo_crops  # Store crops for later visualization
    }

    # If counts don't match, flag as YOLO detection error
    if n_crops != n_gt:
        result['warning'] = f"YOLO detected {n_crops} systems but GT has {n_gt} systems"
        return result

    # Evaluate each system
    for i, (crop, gt) in enumerate(zip(yolo_crops, gt_systems)):
        # Convert crop to numpy array for inference
        system_array = np.array(crop.convert('L'))

        # Run recognition
        pred_result = inference_model.predict(system_array)
        prediction = pred_result['prediction']

        # Compute metrics for this system
        cer, ser, ler = compute_poliphony_metrics([prediction], [gt])

        result['system_metrics'].append({
            'system_idx': i,
            'cer': cer,
            'ser': ser,
            'ler': ler,
            'prediction': prediction,
            'ground_truth': gt,
            'crop': crop  # Store crop image for this system
        })

    return result


def save_yolo_visualization(
    image_path: str,
    yolo_model_path: str,
    output_path: str,
    confidence_threshold: float = 0.5
):
    """
    Save visualization of YOLO bounding boxes on the original image.

    Args:
        image_path: Path to input image
        yolo_model_path: Path to YOLO model weights
        output_path: Where to save visualization
        confidence_threshold: Detection confidence threshold
    """
    from ultralytics import YOLO

    model = YOLO(yolo_model_path)
    results = model(image_path, conf=confidence_threshold, verbose=False)

    # Plot and save
    result_img = results[0].plot()
    cv2.imwrite(output_path, result_img)


def run_system_level_evaluation(
    test_split_file: str,
    yolo_model_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    save_visualizations: bool = False,
    viz_output_dir: str = "./yolo_viz"
):
    """
    Run system-level evaluation on test set.

    Args:
        test_split_file: Path to test split file (image_path gt_path pairs)
        yolo_model_path: Path to YOLO model weights
        checkpoint_path: Path to recognition model checkpoint
        device: Device for inference
        save_visualizations: Whether to save YOLO bounding box visualizations
        viz_output_dir: Directory for visualizations
    """
    # Load test split
    with open(test_split_file, 'r') as f:
        test_pairs = [line.strip().split() for line in f.readlines()]

    print(f"Loaded {len(test_pairs)} test samples")

    # Load recognition model
    print("Loading recognition model...")
    inference_model = FullPageInference(checkpoint_path, device=device)
    print("✓ Model loaded\n")

    # Create visualization directory if needed
    if save_visualizations:
        Path(viz_output_dir).mkdir(parents=True, exist_ok=True)

    # Collect results
    all_results = []
    failed_samples = []

    print("="*60)
    print("SYSTEM-LEVEL EVALUATION")
    print("="*60)

    for img_path, gt_path in tqdm(test_pairs, desc="Evaluating"):
        try:
            # Step 1: Segment with YOLO
            yolo_crops = segment_staves(
                image_path=img_path,
                yolo_model_path=yolo_model_path,
                confidence_threshold=0.3
            )

            # Step 2: Extract GT systems
            with open(gt_path, 'r') as f:
                full_page_gt = f.read()
            gt_systems = extract_gt_systems(full_page_gt)

            # Step 3: Evaluate each system
            result = evaluate_systems(yolo_crops, gt_systems, inference_model, img_path)
            all_results.append(result)

            # Step 4: Optional visualization for count mismatches
            if save_visualizations and result.get('count_mismatch', False):
                img_name = Path(img_path).stem
                # Save YOLO bounding box visualization
                viz_path = f"{viz_output_dir}/mismatch/{img_name}_yolo.jpg"
                Path(f"{viz_output_dir}/mismatch").mkdir(parents=True, exist_ok=True)
                save_yolo_visualization(img_path, yolo_model_path, viz_path)
                # Save original full page image
                original_path = f"{viz_output_dir}/mismatch/{img_name}_original.jpg"
                Image.open(img_path).save(original_path)

        except Exception as e:
            print(f"\n✗ Failed on {img_path}: {e}")
            failed_samples.append({'image': img_path, 'error': str(e)})
            continue

    # Analyze results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    # Count mismatches (YOLO detection errors)
    count_mismatches = [r for r in all_results if r.get('count_mismatch', False)]
    print(f"\nYOLO Detection Issues: {len(count_mismatches)}/{len(all_results)} samples")

    if count_mismatches:
        print("\nSamples with count mismatch:")
        for r in count_mismatches[:5]:  # Show first 5
            print(f"  {r['image']}: YOLO={r['n_yolo_crops']}, GT={r['n_gt_systems']}")
        if len(count_mismatches) > 5:
            print(f"  ... and {len(count_mismatches) - 5} more")

    # Aggregate per-system metrics (only for samples with matching counts)
    valid_results = [r for r in all_results if not r.get('count_mismatch', False)]
    all_system_metrics = []

    for result in valid_results:
        all_system_metrics.extend(result['system_metrics'])

    if all_system_metrics:
        print(f"\n{len(all_system_metrics)} total systems evaluated")

        cers = [m['cer'] for m in all_system_metrics]
        sers = [m['ser'] for m in all_system_metrics]
        lers = [m['ler'] for m in all_system_metrics]

        print(f"\nPer-System Metrics (mean ± std):")
        print(f"  CER: {np.mean(cers):.2f}% ± {np.std(cers):.2f}%")
        print(f"  SER: {np.mean(sers):.2f}% ± {np.std(sers):.2f}%")
        print(f"  LER: {np.mean(lers):.2f}% ± {np.std(lers):.2f}%")

        # Find outliers (systems with high error rates)
        cer_threshold = np.mean(cers) + 2 * np.std(cers)
        outliers = [m for m in all_system_metrics if m['cer'] > cer_threshold]

        print(f"\nOutliers (CER > {cer_threshold:.2f}%): {len(outliers)} systems")
        if outliers:
            # Sort by CER descending
            outliers.sort(key=lambda x: x['cer'], reverse=True)
            print("\nTop 10 worst systems:")
            for i, m in enumerate(outliers[:10]):
                # Find which image this system belongs to
                parent_result = next(r for r in valid_results if m in r['system_metrics'])
                img_name = Path(parent_result['image']).stem
                print(f"  {i+1}. {img_name} - System {m['system_idx']}: CER={m['cer']:.2f}%, SER={m['ser']:.2f}%")

            # Save top 10 worst cropped system images
            if save_visualizations:
                worst_crops_dir = f"{viz_output_dir}/worst_crops"
                Path(worst_crops_dir).mkdir(parents=True, exist_ok=True)
                print(f"\nSaving top 10 worst cropped systems to {worst_crops_dir}/")
                for i, m in enumerate(outliers[:10]):
                    parent_result = next(r for r in valid_results if m in r['system_metrics'])
                    img_name = Path(parent_result['image']).stem
                    crop_img = m['crop']
                    crop_filename = f"{worst_crops_dir}/{i+1:02d}_{img_name}_sys{m['system_idx']}_CER{m['cer']:.1f}.jpg"
                    crop_img.save(crop_filename)
                    print(f"  Saved: {crop_filename}")

    print(f"\nFailed samples: {len(failed_samples)}")
    print("="*60)

    return all_results, failed_samples


if __name__ == "__main__":
    # Configuration (adjust paths as needed)
    checkpoint_path = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/weights/smt_sys_best/smt_pre_syn_medium.ckpt"
    yolo_model_path = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/yolo_weigths/yolov11s_20241108.pt"
    test_split_file = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/data/jazzmus_fullpage/splits/test_0.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run evaluation
    results, failures = run_system_level_evaluation(
        test_split_file=test_split_file,
        yolo_model_path=yolo_model_path,
        checkpoint_path=checkpoint_path,
        device=device,
        save_visualizations=True,  # Save YOLO visualizations for mismatched samples
        viz_output_dir="./yolo_detection_viz"
    )
