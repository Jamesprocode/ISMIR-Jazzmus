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
from inference import FullPageInference, extract_spines, filter_chord_spine
from jazzmus.dataset.eval_functions import compute_poliphony_metrics
from chord_metrics import (
    compute_all_chord_metrics,
    print_chord_metrics,
    extract_chords_from_mxhm,
    compute_page_chord_metrics,
    aggregate_page_chord_metrics,
    print_page_chord_metrics,  # used for system-level ED aggregation
)


def extract_gt_systems(full_page_gt: str) -> Tuple[List[str], List[int]]:
    """
    Extract system-level ground truth from full-page **kern by splitting on linebreak markers.

    Systems containing any line starting with !LO are excluded (Solo, Coda, etc.)
    to match training data preprocessing.

    Args:
        full_page_gt: Full-page **kern string with !!linebreak:original markers

    Returns:
        Tuple of:
            - List of **kern strings, one per system (excluding filtered sections)
            - List of original system indices (for debugging which were kept)
    """
    lines = full_page_gt.strip().split('\n')

    systems = []
    system_indices = []  # Track which original indices were kept
    current_system_lines = []
    header_lines = []
    in_header = True
    current_system_idx = 0

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
                # Strip !LO layout directives (tempo, text, coda markers, etc.)
                # from system content — they are not musical content
                cleaned = [l for l in current_system_lines if not l.startswith('!')]
                if cleaned:
                    system = '\n'.join(header_lines + cleaned + ['*-\t*-'])
                    systems.append(system)
                    system_indices.append(current_system_idx)

                current_system_lines = []
                current_system_idx += 1
        elif line.strip() and not line.startswith('*-'):
            # Add content line (skip spine terminators and empty lines)
            current_system_lines.append(line)

    # Add final system if any content remains
    if current_system_lines:
        cleaned = [l for l in current_system_lines if not l.startswith('!')]
        if cleaned:
            system = '\n'.join(header_lines + cleaned + ['*-\t*-'])
            systems.append(system)
            system_indices.append(current_system_idx)

    return systems, system_indices


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

    # Flag count mismatch but still evaluate what we can
    if n_crops != n_gt:
        if n_crops > n_gt:
            result['warning'] = f"YOLO detected {n_crops} systems but GT has {n_gt} (evaluating {n_gt}, ignoring {n_crops - n_gt} extra crops)"
        else:
            result['warning'] = f"YOLO detected {n_crops} systems but GT has {n_gt} (evaluating {n_crops}, YOLO missed {n_gt - n_crops} systems)"
        print(f"  {result['warning']}")

    # Evaluate based on GT count (top-to-bottom matching)
    # If YOLO has more: use first n_gt crops
    # If YOLO has fewer: evaluate what YOLO detected
    n_eval = min(n_crops, n_gt)
    for i in range(n_eval):
        crop = yolo_crops[i]
        gt = gt_systems[i]
        # Convert crop to numpy array for inference
        system_array = np.array(crop.convert('L'))

        # Run recognition
        pred_result = inference_model.predict(system_array)
        prediction = pred_result['prediction']

        # Compute standard metrics for this system
        cer, ser, ler = compute_poliphony_metrics([prediction], [gt])

        system_result = {
            'system_idx': i,
            'cer': cer,
            'ser': ser,
            'ler': ler,
            'prediction': prediction,
            'ground_truth': gt,
            'crop': crop  # Store crop image for this system
        }

        # Compute chord-specific metrics if **mxhm spine exists
        try:
            pred_spines = extract_spines(prediction)
            gt_spines = extract_spines(gt)
            if '**mxhm' in pred_spines and '**mxhm' in gt_spines:
                chord_metrics = compute_all_chord_metrics(
                    pred_spines['**mxhm'],
                    gt_spines['**mxhm']
                )
                system_result['chord_metrics'] = chord_metrics
                # Edit distance metrics for this system
                pred_chords = extract_chords_from_mxhm(pred_spines['**mxhm'])
                gt_chords = extract_chords_from_mxhm(gt_spines['**mxhm'])
                if pred_chords and gt_chords:
                    system_result['ed_chord_metrics'] = compute_page_chord_metrics(pred_chords, gt_chords)
        except Exception as e:
            system_result['chord_metrics_error'] = str(e)

        result['system_metrics'].append(system_result)

    return result


def save_yolo_visualization(
    image_path: str,
    yolo_model_path: str,
    output_path: str,
    confidence_threshold: float = 0.6
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
            # Step 1: Segment with YOLO (with deskew for rotated images)
            yolo_crops = segment_staves(
                image_path=img_path,
                yolo_model_path=yolo_model_path,
                confidence_threshold=0.2,
                deskew=True
            )

            # Step 2: Extract GT systems (excluding Solo/Coda sections)
            with open(gt_path, 'r') as f:
                full_page_gt = f.read()
            gt_systems, gt_system_indices = extract_gt_systems(full_page_gt)

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

    # Aggregate per-system metrics (include all samples, even with count mismatches)
    valid_results = all_results  # Now we evaluate partial matches too
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

        # Per-page metrics (average across systems within each page)
        page_cers = []
        page_sers = []
        page_lers = []
        for result in valid_results:
            if result['system_metrics']:
                page_cer = np.mean([m['cer'] for m in result['system_metrics']])
                page_ser = np.mean([m['ser'] for m in result['system_metrics']])
                page_ler = np.mean([m['ler'] for m in result['system_metrics']])
                page_cers.append(page_cer)
                page_sers.append(page_ser)
                page_lers.append(page_ler)

        if page_cers:
            print(f"\nPer-Page Metrics (mean ± std across {len(page_cers)} pages):")
            print(f"  CER: {np.mean(page_cers):.2f}% ± {np.std(page_cers):.2f}%")
            print(f"  SER: {np.mean(page_sers):.2f}% ± {np.std(page_sers):.2f}%")
            print(f"  LER: {np.mean(page_lers):.2f}% ± {np.std(page_lers):.2f}%")

        # Aggregate chord-specific metrics
        chord_systems = [m for m in all_system_metrics if 'chord_metrics' in m]
        if chord_systems:
            print(f"\n--- CHORD-SPECIFIC METRICS ({len(chord_systems)} systems) ---")

            # Aggregate root F1 - Position-based (strict)
            root_correct = sum(m['chord_metrics']['root_f1']['correct'] for m in chord_systems)
            root_pred = sum(m['chord_metrics']['root_f1']['pred_count'] for m in chord_systems)
            root_gt = sum(m['chord_metrics']['root_f1']['gt_count'] for m in chord_systems)
            root_precision = root_correct / root_pred * 100 if root_pred > 0 else 0
            root_recall = root_correct / root_gt * 100 if root_gt > 0 else 0
            root_f1 = 2 * root_precision * root_recall / (root_precision + root_recall) if (root_precision + root_recall) > 0 else 0

            # Aggregate root F1 - LCS (handles insertions/deletions)
            lcs_correct = sum(m['chord_metrics']['root_f1']['align_correct'] for m in chord_systems)
            lcs_precision = lcs_correct / root_pred * 100 if root_pred > 0 else 0
            lcs_recall = lcs_correct / root_gt * 100 if root_gt > 0 else 0
            lcs_f1 = 2 * lcs_precision * lcs_recall / (lcs_precision + lcs_recall) if (lcs_precision + lcs_recall) > 0 else 0

            # Aggregate root F1 - Windowed (±3 tolerance for line alignment errors)
            window_correct = sum(m['chord_metrics']['root_f1']['window_correct'] for m in chord_systems)
            window_precision = window_correct / root_pred * 100 if root_pred > 0 else 0
            window_recall = window_correct / root_gt * 100 if root_gt > 0 else 0
            window_f1 = 2 * window_precision * window_recall / (window_precision + window_recall) if (window_precision + window_recall) > 0 else 0

            print(f"\nRoot Detection (aggregate):")
            print(f"  ┌───────────────────┬───────────┬──────────┬──────────┐")
            print(f"  │ Strategy          │ Precision │  Recall  │    F1    │")
            print(f"  ├───────────────────┼───────────┼──────────┼──────────┤")
            print(f"  │ Position-based    │  {root_precision:6.2f}%  │  {root_recall:6.2f}% │  {root_f1:6.2f}% │")
            print(f"  │ Aligned (LCS)     │  {lcs_precision:6.2f}%  │  {lcs_recall:6.2f}% │  {lcs_f1:6.2f}% │")
            print(f"  │ Windowed (±3)     │  {window_precision:6.2f}%  │  {window_recall:6.2f}% │  {window_f1:6.2f}% │")
            print(f"  └───────────────────┴───────────┴──────────┴──────────┘")
            print(f"  Counts: {root_pred} predicted, {root_gt} GT chords")
            print(f"\n  Interpretation:")
            print(f"    - If LCS >> Position: insertion/deletion errors dominate")
            print(f"    - If Windowed >> Position: line alignment shift errors dominate")
            print(f"    - If all similar: true root detection errors")

            # Aggregate quality accuracy
            qual_correct = sum(m['chord_metrics']['quality']['correct'] for m in chord_systems)
            qual_total = sum(m['chord_metrics']['quality']['total_root_matches'] for m in chord_systems)
            qual_acc = qual_correct / qual_total * 100 if qual_total > 0 else 0

            print(f"\nQuality Accuracy (where roots match):")
            print(f"  Accuracy: {qual_acc:.2f}% ({qual_correct}/{qual_total})")

            # Aggregate extension accuracy
            ext_correct = sum(m['chord_metrics']['extension']['correct'] for m in chord_systems)
            ext_total = sum(m['chord_metrics']['extension']['total_root_matches'] for m in chord_systems)
            ext_acc = ext_correct / ext_total * 100 if ext_total > 0 else 0

            print(f"\nExtension Accuracy (where roots match):")
            print(f"  Accuracy: {ext_acc:.2f}% ({ext_correct}/{ext_total})")

            # Aggregate full chord P/R/F1
            full_correct = sum(m['chord_metrics']['full_chord']['correct'] for m in chord_systems)
            full_pred = sum(m['chord_metrics']['full_chord']['pred_count'] for m in chord_systems)
            full_gt = sum(m['chord_metrics']['full_chord']['gt_count'] for m in chord_systems)

            full_precision = full_correct / full_pred * 100 if full_pred > 0 else 0
            full_recall = full_correct / full_gt * 100 if full_gt > 0 else 0
            full_f1 = 2 * full_precision * full_recall / (full_precision + full_recall) if (full_precision + full_recall) > 0 else 0

            print(f"\nFull Chord Match (root + quality + extension):")
            print(f"  Precision: {full_precision:.2f}% ({full_correct}/{full_pred} predicted)")
            print(f"  Recall:    {full_recall:.2f}% ({full_correct}/{full_gt} GT)")
            print(f"  F1:        {full_f1:.2f}%")

            # Count alignment mismatches
            count_mismatches_chord = sum(
                1 for m in chord_systems
                if not m['chord_metrics']['alignment']['counts_match']
            )
            print(f"\nChord count mismatches: {count_mismatches_chord}/{len(chord_systems)} systems")

            # Find systems with worst chord count mismatches
            systems_with_mismatch = [
                m for m in chord_systems
                if not m['chord_metrics']['alignment']['counts_match']
            ]
            # Sort by absolute count difference (biggest mismatches first)
            systems_with_mismatch.sort(
                key=lambda x: abs(x['chord_metrics']['alignment']['count_diff']),
                reverse=True
            )

            if systems_with_mismatch:
                print(f"\nTop 10 worst chord count mismatches:")
                for i, m in enumerate(systems_with_mismatch[:10]):
                    parent_result = next(r for r in valid_results if m in r['system_metrics'])
                    img_name = Path(parent_result['image']).stem
                    diff = m['chord_metrics']['alignment']['count_diff']
                    pred_cnt = m['chord_metrics']['alignment']['pred_count']
                    gt_cnt = m['chord_metrics']['alignment']['gt_count']
                    diff_str = f"+{diff}" if diff > 0 else str(diff)
                    error_type = "insertions" if diff > 0 else "deletions"
                    print(f"  {i+1}. {img_name} - System {m['system_idx']}: "
                          f"pred={pred_cnt}, gt={gt_cnt} ({diff_str} {error_type})")

                # Save top 10 chord mismatch crops
                if save_visualizations:
                    chord_mismatch_dir = f"{viz_output_dir}/chord_mismatch_crops"
                    Path(chord_mismatch_dir).mkdir(parents=True, exist_ok=True)
                    print(f"\nSaving top 10 chord mismatch systems to {chord_mismatch_dir}/")
                    for i, m in enumerate(systems_with_mismatch[:10]):
                        parent_result = next(r for r in valid_results if m in r['system_metrics'])
                        img_name = Path(parent_result['image']).stem
                        diff = m['chord_metrics']['alignment']['count_diff']
                        diff_str = f"+{diff}" if diff > 0 else str(diff)
                        base_filename = f"{i+1:02d}_{img_name}_sys{m['system_idx']}_diff{diff_str}"

                        # Save cropped image
                        crop_img = m['crop']
                        crop_img.save(f"{chord_mismatch_dir}/{base_filename}.jpg")

                        # Save predicted and GT kern
                        with open(f"{chord_mismatch_dir}/{base_filename}_pred.txt", 'w') as f:
                            f.write(m['prediction'])
                        with open(f"{chord_mismatch_dir}/{base_filename}_gt.txt", 'w') as f:
                            f.write(m['ground_truth'])

                        print(f"  Saved: {base_filename} (.jpg, _pred.txt, _gt.txt)")

        # System-level edit distance chord metrics
        system_ed_metrics = [m['ed_chord_metrics'] for m in all_system_metrics if 'ed_chord_metrics' in m]
        if system_ed_metrics:
            agg_system_ed = aggregate_page_chord_metrics(system_ed_metrics)
            print_page_chord_metrics(agg_system_ed, unit_label="system")

        # Find outliers (systems with high error rates)
        cer_threshold = np.mean(cers) + 2 * np.std(cers)
        outliers = [m for m in all_system_metrics if m['cer'] > cer_threshold]

        print(f"\n--- OUTLIER ANALYSIS ---")
        print(f"Outliers (CER > {cer_threshold:.2f}%): {len(outliers)} systems")
        if outliers:
            # Sort by CER descending
            outliers.sort(key=lambda x: x['cer'], reverse=True)
            print("\nTop 10 worst systems:")
            for i, m in enumerate(outliers[:10]):
                # Find which image this system belongs to
                parent_result = next(r for r in valid_results if m in r['system_metrics'])
                img_name = Path(parent_result['image']).stem
                print(f"  {i+1}. {img_name} - System {m['system_idx']}: CER={m['cer']:.2f}%, SER={m['ser']:.2f}%")

            # Save top 10 worst cropped system images with prediction and ground truth
            if save_visualizations:
                worst_crops_dir = f"{viz_output_dir}/worst_crops"
                Path(worst_crops_dir).mkdir(parents=True, exist_ok=True)
                print(f"\nSaving top 10 worst cropped systems to {worst_crops_dir}/")
                for i, m in enumerate(outliers[:10]):
                    parent_result = next(r for r in valid_results if m in r['system_metrics'])
                    img_name = Path(parent_result['image']).stem
                    base_filename = f"{i+1:02d}_{img_name}_sys{m['system_idx']}_CER{m['cer']:.1f}"

                    # Save cropped image
                    crop_img = m['crop']
                    crop_img.save(f"{worst_crops_dir}/{base_filename}.jpg")

                    # Save predicted kern
                    with open(f"{worst_crops_dir}/{base_filename}_pred.txt", 'w') as f:
                        f.write(m['prediction'])

                    # Save ground truth kern
                    with open(f"{worst_crops_dir}/{base_filename}_gt.txt", 'w') as f:
                        f.write(m['ground_truth'])

                    print(f"  Saved: {base_filename} (.jpg, _pred.txt, _gt.txt)")


    print(f"\nFailed samples: {len(failed_samples)}")
    print("="*60)

    return all_results, failed_samples


if __name__ == "__main__":
    # Configuration (adjust paths as needed)
    checkpoint_path = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/weights/smt/smt_0.ckpt"
    yolo_model_path = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/yolo_weigths/yolov11s_20241108.pt"
    test_split_file = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/data/jazzmus_fullpage/splits/test_0.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run evaluation
    results, failures = run_system_level_evaluation(
        test_split_file=test_split_file,
        yolo_model_path=yolo_model_path,
        checkpoint_path=checkpoint_path,
        device=device,
        save_visualizations=False,  # Save YOLO visualizations for mismatched samples
        viz_output_dir="./yolo_detection_viz"
    )