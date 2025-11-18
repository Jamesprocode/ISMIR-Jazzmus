"""
Inference pipeline for full-page jazz leadsheet recognition.

Steps:
1. Load trained model checkpoint
2. Process image (resize, normalize)
3. Generate predictions token-by-token
4. Decode to kern format
5. Display/save results
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from jazzmus.smt_trainer import SMT_Trainer
from jazzmus.dataset.tokenizer import untokenize, process_text
from jazzmus.metrics import compute_metrics
from jazzmus.dataset.eval_functions import compute_poliphony_metrics
from collections import defaultdict

def extract_spines(kern_text):
    """
    Extract individual spines from **kern format.

    Returns dict with spine name -> content mapping.
    E.g., {'**kern': '...melody...', '**mxhm': '...chords...'}
    """
    lines = kern_text.strip().split('\n')
    spines = {}
    spine_indices = {}

    # Find spine headers (lines starting with **)
    for i, line in enumerate(lines):
        if line.startswith('**'):
            parts = line.split('\t')
            for j, part in enumerate(parts):
                if part.startswith('**'):
                    spine_name = part
                    if spine_name not in spines:
                        spines[spine_name] = []
                        spine_indices[spine_name] = j

    # Extract content for each spine
    for line in lines:
        if line.startswith('*') or line.startswith('=') or line.startswith('!'):
            # Metadata/formatting line - include for all spines
            parts = line.split('\t')
            for spine_name, idx in spine_indices.items():
                if idx < len(parts):
                    spines[spine_name].append(parts[idx])
        else:
            # Data line
            parts = line.split('\t')
            for spine_name, idx in spine_indices.items():
                if idx < len(parts):
                    spines[spine_name].append(parts[idx])

    # Join lines back together
    result = {}
    for spine_name, content_list in spines.items():
        result[spine_name] = '\n'.join(content_list)

    return result

def process_ground_truth_from_file(gt_path, model, tokenizer_type="word"):
    """
    Process ground truth from file exactly like training does.

    Training pipeline:
    1. Load raw file
    2. Tokenize with process_text()
    3. Add <bos> and <eos>
    4. Convert to token IDs with w2i
    5. Convert back to strings with i2w
    6. Untokenize to get readable format

    Args:
        gt_path: Path to ground truth kern file
        model: Trained SMTModelForCausalLM with w2i and i2w mappings
        tokenizer_type: "word", "character", or "medium"

    Returns:
        Untokenized readable ground truth string
    """
    # Load raw file
    with open(gt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Tokenize (same as dataset preprocessing)
    tokens = process_text(lines, tokenizer_type=tokenizer_type)

    # Add special tokens (same as dataset preprocessing, line 257 in smt_dataset.py)
    tokens = ["<bos>"] + tokens + ["<eos>"]

    # Convert token strings to token IDs using w2i (same as __getitem__ line 248)
    token_ids = [model.w2i[token] for token in tokens]
    token_ids = torch.tensor(token_ids, dtype=torch.long)

    # Convert back to strings using i2w, excluding the last token (<eos>) like training does (line 194)
    # gt = untokenize([self.model.i2w[token.item()] for token in y_single[:-1]])
    gt_tokens = [model.i2w[token.item()] for token in token_ids[:-1]]

    # Untokenize to get readable format
    gt_readable = untokenize(gt_tokens)

    return gt_readable

def calculate_spine_metrics(prediction, ground_truth):
    """
    Calculate CER/SER/LER for individual spines and overall.

    Returns dict with metrics for each spine and overall.
    """
    # Extract spines
    pred_spines = extract_spines(prediction)
    gt_spines = extract_spines(ground_truth)

    # Get all spine names
    all_spines = set(pred_spines.keys()) | set(gt_spines.keys())

    results = {}

    # Calculate metrics for each spine
    for spine_name in sorted(all_spines):
        pred_spine = pred_spines.get(spine_name, "")
        gt_spine = gt_spines.get(spine_name, "")

        if not gt_spine:  # Skip if no ground truth for this spine
            continue

        try:
            cer, ser, ler = compute_poliphony_metrics([pred_spine], [gt_spine])
            results[spine_name] = {
                "cer": cer,
                "ser": ser,
                "ler": ler,
            }
        except Exception as e:
            results[spine_name] = {
                "cer": 100.0,
                "ser": 100.0,
                "ler": 100.0,
                "error": str(e),
            }

    # Calculate overall metrics
    try:
        cer_overall, ser_overall, ler_overall = compute_poliphony_metrics([prediction], [ground_truth])
        results["OVERALL"] = {
            "cer": cer_overall,
            "ser": ser_overall,
            "ler": ler_overall,
        }
    except Exception as e:
        results["OVERALL"] = {
            "cer": 100.0,
            "ser": 100.0,
            "ler": 100.0,
            "error": str(e),
        }

    return results


class FullPageInference:
    """Inference pipeline for full-page jazz leadsheet recognition."""

    def __init__(self, checkpoint_path, device):
        """
        Initialize inference pipeline.

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run on (cuda or cpu)
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # Load model
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create model with same config as checkpoint
        # Note: You'll need to load the config from somewhere
        # For now, we'll load it manually
        self.model = SMT_Trainer.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
        )
        self.model.eval()
        self.model.to(self.device)

        print("✓ Model loaded successfully")

    def preprocess_image(self, image_path, max_height=128, max_width=1000):
        """
        Preprocess image for inference.

        Args:
            image_path: Path to input image
            max_height: Maximum image height
            max_width: Maximum image width

        Returns:
            torch.Tensor: Preprocessed image
        """
        # Load image
        if isinstance(image_path, str):
            print("image loaded")
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = np.array(image_path)

        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Resize with aspect ratio preservation
        height, width = img.shape
        aspect_ratio = width / height

        # Fit to max dimensions
        if height > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_height = height
            new_width = width

        if new_width > max_width:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)

        # Resize
        img = cv2.resize(img, (new_width, new_height))

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

        # Pad to max size with ones (white padding, matching training)
        padded = torch.ones(1, 1, max_height, max_width)
        padded[:, :, :new_height, :new_width] = img_tensor[:, :, :new_height, :new_width]

        print(f"✓ Image preprocessed: {(height, width)}->{(new_height, new_width)} -> {padded.shape}")

        return padded.to(self.device)

    def predict(self, image_path, return_probs=False):
        """
        Predict on full-page image.

        Args:
            image_path: Path to input image
            return_probs: Return token probabilities

        Returns:
            dict: Prediction results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)

        # Generate predictions
        print("Generating predictions...")
        with torch.no_grad():
            predicted_tokens, logits = self.model.model.predict(input=image_tensor[0])

        # Decode tokens to string
        # predicted_tokens are already strings, no need to convert to int
        if isinstance(predicted_tokens[0], int):
            # If tokens are integers, look them up in i2w
            token_strs = [self.model.model.i2w.get(int(t), "<unk>") for t in predicted_tokens]
        else:
            # If tokens are already strings, use them directly
            token_strs = [str(t) for t in predicted_tokens]
        prediction_str = untokenize(token_strs)
        results = {
            "tokens": token_strs,
            "prediction": prediction_str,
            "num_tokens": len(predicted_tokens),
        }

        if return_probs:
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
            results["logits"] = logits.cpu().numpy()
            results["top_probs"] = top_probs.cpu().numpy()
            results["top_indices"] = top_indices.cpu().numpy()

        return results

    def save_result(self, result, output_path):
        """
        Save prediction result to file.

        Args:
            result: Prediction result dictionary
            output_path: Path to save result
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['prediction'])

        print(f"✓ Result saved to: {output_path}")

    def display_result(self, result):
        """
        Display prediction result.

        Args:
            result: Prediction result dictionary
        """
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"Number of tokens: {result['num_tokens']}")
        print(f"\nPredicted kern:")
        print(result['prediction'])
        print("=" * 60)

    def score_with_ground_truth(self, prediction, ground_truth_path, per_spine=True, tokenizer_type="medium"):
        """
        Calculate metrics by comparing prediction with ground truth.

        Args:
            prediction: Predicted **kern string
            ground_truth_path: Path to ground truth file
            per_spine: Whether to calculate metrics per individual spine

        Returns:
            dict: Metrics (SER, SEQ-ER, and per-spine if requested)
        """
        # Load ground truth
        ground_truth = process_ground_truth_from_file(ground_truth_path, self.model, tokenizer_type)

        print("\n" + "=" * 60)
        print("ground truth")
        print(ground_truth)
        # Display metrics
        print("\n" + "=" * 60)
        print("SCORING RESULTS")
        print("=" * 60)


        # Calculate per-spine metrics
        spine_metrics = calculate_spine_metrics(prediction, ground_truth)

        print("\nPER-SPINE METRICS:")
        print("-" * 60)
        for spine_name in sorted(spine_metrics.keys()):
            metrics = spine_metrics[spine_name]
            print(f"\n{spine_name}:")
            print(f"  CER: {metrics['cer']:.2f}%")
            print(f"  SER: {metrics['ser']:.2f}%")
            print(f"  LER: {metrics['ler']:.2f}%")

    # Calculate overall metrics only
        cer, ser, ler = compute_poliphony_metrics([prediction], [ground_truth])
        spine_metrics = {"OVERALL": {"cer": cer, "ser": ser, "ler": ler}}
        print(f"SER (System Error Rate):     {ser:.2f}%")
        print(f"CER (Character Error Rate): {cer:.2f}%")
        print(f"LER (Line Error Rate):      {ler:.2f}%")

        print(f"\nGround truth length: {len(ground_truth)}")
        print(f"Prediction length:   {len(prediction)}")
        print("=" * 60 + "\n")

        return spine_metrics


def evaluate_test_set(
    checkpoint_path,
    test_dir,
    split="test",
    fold=0,
    output_dir=None,
    device="cuda",
    tokenizer_type="medium",
):
    """
    Evaluate model on entire test set and compute aggregate metrics.

    Args:
        checkpoint_path: Path to trained model checkpoint
        test_dir: Path to dataset directory with splits
        split: Which split to evaluate (train/val/test)
        fold: Fold number
        output_dir: Optional directory to save predictions
        device: Device to run on
    """
    test_dir = Path(test_dir)
    splits_dir = test_dir / "splits"

    # Read split file
    split_file = splits_dir / f"{split}_{fold}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, 'r') as f:
        lines = f.readlines()
        img_paths = [line.split()[0] for line in lines]
        gt_paths = [line.split()[1] for line in lines]

    print("\n" + "=" * 70)
    print(f"EVALUATING ON {split.upper()} SET")
    print("=" * 70)
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Test images:    {len(img_paths)}")
    print(f"Dataset:        {test_dir}")
    print("=" * 70 + "\n")

    # Initialize inference
    print("Loading model...")
    inference = FullPageInference(checkpoint_path, device=device)
    print()

    # Collect all predictions and ground truths
    all_predictions = []
    all_ground_truths = []
    individual_metrics = []
    spine_predictions = defaultdict(list)  # spine -> list of predictions
    spine_ground_truths = defaultdict(list)  # spine -> list of ground truths
    failed = []

    print(f"Running inference on {len(img_paths)} images...\n")
    for idx, (img_path, gt_path) in enumerate(tqdm(zip(img_paths, gt_paths), total=len(img_paths), desc="Evaluating")):
        try:
            # Run inference
            result = inference.predict(img_path)
            prediction = result['prediction']

            # Load ground truth
            ground_truth = process_ground_truth_from_file(gt_path, inference.model, tokenizer_type)

            all_predictions.append(prediction)
            all_ground_truths.append(ground_truth)

            # Calculate per-sample metrics (overall)
            cer, ser, ler = compute_poliphony_metrics([prediction], [ground_truth])
            individual_metrics.append({
                "image": Path(img_path).name,
                "cer": cer,
                "ser": ser,
                "ler": ler,
            })

            # Extract and collect spines for per-spine aggregate metrics
            try:
                pred_spines = extract_spines(prediction)
                gt_spines = extract_spines(ground_truth)

                for spine_name in pred_spines.keys():
                    spine_predictions[spine_name].append(pred_spines[spine_name])
                    spine_ground_truths[spine_name].append(gt_spines.get(spine_name, ""))
            except Exception as e:
                # If spine extraction fails, skip per-spine metrics for this sample
                pass

            # Save prediction if output dir provided
            if output_dir:
                output_path = Path(output_dir) / (Path(img_path).stem + "_pred.txt")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(prediction)

        except Exception as e:
            failed.append((Path(img_path).name, str(e)))
            continue

    # Calculate aggregate metrics
    print("\n\nCalculating aggregate metrics...\n")
    cer_agg, ser_agg, ler_agg = compute_poliphony_metrics(all_predictions, all_ground_truths)

    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"✓ Successfully evaluated: {len(all_predictions)}/{len(img_paths)}")
    if failed:
        print(f"✗ Failed: {len(failed)}")
        for filename, reason in failed[:5]:
            print(f"   - {filename}: {reason}")
        if len(failed) > 5:
            print(f"   ... and {len(failed) - 5} more")

    print("\n" + "=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)

    print("\nOVERALL (All Spines Combined):")
    print("-" * 70)
    print(f"CER (Character Error Rate):  {cer_agg:.2f}%")
    print(f"SER (Sequence Error Rate):   {ser_agg:.2f}%")
    print(f"LER (Line Error Rate):       {ler_agg:.2f}%")

    # Show per-spine aggregate metrics using same calculation method
    if spine_predictions:
        print("\n" + "-" * 70)
        print("PER-SPINE METRICS:")
        print("-" * 70)

        for spine_name in sorted(spine_predictions.keys()):
            pred_list = spine_predictions[spine_name]
            gt_list = spine_ground_truths[spine_name]

            if not pred_list or not gt_list:
                continue

            try:
                cer_spine, ser_spine, ler_spine = compute_poliphony_metrics(pred_list, gt_list)
                print(f"\n{spine_name}:")
                print(f"  CER: {cer_spine:.2f}%")
                print(f"  SER: {ser_spine:.2f}%")
                print(f"  LER: {ler_spine:.2f}%")
            except Exception as e:
                print(f"\n{spine_name}:")
                print(f"  Error calculating metrics: {str(e)}")

    print("\n" + "=" * 70)

    # Show per-sample statistics
    if individual_metrics:
        print("\nPER-SAMPLE STATISTICS (OVERALL)")
        print("-" * 70)
        cer_values = [m["cer"] for m in individual_metrics]
        ser_values = [m["ser"] for m in individual_metrics]
        ler_values = [m["ler"] for m in individual_metrics]

        print(f"CER - Mean: {np.mean(cer_values):.2f}%, Std: {np.std(cer_values):.2f}%, Min: {np.min(cer_values):.2f}%, Max: {np.max(cer_values):.2f}%")
        print(f"SER - Mean: {np.mean(ser_values):.2f}%, Std: {np.std(ser_values):.2f}%, Min: {np.min(ser_values):.2f}%, Max: {np.max(ser_values):.2f}%")
        print(f"LER - Mean: {np.mean(ler_values):.2f}%, Std: {np.std(ler_values):.2f}%, Min: {np.min(ler_values):.2f}%, Max: {np.max(ler_values):.2f}%")
        print("=" * 70 + "\n")

    if output_dir:
        print(f"✓ Predictions saved to: {output_dir}\n")

    return {
        "cer": cer_agg,
        "ser": ser_agg,
        "ler": ler_agg,
        "individual_metrics": individual_metrics,
    }


def run_inference(
    checkpoint_path,
    image_path=None,
    output_path=None,
    ground_truth_path=None,
    test_dir=None,
    split="test",
    fold=0,
    device="cuda",
    tokenizer_type="medium",
):
    """
    Run inference on single image or entire test set.

    Args:
        checkpoint_path: Path to trained model checkpoint
        image_path: Path to input image (for single inference)
        output_path: Optional path to save prediction
        ground_truth_path: Optional path to ground truth for scoring
        test_dir: Path to test directory (for batch evaluation)
        split: Which split to evaluate (train/val/test)
        fold: Fold number
        device: Device to run on
    """
    # Batch evaluation mode
    if test_dir:
        evaluate_test_set(checkpoint_path, test_dir, split, fold, output_path, device, tokenizer_type)
        return

    # Single image mode
    if not image_path:
        raise ValueError("Either image_path or test_dir must be provided")

    # Initialize inference
    inference = FullPageInference(checkpoint_path, device=device)

    # Predict
    result = inference.predict(image_path)

    # Display result
    inference.display_result(result)
    # Score if ground truth provided
    if ground_truth_path:
        inference.score_with_ground_truth(result['prediction'], ground_truth_path, tokenizer_type)



if __name__ == "__main__":
    # ============ CONFIGURATION ============
    # Edit these parameters and run: python inference.py

    # Single image inference
    IMAGE_PATH = "data/jazzmus_systems/jpg/img_10_1.jpg"  # Path to image (e.g., "path/to/image.jpg")
    GROUND_TRUTH_PATH = "data/jazzmus_systems/gt/img_10_1.txt"  # Path to ground truth (e.g., "path/to/gt.txt")
    # IMAGE_PATH = None
    # GROUND_TRUTH_PATH = None
    # Batch evaluation
    TEST_DIR = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/data/jazzmus_systems"  # Dataset directory
    TEST_DIR = None
    SPLIT = "test"  # Which split: train/val/test
    FOLD = 0  # Fold number
    TOKENIZER_TYPE = "medium"  # Tokenizer type: "word", "character", or "medium"
    
    # Model
    CHECKPOINT_PATH = "weights/smt/smt_0-v1.ckpt"
    DEVICE = "cuda"  # cuda or cpu

    # Output (optional, leave None to skip saving)
    OUTPUT_DIR = None  # Directory to save predictions

    # ============ RUN INFERENCE ============
    run_inference(
        checkpoint_path=CHECKPOINT_PATH,
        image_path=IMAGE_PATH,
        output_path=OUTPUT_DIR,
        ground_truth_path=GROUND_TRUTH_PATH,
        test_dir=TEST_DIR,
        split=SPLIT,
        fold=FOLD,
        device=DEVICE,
        tokenizer_type=TOKENIZER_TYPE,
    )
