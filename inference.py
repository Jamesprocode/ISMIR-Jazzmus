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
from jazzmus.dataset.tokenizer import untokenize
from jazzmus.metrics import compute_metrics
from jazzmus.dataset.eval_functions import compute_poliphony_metrics

class FullPageInference:
    """Inference pipeline for full-page jazz leadsheet recognition."""

    def __init__(self, checkpoint_path="ismiromrjazz/ISMIR-Jazzmus/weights/smt_0-v6.ckpt", device="cpu"):
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

    def preprocess_image(self, image_path, max_height=1024, max_width=2048):
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
        # img = cv2.resize(img, (new_width, new_height))

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

        # Pad to max size
        padded = torch.zeros(1, 1, max_height, max_width)
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
        print(f"\nPredicted kern (first 500 chars):")
        print(result['prediction'][:500])
        if len(result['prediction']) > 500:
            print("...")
        print("=" * 60)

    def score_with_ground_truth(self, prediction, ground_truth_path):
        """
        Calculate metrics by comparing prediction with ground truth.

        Args:
            prediction: Predicted **kern string
            ground_truth_path: Path to ground truth file

        Returns:
            dict: Metrics (SER, SEQ-ER)
        """
        # Load ground truth
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()

        # Calculate metrics
        # metrics = compute_metrics([ground_truth], [prediction])
        cer, ser, ler = compute_poliphony_metrics([prediction], [ground_truth])

        # Display metrics
        print("\n" + "=" * 60)
        print("SCORING RESULTS")
        print("=" * 60)
        print(f"SER (System Error Rate):     {ser:.2f}%")
        print(f"CER (Character Error Rate): {cer:.2f}%")
        print(f"LER (Line Error Rate): {ler:.2f}%")
        print(f"Ground truth length:         {len(ground_truth)}")
        print(f"Prediction length:           {len(prediction)}")
        print("=" * 60 + "\n")

        return {"ser": ser, "cer": cer, "ler": ler}


def evaluate_test_set(
    checkpoint_path="smt_0-v6.ckpt",
    test_dir="data/handwritten-fullpage",
    split="test",
    fold=0,
    output_dir=None,
    device="cuda",
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
    failed = []

    print(f"Running inference on {len(img_paths)} images...\n")
    for idx, (img_path, gt_path) in enumerate(tqdm(zip(img_paths, gt_paths), total=len(img_paths), desc="Evaluating")):
        try:
            # Run inference
            result = inference.predict(img_path)
            prediction = result['prediction']

            # Load ground truth
            with open(gt_path, 'r', encoding='utf-8') as f:
                ground_truth = f.read().strip()

            all_predictions.append(prediction)
            all_ground_truths.append(ground_truth)

            # Calculate per-sample metrics
            cer, ser, ler = compute_poliphony_metrics([prediction], [ground_truth])
            individual_metrics.append({
                "image": Path(img_path).name,
                "cer": cer,
                "ser": ser,
                "ler": ler,
            })

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

    print("\n" + "-" * 70)
    print("AGGREGATE METRICS")
    print("-" * 70)
    print(f"CER (Character Error Rate):  {cer_agg:.2f}%")
    print(f"SER (Sequence Error Rate):   {ser_agg:.2f}%")
    print(f"LER (Line Error Rate):       {ler_agg:.2f}%")
    print("=" * 70)

    # Show per-sample statistics
    if individual_metrics:
        print("\nPER-SAMPLE STATISTICS")
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
    checkpoint_path = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/weights/smt/smt_0.ckpt",
    image_path=None,
    output_path=None,
    ground_truth_path=None,
    test_dir="/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/data/handwritten",
    split="test",
    fold=0,
    device="cuda",
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
        evaluate_test_set(checkpoint_path, test_dir, split, fold, output_path, device)
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
        inference.score_with_ground_truth(result['prediction'], ground_truth_path)



if __name__ == "__main__":
    import fire

    fire.Fire(run_inference)
