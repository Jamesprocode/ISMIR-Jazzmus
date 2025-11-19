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
from collections import defaultdict
from jazzmus.dataset.data_preprocessing import convert_img_to_tensor


IMAGE_PATH = "data/jazzmus_systems/jpg/img_10_1.jpg"  # Path to image (e.g., "path/to/image.jpg")
GROUND_TRUTH_PATH = "data/jazzmus_systems/gt/img_10_1.txt"  # Path to ground truth (e.g., "path/to/gt.txt")
checkpoint_path = "weights/smt/smt_0-v1.ckpt"  # Path to model checkpoint (e.g., "path/to/checkpoint.ckpt")

device = "cuda"
model = SMT_Trainer.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
        )

model.eval()
model.to(device)
print("Model loaded.")

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
img = np.array(img)
og_height, og_width = img.shape

height = 128
width = 1000

img = cv2.resize(img, (width, height))
print(f"✓ Image preprocessed: {(og_height, og_width)}->{(height, width)}")


img_tensor = convert_img_to_tensor(img)  # Returns (C, H, W) = (1, H, W)
img_tensor = img_tensor.unsqueeze(0) 

img_tensor = img_tensor.to(device)

print(f"Image tensor shape before unsqueeze: {img_tensor},{img_tensor.shape}")

print("Generating predictions...")
with torch.no_grad():
    predicted_tokens, logits = model.model.predict(input=img_tensor[0])
            
# Decode tokens to string
# predicted_tokens are already strings, no need to convert to int
if isinstance(predicted_tokens[0], int):
    # If tokens are integers, look them up in i2w
    token_strs = [model.model.i2w.get(int(t), "<unk>") for t in predicted_tokens]
else:
    print("If tokens are already strings, use them directly")
    token_strs = [str(t) for t in predicted_tokens]

prediction_str = untokenize(token_strs)



with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
    ground_truth = f.read().strip()
    
print("\n" + "=" * 60)
print("ground truth:")
print(ground_truth)
print("\n" + "=" * 60)
print("prediction:")
print("Prediction:", prediction_str)
print("=" * 60)



cer, ser, ler = compute_poliphony_metrics([prediction_str], [ground_truth])
print("\n" + "=" * 60)
print("SCORING RESULTS")
print(f"CER: {cer:.4f}")
print(f"SER: {ser:.4f}")
print(f"LER: {ler:.4f}")
print("=" * 60)