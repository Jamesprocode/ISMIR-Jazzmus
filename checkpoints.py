import torch, glob
from tqdm import tqdm  # make sure you have it: pip install tqdm

# Path pattern for your checkpoints
ckpt_paths = sorted(glob.glob("weights/smt/*.ckpt"))

scores = {}

print(f"Found {len(ckpt_paths)} checkpoints. Evaluating...\n")

for path in tqdm(ckpt_paths, desc="Loading checkpoints"):
    try:
        ckpt = torch.load(path, map_location="cpu")
        cb_key = list(ckpt["callbacks"].keys())[0]
        score = ckpt["callbacks"][cb_key].get("best_model_score", None)
        scores[path] = float(score) if score is not None else None
    except Exception as e:
        scores[path] = None
        print(f"Error reading {path}: {e}")

print("\nValidation SER values:")
print("-" * 50)
for name, val in scores.items():
    print(f"{name:<45}  {val}")

# find the best model (lowest SER)
valid_scores = {k: v for k, v in scores.items() if v is not None}
if valid_scores:
    best_model = min(valid_scores, key=valid_scores.get)
    print("\n✅ Best checkpoint:", best_model)
    print("   Lowest validation SER:", valid_scores[best_model])
else:
    print("\nNo valid scores found in checkpoints.")
