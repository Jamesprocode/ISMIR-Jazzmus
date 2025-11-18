"""
Run test evaluation using the trainer's built-in test() method.
This ensures we use the exact same preprocessing and inference pipeline as training.
"""

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

from jazzmus.smt_trainer import SMT_Trainer
from jazzmus.dataset.smt_dataset import GrandStaffDataset
from jazzmus.utils.file_utils import check_folders, print_smt_batch


# Configuration
CHECKPOINT_PATH = "weights/smt/smt_0-v1.ckpt"
DATA_DIR = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/data/jazzmus_systems/splits"
SPLIT = "test"
FOLD = 0
DEVICE = "cuda"

print(f"Loading checkpoint: {CHECKPOINT_PATH}")
print(f"Data directory: {DATA_DIR}")
print(f"Testing on {SPLIT} split\n")

# Load model from checkpoint
model = SMT_Trainer.load_from_checkpoint(CHECKPOINT_PATH)
datamodule = GrandStaffDataset(data_path=DATA_DIR, fold=FOLD, batch_size=64)

print_smt_batch(datamodule.train_dataloader())

# Create trainer for testing
trainer = pl.Trainer(
    accelerator=DEVICE,
    devices=1,
    logger=False,
    enable_progress_bar=True,
)

# Run test - this uses the exact same inference pipeline as training
print("=" * 70)
print("RUNNING TEST EVALUATION")
print("=" * 70 + "\n")
model.eval()
trainer.test(model, datamodule)

print("\n" + "=" * 70)
print("Test predictions saved to: test_predictions/{texture}/{fold}/")
print("=" * 70)