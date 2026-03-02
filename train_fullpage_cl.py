"""
Full-page curriculum learning training for jazz lead sheet recognition.

Phase 2 of the two-phase training pipeline:
  Phase 1 → train.py           (system-level, produces the starting checkpoint)
  Phase 2 → train_fullpage_cl  (this script, curriculum from systems → full pages)

Curriculum stages (each lasts `increase_steps` steps):
  stage 1: single systems (vocab warm-up)
  stage 2: stack up to 2 systems
  stage 3: stack up to 3 systems
  stage 4: stack up to 4 systems
  stage 5: stack up to 5 systems
  fine-tune A: stacked (prob 90%→20%) + real full pages
  fine-tune B: real full pages only

Example usage:
  python train_fullpage_cl.py \
      --config config/full-page/fullpage_cl.gin \
      --checkpoint_path weights/smt/smt_0.ckpt \
      --fold 0
"""

import gc

import fire
import gin
import torch
from torch.nn import Conv1d

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from jazzmus.curriculum.trainer import CurriculumSMTTrainer
from jazzmus.curriculum.dataset import JazzCLDataModule
from jazzmus.utils.file_utils import check_folders


def train(
    config: str,
    checkpoint_path: str,
    fold: int = 0,
    epochs: int = 10000,
    batch_size: int = 1,
    accumulate_grad_batches: int = None,  # read from gin; CLI overrides
    lr: float = None,                     # read from gin; CLI overrides
    skip_cl_epochs: int = 0,
    debug: bool = False,
):
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(seed=42, workers=True)

    gin.parse_config_file(config)
    check_folders()

    # Resolve hyperparams: gin config is the default, CLI arg overrides
    if lr is None:
        lr = gin.query_parameter("train.lr")
    if accumulate_grad_batches is None:
        accumulate_grad_batches = gin.query_parameter("train.accumulate_grad_batches")

    print("FULL-PAGE CURRICULUM TRAINING")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Config     : {config}")
    print(f"  Fold       : {fold}")
    print(f"  LR         : {lr}")
    print(f"  Skip epochs: {skip_cl_epochs}")

    # ── datamodule ─────────────────────────────────────────────────────────────
    datamodule = JazzCLDataModule(fold=fold, batch_size=batch_size)

    train_h, train_w = datamodule.train_set.get_max_hw()
    test_h,  test_w  = datamodule.test_set.get_max_hw()
    max_height = max(train_h, test_h)
    max_width  = max(train_w, test_w)

    # Take the max across train and test, then add 10 % headroom so the model
    # is never forced to truncate a sequence it has never seen before.
    max_len = int(max(
        datamodule.train_set.get_max_seqlen(),
        datamodule.test_set.get_max_seqlen(),
    ) * 1.1)

    print(f"  Max H×W    : {max_height} × {max_width}")
    print(f"  Max seq len: {max_len} (with 10 % buffer)")
    print(f"  Vocab size : {len(datamodule.train_set.w2i)}")

    # ── model ──────────────────────────────────────────────────────────────────
    # Load system-level checkpoint into a model with larger positional encoding
    # (maxlen grows because we're now stacking multiple systems).
    # strict=False lets mismatched PE layers be re-initialised while keeping
    # all encoder / attention / decoder body weights.
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model = CurriculumSMTTrainer.load_from_checkpoint(
        checkpoint_path,
        maxh=int(max_height),
        maxw=int(max_width),
        maxlen=int(max_len),
        out_categories=len(datamodule.train_set.w2i),
        padding_token=datamodule.train_set.w2i["<pad>"],
        in_channels=1,
        w2i=datamodule.train_set.w2i,
        i2w=datamodule.train_set.i2w,
        lr=lr,
        fold=fold,
        load_pretrained=False,   # override saved hparam — weights come from ckpt
        strict=False,
    )

    # If vocab changed (system vs full-page joint vocab), replace the output head
    out_layer = model.model.decoder.out_layer
    expected_vocab = len(datamodule.train_set.w2i)
    if out_layer.out_channels != expected_vocab:
        print(f"  Replacing output layer: {out_layer.out_channels} → {expected_vocab}")
        model.model.decoder.out_layer = Conv1d(
            out_layer.in_channels, expected_vocab, kernel_size=1
        )
    else:
        print(f"  Output layer OK: {expected_vocab} tokens")

    # Wire up curriculum stage tracking
    model.set_stage(datamodule.train_set.curriculum_stage_beginning)
    model.set_stage_calculator(datamodule.train_set.get_stage_calculator())

    # ── callbacks ──────────────────────────────────────────────────────────────
    increase_epochs = gin.query_parameter("JazzCurriculumDataset.increase_epochs")
    num_cl_stages   = gin.query_parameter("JazzCurriculumDataset.num_cl_stages")

    best_ckpt = ModelCheckpoint(
        dirpath="weights/curriculum",
        filename=f"jazz_cl_fold{fold}_best",
        monitor="val/ser",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=True,
        save_on_train_epoch_end=False,
    )

    # One checkpoint saved at the end of each CL stage
    stage_ckpt = ModelCheckpoint(
        dirpath="weights/curriculum",
        filename=f"jazz_cl_fold{fold}_stage",
        monitor="curriculum/stage",
        mode="max",
        save_top_k=num_cl_stages,
        verbose=True,
        enable_version_counter=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ── logger ─────────────────────────────────────────────────────────────────
    tokenizer_type = gin.query_parameter("JazzCurriculumDataset.tokenizer_type")
    wandb_logger = WandbLogger(
        project="jazz full page omr cl",
        name=f"cl_{tokenizer_type}_lr{lr}_fold{fold}",
        group="curriculum_fullpage",
        log_model=False,
        save_dir="logs",
    )

    # ── trainer ────────────────────────────────────────────────────────────────
    # One epoch = dataset_length / batch_size ≈ 3500 / 64 ≈ 55 optimizer steps —
    # same data volume as one Phase-1 epoch.  Stage advances every increase_epochs
    # epochs, independent of batch_size.
    total_cl_epochs = increase_epochs * num_cl_stages
    min_epochs      = total_cl_epochs - skip_cl_epochs * increase_epochs

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[best_ckpt, stage_ckpt, lr_monitor],
        max_epochs=epochs,
        min_epochs=min_epochs,
        precision="bf16-mixed",
        accelerator="auto",
        accumulate_grad_batches=accumulate_grad_batches,
        fast_dev_run=debug,
        deterministic=False,
    )

    # Give both train and val datasets access to trainer.global_step
    datamodule.train_set.set_trainer_data(trainer)
    datamodule.val_set.set_trainer_data(trainer)

    trainer.fit(model=model, datamodule=datamodule)

    # ── test ───────────────────────────────────────────────────────────────────
    best_model = CurriculumSMTTrainer.load_from_checkpoint(best_ckpt.best_model_path)
    best_model.freeze()
    best_model.eval()
    trainer.validate(best_model, datamodule=datamodule)
    trainer.test(best_model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(train)
