import gc

import fire
import gin
import torch
from torch.nn import Conv1d

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger

from jazzmus.dataset.full_page_smt_dataset import FullPageGrandStaffDataset
from jazzmus.smt_trainer import SMT_Trainer
from jazzmus.utils.file_utils import check_folders, print_smt_batch


def train_fullpage(
    debug: bool = False,
    fold: int = 0,
    epochs: int = 100,
    patience: int = 10,
    batch_size: int = 4,
    accumulate_grad_batches: int = 16,
    config: str = None,
    lr: float = 1e-6,
    checkpoint_path: str = "weights/smt_sys_best/smt_pre_syn_medium.ckpt",
    only_test: str = None,
):
    """
    Training script for full-page SMT model.
    Loads from a system-level checkpoint and fine-tunes on full-page images.
    """
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(seed=42, workers=True)

    # Setup gin configuration
    gin.parse_config_file(config)

    check_folders()

    print("FULL-PAGE SMT TRAINING")
    print(f"\tFold: {fold}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tBatch size: {batch_size}")
    print(f"\tAccumulate grad batches: {accumulate_grad_batches}")
    print(f"\tLearning rate: {lr}")
    print(f"\tCheckpoint: {checkpoint_path}")

    # Initialize datamodule
    checkpoint = torch.load(checkpoint_path)
    ckpt_w2i = checkpoint['hyper_parameters']['w2i']
    ckpt_i2w = checkpoint['hyper_parameters']['i2w']
    
    datamodule = FullPageGrandStaffDataset(fold=fold, batch_size=batch_size)
    print_smt_batch(datamodule.train_dataloader())
    datamodule.train_set.set_dictionaries(ckpt_w2i, ckpt_i2w)
    datamodule.val_set.set_dictionaries(ckpt_w2i, ckpt_i2w)
    datamodule.test_set.set_dictionaries(ckpt_w2i, ckpt_i2w)
    # Get max dimensions from all splits
    max_height = max(
        datamodule.train_set.get_max_hw()[0],
        datamodule.val_set.get_max_hw()[0],
        datamodule.test_set.get_max_hw()[0]
    )

    max_width = max(
        datamodule.train_set.get_max_hw()[1],
        datamodule.val_set.get_max_hw()[1],
        datamodule.test_set.get_max_hw()[1]
    )

    max_len = max(
        datamodule.train_set.get_max_seqlen(),
        datamodule.val_set.get_max_seqlen(),
        datamodule.test_set.get_max_seqlen()
    )

    # Add buffer to max_len for safety
    max_len = int(max_len * 10)

    print(f"\tMax height: {max_height}")
    print(f"\tMax width: {max_width}")
    print(f"\tMax sequence length: {max_len}")
    print(f"\tVocab size: {len(datamodule.train_set.w2i)}")

    # Load model from checkpoint
    if only_test is None:
        print(f"\nLoading model from checkpoint: {checkpoint_path}")

        # Use simplified approach from fp-train-2.py
        # Just pass full-page dimensions directly to load_from_checkpoint
        # PyTorch Lightning will handle the initialization and state dict loading
    # Step 1: Create model with NEW dimensions
    model = SMT_Trainer(
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
    )

    # Step 2: Load checkpoint weights with strict=False
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # Drop positional encoding buffers so they stay re-initialized with new max_len
    # for k in list(state_dict.keys()):
    #     if "positional_1D.pe" in k:
    #         del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    
    print(f"\nLoading checkpoint with strict=False:")
    # print(f"  Missing keys: {len(missing_keys)}")
    # print(f"  Unexpected keys: {len(unexpected_keys)}")

    # This should show that positional encoding layers have size mismatches
    # Those will keep the NEW initialized values, everything else loads from checkpoint
    print(f"Loaded checkpoint with full-page dimensions")
    print(f"  Input: maxh={int(max_height)}, maxw={int(max_width)}, vocab={len(datamodule.train_set.w2i)}")

    tokenizer_type = gin.query_parameter("GrandStaffFullPage.tokenizer_type")
    print(f"\tTokenizer type: {tokenizer_type}")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="weights/smt_fullpage",
            filename=f"smt_fullpage_{fold}",
            monitor="val/ser",
            verbose=True,
            save_top_k=1,
            save_last=False,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=False,
            every_n_epochs=5,
            save_on_train_epoch_end=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Wandb logger
    wandb_name = f"smt_fullpage_{tokenizer_type}_lr{lr}_e{epochs}"

    my_logger = WandbLogger(
        project="jazzmus",
        name=wandb_name,
        log_model=True,
        group="smt_fullpage",
        save_dir="logs",
    )

    # Trainer
    trainer = Trainer(
        logger=my_logger,
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=5,
        deterministic=False,
        benchmark=False,
        precision="bf16-mixed",
        accelerator="auto",
        accumulate_grad_batches=accumulate_grad_batches,
        fast_dev_run=debug,
    )

    if only_test is None:
        trainer.fit(model=model, datamodule=datamodule)

        # Load best model for testing
        model = SMT_Trainer.load_from_checkpoint(callbacks[0].best_model_path)
    else:
        # Load from wandb artifact for testing
        if not only_test.startswith("university"):
            only_test = "university-alicante/jazzmus/model-" + only_test + ":best"
        
        import wandb
        run = wandb.init()
        artifact = run.use_artifact(only_test, type="model")
        artifact_dir = artifact.download()
        model = SMT_Trainer.load_from_checkpoint(artifact_dir + "/model.ckpt")

    model.freeze()
    model.eval()

    # Validation and test
    trainer.validate(model, datamodule=datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    fire.Fire(train_fullpage)