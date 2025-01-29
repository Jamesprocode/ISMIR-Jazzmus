import gc

import fire
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from nn.crnn.model import CTCTrainedCRNN
from utils.ctc_datamodule import CTCDataModule
from utils.file_utils import check_folders, load_config
from utils.seed import seed_everything


def train(
    ds_name: str,
    fold: int = 0,
    config: str = None,
):
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(42, benchmark=True)
    check_folders()

    model_type, epochs, patience, batch_size, logger = load_config(config).values()

    print("EXPERIMENT TRAINING")
    print(f"\tDataset: {ds_name}")
    print(f"\tFold: {fold}")
    print(f"\tModel type: {model_type}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tBatch size: {batch_size}")
    print(f"\tLogger: {logger}")

    datamodule = CTCDataModule(ds_name, fold, batch_size).setup(stage="fit")
    w2i, i2w = datamodule.get_w2i_and_i2w()

    model = CTCTrainedCRNN(
        w2i, i2w, max_image_len=datamodule.get_max_img_len(), fold=fold
    )

    datamodule.width_reduction = model.width_reduction

    callbacks = [
        ModelCheckpoint(
            dirpath=f"weigths/{model_type}",
            filename=ds_name + f"_{fold}",
            monitor="val_ser",
            verbose=True,
            save_top_k=1,
            save_last=False,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=False,
            every_n_epochs=5,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            monitor="val_ser",
            min_delta=0.1,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            divergence_threshold=100.0,
            check_on_train_epoch_end=False,
        ),
    ]

    # Something Wandb in servers crashes
    my_logger = None
    if logger:
        my_logger = WandbLogger(
            project="jazzmus",
            name=f"{ds_name}_{fold}",
            log_model=True,
            group=f"{model_type}",
        )
    else:
        my_logger = CSVLogger("logs", name=f"{ds_name}_{fold}")

    trainer = Trainer(
        logger=my_logger,
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=5,
        deterministic=False,
        benchmark=False,
        precision="16-mixed",
    )

    trainer.fit(model, datamodule)

    # End of training, test partition
    model = CTCTrainedCRNN.load_from_checkpoint(callbacks[0].best_model_path)

    model.freeze()
    trainer.test(model, datamodule)


if __name__ == "__main__":
    fire.Fire(train)
