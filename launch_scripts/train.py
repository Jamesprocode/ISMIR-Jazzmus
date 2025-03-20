import gc

import fire
import gin
import torch

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from jazzmus.dataset.ctc_datamodule import CTCDataModule
from jazzmus.dataset.smt_dataset import GrandStaffDataset
from jazzmus.model.crnn.model import CTCTrainedCRNN
from jazzmus.smt_trainer import SMT_Trainer
from jazzmus.utils.file_utils import check_folders


PYTORCH_ENABLE_MPS_FALLBACK = 1


def train(
    debug: bool = False,
    fold: int = 0,
    model_type: str = None,
    epochs: int = 300,
    patience: int = 10,
    batch_size: int = 16,
    config: str = None,
):
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(seed=42, workers=True)

    # setup all the configurations
    gin.parse_config_file(config)

    check_folders()

    print("EXPERIMENT TRAINING")
    # print(f"\tDataset: {ds_name}")
    print(f"\tFold: {fold}")
    print(f"\tModel type: {model_type}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tBatch size: {batch_size}")
    # print(f"\tSplit encoding: {split_enc}")
    # print(f"\tHarmony processing: {harm_proc}")

    if model_type == "crnn":
        datamodule = CTCDataModule(fold, batch_size)
        datamodule.setup(stage="fit")
        w2i, i2w = datamodule.get_w2i_and_i2w()

        model = CTCTrainedCRNN(
            w2i=w2i, i2w=i2w, max_image_len=datamodule.get_max_img_len(), fold=fold
        )

        datamodule.width_reduction = model.width_reduction

    elif model_type == "smt":
        # datamodule
        datamodule = GrandStaffDataset(fold=fold, batch_size=batch_size)
        train_dataloader_ = datamodule.train_dataloader()

        # batch = next(iter(train_dataloader_))
        # images = batch[0]
        # from matplotlib import pyplot as plt

        # fig = plt.figure()
        # gs = fig.add_gridspec(len(images), hspace=0)
        # axs = gs.subplots(sharex=True)
        # fig.suptitle("Batch of images")

        # for i in range(len(images)):
        #     print(images[i].shape)
        #     image_to_plot = images[i].numpy().squeeze(0)
        #     axs[i].imshow(image_to_plot, cmap="gray")
        # for ax in axs:
        #     ax.label_outer()
        # plt.savefig("random_batch.png")

        max_height, max_width = datamodule.train_set.get_max_hw()
        max_len = datamodule.train_set.get_max_seqlen()

        model = SMT_Trainer(
            maxh=int(max_height),
            maxw=int(max_width),
            maxlen=int(max_len),
            out_categories=len(datamodule.train_set.w2i),
            padding_token=datamodule.train_set.w2i["<pad>"],
            in_channels=1,
            w2i=datamodule.train_set.w2i,
            i2w=datamodule.train_set.i2w,
            d_model=256,
            dim_ff=256,
            num_dec_layers=8,
            fold=fold,
        )

    else:
        raise ValueError(f"Model type {model_type} not recognized")

    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{model_type}",
            filename=f"{model_type}_{fold}",
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
            min_delta=0.01,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            divergence_threshold=100.0,
            check_on_train_epoch_end=False,
        ),
    ]

    my_logger = WandbLogger(
        project="jazzmus",
        name=f"{model_type}_{fold}",
        log_model=True,
        group=f"{model_type}",
        save_dir="logs",
    )

    trainer = Trainer(
        logger=my_logger,
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=5,
        deterministic=False,
        benchmark=False,
        precision="16-mixed",
        accelerator="auto",
        accumulate_grad_batches=8,
        fast_dev_run=debug,
    )

    trainer.fit(model=model, datamodule=datamodule)

    # End of training, test partition
    if model_type == "crnn":
        model = CTCTrainedCRNN.load_from_checkpoint(callbacks[0].best_model_path)
    elif model_type == "smt":
        model = SMT_Trainer.load_from_checkpoint(callbacks[0].best_model_path)

    model.freeze()
    trainer.test(model, datamodule)


if __name__ == "__main__":
    fire.Fire(train)
