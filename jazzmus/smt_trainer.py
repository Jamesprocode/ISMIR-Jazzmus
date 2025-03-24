import random

import lightning.pytorch as L
import torch

from jazzmus.dataset.eval_functions import compute_poliphony_metrics
from jazzmus.model.smt.configuration_smt import SMTConfig
from jazzmus.model.smt.modeling_smt import SMTModelForCausalLM
from torchinfo import summary
from jazzmus.dataset.tokenizer import untokenize
import gin
import wandb

@gin.configurable
class SMT_Trainer(L.LightningModule):
    def __init__(
        self,
        maxh,
        maxw,
        maxlen,
        out_categories,
        padding_token,
        in_channels,
        w2i,
        i2w,
        d_model=256,
        dim_ff=256,
        num_dec_layers=8,
        cp=None,
        texture=None,
        fold=None,
        lr=1e-4,
    ):
        super().__init__()
        self.config = SMTConfig(
            maxh=maxh,
            maxw=maxw,
            maxlen=maxlen,
            out_categories=out_categories,
            padding_token=padding_token,
            in_channels=in_channels,
            w2i=w2i,
            i2w=i2w,
            d_model=d_model,
            dim_ff=dim_ff,
            attn_heads=4,
            num_dec_layers=num_dec_layers,
            use_flash_attn=True,
        )
        self.model = SMTModelForCausalLM(self.config)
        self.padding_token = padding_token
        self.texture = texture
        self.fold = fold
        self.lr = lr

        self.preds = []
        self.grtrs = []

        self.save_hyperparameters()

        summary(
            self,
            input_size=[
                (1, 1, self.config.maxh, self.config.maxw),
                (1, self.config.maxlen),
            ],
            dtypes=[torch.float, torch.long],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.model.encoder.parameters())
            + list(self.model.decoder.parameters()),
            lr=self.lr,  # Peak LR
            amsgrad=False,
        )

        return optimizer

    def forward(self, input, last_preds):
        return self.model(input, last_preds)

    def training_step(self, batch, batch_idx):
        (
            x,
            di,
            y,
        ) = batch
        outputs = self.model(x, di[:, :-1], labels=y)
        loss = outputs.loss
        self.log("loss", loss, on_epoch=True, batch_size=1, prog_bar=True)

        if batch_idx == 0:
            # log on wandb an image with the first batch
            # Stack all images in the batch vertically
            stacked_images = torch.cat([x[i] for i in range(x.shape[0])], dim=-2)  # dim=-2 is height dimension
            self.logger.experiment.log({
                "input_images": wandb.Image(stacked_images.squeeze().cpu().numpy(), 
                                        caption=f"Input images - Epoch {self.current_epoch}")
            })

        return loss

    def validation_step(self, val_batch, batch_idx):
        (
            x,
            di,
            y,
        ) = val_batch

        predicted_sequence, _ = self.model.predict(input=x)

        dec = untokenize(predicted_sequence)
        gt = untokenize([self.model.i2w[token.item()] for token in y.squeeze(0)[:-1]])

        # dec = "".join(predicted_sequence)
        # dec = dec.replace("<t>", "\t")
        # dec = dec.replace("<b>", "\n")
        # dec = dec.replace("<s>", " ")

        # gt = "".join([self.model.i2w[token.item()] for token in y.squeeze(0)[:-1]])
        # gt = gt.replace("<t>", "\t")
        # gt = gt.replace("<b>", "\n")
        # gt = gt.replace("<s>", " ")

        self.preds.append(dec)
        self.grtrs.append(gt)

    def on_validation_epoch_end(self, metric_name="val") -> None:
        cer, ser, ler = compute_poliphony_metrics(self.preds, self.grtrs)

        random_index = random.randint(0, len(self.preds) - 1)
        predtoshow = self.preds[random_index]
        gttoshow = self.grtrs[random_index]
        print(f"\n[Prediction] - {predtoshow}")
        print(f"\n[GT] - {gttoshow}")

        self.log(
            f"{metric_name}_cer",
            100 if cer > 100 else cer,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{metric_name}_ser",
            100 if ser > 100 else ser,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{metric_name}_ler",
            100 if ler > 100 else ler,
            on_epoch=True,
            prog_bar=True,
        )

        if metric_name == "test":
            import os

            os.makedirs(f"test_predictions/{self.texture}/{self.fold}/", exist_ok=True)
            # save all the test predictions in a folder with pairs of prediction and ground truth .kern files
            for i in range(len(self.preds)):
                with open(
                    f"test_predictions/{self.texture}/{self.fold}/{i}_pred.kern", "w"
                ) as f:
                    f.write(self.preds[i])
                with open(
                    f"test_predictions/{self.texture}/{self.fold}/{i}_gt.kern", "w"
                ) as f:
                    f.write(self.grtrs[i])

        self.preds = []
        self.grtrs = []

        return ser

    def test_step(self, test_batch):
        return self.validation_step(test_batch, metric_name="test")

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end("test")
