"""
Curriculum-aware SMT trainer.

Subclasses the base SMT_Trainer without touching it, adding only:
  - set_stage() / set_stage_calculator()  for curriculum stage tracking
  - stage logging to wandb on every training step

All system-level and existing full-page training scripts continue to use
the original jazzmus.smt_trainer.SMT_Trainer unchanged.
"""

from typing import Callable

import wandb
from jazzmus.smt_trainer import SMT_Trainer


class CurriculumSMTTrainer(SMT_Trainer):
    """
    Extends SMT_Trainer with curriculum stage tracking.

    Usage in train_fullpage_cl.py:
        model = CurriculumSMTTrainer.load_from_checkpoint(
            ckpt_path, maxh=..., maxw=..., maxlen=..., strict=False, ...
        )
        model.set_stage(datamodule.train_set.curriculum_stage_beginning)
        model.set_stage_calculator(datamodule.train_set.get_stage_calculator())
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_stage: int = 2
        self._stage_calculator: Callable[[int], int] = lambda step: self.current_stage

    # ── stage control ──────────────────────────────────────────────────────────

    def set_stage(self, stage: int):
        self.current_stage = stage

    def set_stage_calculator(self, calc: Callable[[int], int]):
        """
        Inject a step → stage calculator so training_step can log the
        current curriculum stage to wandb.
        """
        self._stage_calculator = calc

    # ── training / validation step overrides ──────────────────────────────────

    def on_load_checkpoint(self, checkpoint):
        """Drop weights whose shape doesn't match the current model.

        Handles vocab-size mismatches between the pretrained checkpoint
        (e.g. 20578 tokens) and the curriculum model (jazz vocab, 154 tokens).
        Mismatched layers (embedding, output Conv1d) are skipped so they keep
        their random initialisation; all other weights load normally.
        """
        state = checkpoint["state_dict"]
        own   = self.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in own and v.shape == own[k].shape}
        skipped = [k for k in state if k not in filtered]
        if skipped:
            print(f"  Skipping {len(skipped)} mismatched weights: {skipped}")
        checkpoint["state_dict"] = filtered

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)

        stage = self._stage_calculator(self.global_step)
        self.log("curriculum/stage", float(stage), on_step=True, prog_bar=True)

        # Log a sample image every 200 optimizer steps so WandB shows
        # curriculum stacks at each stage (base class only logs at batch_idx==0
        # which is once per 40 000-sample epoch — far too infrequent).
        if self.global_step % 200 == 0:
            x = batch[0]
            img_np = x[0].squeeze().cpu().numpy()
            self.logger.experiment.log({
                "curriculum/sample_image": wandb.Image(
                    img_np,
                    caption=f"step={self.global_step}  stage={stage}",
                ),
            })

        return loss

    def validation_step(self, batch, batch_idx):
        (x, di, y, path_to_images) = batch
        loss = self.compute_loss(batch)
        self.log("val/loss", loss, on_epoch=True, batch_size=x.shape[0], prog_bar=True)

        # Cap greedy-decode steps proportionally to the current curriculum stage.
        # model.maxlen is sized for the worst case (5 stacked systems + 10 % buffer).
        # At stage 2 the GT is only ~2/5 of that length, so decoding to the full
        # maxlen wastes ~3× the time.  We cap at (stage × 550) tokens which gives
        # generous headroom (~50 % above the expected per-system token count of ~346).
        stage = int(self._stage_calculator(self.global_step))
        capped_maxlen = min(self.model.maxlen, max(512, stage * 550))
        old_maxlen = self.model.maxlen
        self.model.maxlen = capped_maxlen
        try:
            self.predict_output(batch)
        finally:
            self.model.maxlen = old_maxlen
