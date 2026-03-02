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

    # ── training step override (adds stage logging) ───────────────────────────

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

        return loss
