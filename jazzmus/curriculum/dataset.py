"""
Curriculum dataset for full-page jazz lead sheet recognition.

Strategy: start from a system-level checkpoint and gradually expose the model
to longer inputs by stacking N real system images vertically.

Stages (each lasts `increase_steps` steps):
    stage 2 → N ∈ {1, 2}
    stage 3 → N ∈ {1, 2, 3}
    stage 4 → N ∈ {1, 2, 3, 4}
    stage 5 → N ∈ {1, 2, 3, 4, 5}
Fine-tune → probability P (90→20 %) stacked, rest = real full pages
"""

import random
import numpy as np
import cv2
import torch
import gin

from torch.utils.data import Dataset
from lightning import LightningDataModule

from jazzmus.dataset.data_preprocessing import augment, convert_img_to_tensor
from jazzmus.dataset.smt_dataset import load_set as load_system_set
from jazzmus.dataset.full_page_smt_dataset import (
    GrandStaffFullPage,
    batch_preparation_img2seq,
)
from jazzmus.dataset.smt_dataset_utils import check_and_retrieveVocabulary
from jazzmus.dataset.tokenizer import process_text


# ─────────────────────────── core stacking primitive ──────────────────────────

def _strip_header(tokens):
    """Return tokens from first barline onward (drops kern header lines)."""
    for i, tok in enumerate(tokens):
        if tok.startswith("="):
            return tokens[i:]
    return tokens


def _strip_leading_linebreaks(tokens):
    """Remove !!linebreak:original lines that appear before the first barline.

    System files from the middle of a page include !!linebreak:original in
    their header area (between the kern spine declarations and the first =
    barline).  When used as the first system in a stack (j=0), this must be
    dropped so the stacked page never opens with a linebreak marker.
    """
    lines, cur = [], []
    for tok in tokens:
        cur.append(tok)
        if tok == "<n>":
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)

    result = []
    first_barline_seen = False
    for line in lines:
        content = [t for t in line if t not in ("<t>", "<n>")]
        if first_barline_seen:
            result.extend(line)
        elif any(t.startswith("=") for t in content):
            first_barline_seen = True
            result.extend(line)
        elif any(t == "!!linebreak:original" for t in content):
            pass  # drop this line — it has no place before the first barline
        else:
            result.extend(line)
    return result


def _strip_trailing_double_barlines(tokens):
    """Remove trailing double-barline (==) lines from a music token list.

    Individual system files often close their last measure with == (a kern
    final barline).  In a stacked, non-last system this is misleading because
    more music follows.  The !!linebreak:original separator already marks the
    system boundary, so the == is redundant and should be removed.
    """
    lines, cur = [], []
    for tok in tokens:
        cur.append(tok)
        if tok == "<n>":
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)

    while lines:
        content = [t for t in lines[-1] if t not in ("<t>", "<n>")]
        if content and all(t.startswith("==") for t in content):
            lines.pop()
        else:
            break
    return [t for line in lines for t in line]


def _split_music_tail(tokens):
    """Split a token sequence into (music, tail).

    Tail = trailing lines that contain no music tokens (i.e., only spine
    terminators like *-, !!linebreak:original, and similar).
    A "music line" is any line that has at least one token starting with a
    digit (note duration) or '=' (barline).
    """
    # Group tokens into lines (each ends with <n>)
    lines, cur = [], []
    for tok in tokens:
        cur.append(tok)
        if tok == "<n>":
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)

    # Walk backward to find the last music line
    split = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        content = [t for t in lines[i] if t not in ("<t>", "<n>")]
        if any(t[0].isdigit() or t.startswith("=") for t in content if t):
            split = i + 1
            break

    flat = lambda ls: [t for line in ls for t in line]
    return flat(lines[:split]), flat(lines[split:])


def stack_systems(imgs, gts, n, system_height=128, paths=None, width_tolerance=0.15):
    """
    Randomly sample n systems with similar widths and stack them vertically.

    Picks an anchor system at random, then samples the remaining N-1 from
    systems whose width is within `width_tolerance` of the anchor.  This keeps
    all rows on a synthetic page at roughly the same scale — mirroring how a
    real lead sheet has all systems spanning the same page width.

    Systems are normalised to `system_height` pixels tall (preserving aspect
    ratio).  Any remaining width difference inside the stack is padded with
    white (255).  GTs are concatenated: <bos> sys1_tokens sys2_tokens … <eos>.

    Args:
        imgs:            list of numpy (H, W) grayscale arrays.
        gts:             list of token lists starting with <bos>, ending with <eos>.
        n:               number of systems to stack.
        system_height:   target height in pixels for every row.
        paths:           optional list of file paths for debug logging.
        width_tolerance: max relative width difference allowed (default 15 %).

    Returns:
        stacked (numpy H×W), combined_gt (list of str tokens),
        sampled_paths (list of str | None)
    """
    # Pick an anchor, then find systems within width_tolerance of it
    anchor_idx = random.randrange(len(imgs))
    anchor_w   = imgs[anchor_idx].shape[1]

    compatible = [
        i for i, img in enumerate(imgs)
        if abs(img.shape[1] - anchor_w) / max(anchor_w, 1) <= width_tolerance
    ]

    # Fallback: if not enough compatible systems, open the pool to all
    if len(compatible) < n:
        compatible = list(range(len(imgs)))

    indices = [anchor_idx] + random.choices(compatible, k=n - 1)

    resized = []
    for idx in indices:
        img = imgs[idx]
        h, w = img.shape[:2]
        if h != system_height:
            new_w = max(1, int(round(w * system_height / h)))
            img = cv2.resize(img, (new_w, system_height), interpolation=cv2.INTER_LINEAR)
        resized.append(img)

    # Pad every row to the widest system in this stack
    max_w = max(img.shape[1] for img in resized)
    padded = [
        np.pad(img, ((0, 0), (0, max_w - img.shape[1])), constant_values=255)
        for img in resized
    ]
    stacked = np.vstack(padded)  # shape: (n * system_height, max_w)

    # Concatenate ground-truth token sequences:
    #   sys1: full kern header + music (tail stripped)
    #   sys2…N-1: music only (header stripped, tail stripped)
    #   sysN: music + *- spine terminators (tail kept)
    # !!linebreak:original is filtered by the tokenizer so never appears here.
    combined = ["<bos>"]
    for j, gt in enumerate([gts[i] for i in indices]):
        inner = gt[1:-1]          # strip <bos> / <eos>
        is_last = (j == n - 1)

        if j > 0:
            inner = _strip_header(inner)   # drop repeated kern header

        if is_last:
            combined += inner              # keep *- spine terminators
        else:
            music, _ = _split_music_tail(inner)   # drop *- tail
            combined += music
    combined += ["<eos>"]

    sampled_paths = [paths[i] for i in indices] if paths is not None else None
    return stacked, combined, sampled_paths


# ─────────────────────────── dataset ──────────────────────────────────────────

@gin.configurable
class JazzCurriculumDataset(Dataset):
    """
    Curriculum training set built by stacking real system-level images.

    No Verovio / MuseScore required – the pool is the system-level split
    already used for Phase-1 pre-training.
    """

    def __init__(
        self,
        system_data_path: str,
        fold: int = 0,
        split: str = "train",
        tokenizer_type: str = "medium",
        system_height: int = 128,
        increase_steps: int = 6000,
        num_cl_stages: int = 4,       # stages: 2, 3, 4, 5 systems
        curriculum_start: int = 2,    # first stage stacks up to this many systems
        max_synth_prob: float = 0.9,
        min_synth_prob: float = 0.2,
        finetune_steps: int = 20000,
        width_tolerance: float = 0.15,
        augment_images: bool = False,
        dataset_length: int = 40000,
        fullpage_data_path: str = "",
    ):
        super().__init__()

        self.system_height = system_height
        self.increase_steps = increase_steps
        self.num_cl_stages = num_cl_stages
        self.curriculum_start = curriculum_start
        self.max_synth_prob = max_synth_prob
        self.min_synth_prob = min_synth_prob
        self.finetune_steps = finetune_steps
        self.width_tolerance = width_tolerance
        self.augment_images = augment_images
        self.tokenizer_type = tokenizer_type
        self.teacher_forcing_error_rate = 0.2
        self.trainer = None
        self._dataset_length = dataset_length

        # Load system images at fixed height so stacking only needs width-padding
        raw_x, raw_y, raw_paths = load_system_set(
            system_data_path,
            fold=fold,
            split=split,
            fixed_img_height=system_height,
            max_fix_img_width=None,
            include_synthetic=False,
        )

        # Drop outlier systems whose compatible pool is too small to ever serve
        # as an anchor without triggering the fallback.
        min_pool = curriculum_start + num_cl_stages  # max n we'll ever stack
        widths = np.array([img.shape[1] for img in raw_x])
        keep = [
            i for i, w in enumerate(widths)
            if int(np.sum(np.abs(widths - w) / np.maximum(w, 1) <= width_tolerance)) >= min_pool
        ]
        raw_x    = [raw_x[i]    for i in keep]
        raw_y    = [raw_y[i]    for i in keep]
        raw_paths = [raw_paths[i] for i in keep]
        n_dropped = len(widths) - len(keep)
        if n_dropped:
            print(f"  Dropped {n_dropped} outlier systems (pool < {min_pool} at ±{int(width_tolerance*100)}%)")

        self.system_x = raw_x      # list of numpy (system_height, W) arrays
        self._raw_y = raw_y        # list of raw kern lines (list of str per image)
        self._paths = raw_paths    # for debug
        self.system_y = None       # populated by set_dictionaries()

        self.w2i = None
        self.i2w = None
        self.padding_token = None

        # Public: used by training script to initialise the model stage
        self.curriculum_stage_beginning = curriculum_start

        # Optional real full-page images for fine-tuning mix (prob decays 0.9→0.2)
        self._fullpage_dataset = None
        if fullpage_data_path:
            try:
                self._fullpage_dataset = GrandStaffFullPage(
                    data_path=fullpage_data_path,
                    split=split,
                    fold=fold,
                    augment=augment_images,
                )
                print(f"  Fine-tune mix: loaded {len(self._fullpage_dataset)} real full-page images")
            except FileNotFoundError:
                print(f"  Warning: no full-page {split} split at {fullpage_data_path}; fine-tune mix disabled")

    # ── vocabulary ────────────────────────────────────────────────────────────

    def _tokenize_gt(self):
        result = []
        for lines in self._raw_y:
            tokens = (
                ["<bos>"]
                + process_text(lines=lines, tokenizer_type=self.tokenizer_type)
                + ["<eos>"]
            )
            result.append(tokens)
        return result

    def get_gt(self):
        """Return tokenised GT for vocabulary building (lazy)."""
        if self.system_y is None:
            self.system_y = self._tokenize_gt()
        result = list(self.system_y)
        if self._fullpage_dataset is not None:
            result.extend(self._fullpage_dataset.get_gt())
        return result

    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i["<pad>"]
        if self.system_y is None:
            self.system_y = self._tokenize_gt()
        if self._fullpage_dataset is not None:
            self._fullpage_dataset.set_dictionaries(w2i, i2w)

    # ── stage / schedule helpers (mirrors SMT CurriculumTrainingDataset) ──────

    def set_trainer_data(self, trainer):
        """Inject the Lightning Trainer so __getitem__ can read global_step."""
        self.trainer = trainer

    def get_stage_calculator(self):
        """Return a closure: step → current curriculum stage integer."""
        increase_steps = self.increase_steps
        start = self.curriculum_start

        def calc(step):
            return (step // increase_steps) + start

        return calc

    def linear_scheduler_synthetic(self, step):
        """Probability of using a stacked sample during fine-tuning (decays 0.9→0.2)."""
        max_cl_steps = self.increase_steps * self.num_cl_stages
        frac = (step - max_cl_steps) / max(self.finetune_steps, 1)
        prob = self.max_synth_prob + frac * (self.min_synth_prob - self.max_synth_prob)
        return round(max(self.min_synth_prob, min(self.max_synth_prob, prob)), 4)

    # ── size helpers (used by DataModule / Trainer) ───────────────────────────

    def get_max_hw(self):
        max_sys_h = self.system_height * (self.curriculum_start + self.num_cl_stages)
        max_w = max(img.shape[1] for img in self.system_x)
        return max_sys_h, max_w

    def get_max_seqlen(self):
        max_sys_len = max(len(gt) for gt in self.system_y)
        return max_sys_len * (self.curriculum_start + self.num_cl_stages)

    # ── teacher forcing ───────────────────────────────────────────────────────

    def apply_teacher_forcing(self, sequence):
        errored = sequence.clone()
        for t in range(1, len(sequence)):
            if (
                np.random.rand() < self.teacher_forcing_error_rate
                and sequence[t] != self.padding_token
            ):
                errored[t] = np.random.randint(0, len(self.w2i))
        return errored

    # ── dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return self._dataset_length

    def __getitem__(self, index):
        step = self.trainer.global_step if self.trainer is not None else 0
        max_cl_steps = self.increase_steps * self.num_cl_stages

        if step < max_cl_steps:
            stage = (step // self.increase_steps) + self.curriculum_start
            n = random.randint(self.curriculum_start, stage)
        else:
            fine_tune_step = step - max_cl_steps
            if self._fullpage_dataset is not None and fine_tune_step >= self.finetune_steps:
                # Phase B: real full pages only
                idx = random.randrange(len(self._fullpage_dataset))
                return self._fullpage_dataset[idx]
            # Phase A: mix — synth prob decays 0.9 → 0.2 over finetune_steps
            synth_prob = self.linear_scheduler_synthetic(step)
            if self._fullpage_dataset is not None and random.random() > synth_prob:
                idx = random.randrange(len(self._fullpage_dataset))
                return self._fullpage_dataset[idx]
            # Stacked synthetic sample
            n = random.randint(self.curriculum_start, self.curriculum_start + self.num_cl_stages - 1)

        img, y_tokens, _ = stack_systems(
            self.system_x, self.system_y, n, self.system_height,
            width_tolerance=self.width_tolerance,
        )

        if self.augment_images:
            x = augment(img)
        else:
            x = convert_img_to_tensor(img)

        # Map tokens → ids (skip OOV, shouldn't happen with a complete vocab)
        y_ids = [self.w2i[tok] for tok in y_tokens if tok in self.w2i]
        y = torch.from_numpy(np.asarray(y_ids, dtype=np.int64))
        decoder_input = self.apply_teacher_forcing(y)

        label = f"stage{stage if step < max_cl_steps else 'ft'}_n{n}"
        return x, decoder_input, y, label


# ─────────────────────────── stacked val dataset ──────────────────────────────

@gin.configurable
class JazzStackedValDataset(Dataset):
    """
    Validation set generated on the fly from the system-level val split.

    Always stacks at the current curriculum stage's **maximum n** (not a random
    n like train), so val difficulty tracks training difficulty stage by stage:
        stage 2 → always 2 systems stacked
        stage 3 → always 3 systems stacked
        …
    No teacher forcing — decoder_input == target y.
    """

    def __init__(
        self,
        system_data_path: str,
        fold: int = 0,
        tokenizer_type: str = "medium",
        system_height: int = 128,
        increase_steps: int = 6000,
        num_cl_stages: int = 4,
        curriculum_start: int = 2,
        width_tolerance: float = 0.15,
        dataset_length: int = 200,
    ):
        super().__init__()
        self.system_height = system_height
        self.increase_steps = increase_steps
        self.num_cl_stages = num_cl_stages
        self.curriculum_start = curriculum_start
        self.width_tolerance = width_tolerance
        self._dataset_length = dataset_length
        self.tokenizer_type = tokenizer_type
        self.trainer = None

        raw_x, raw_y, _ = load_system_set(
            system_data_path,
            fold=fold,
            split="val",
            fixed_img_height=system_height,
            max_fix_img_width=None,
            include_synthetic=False,
        )

        # Same outlier filter as train
        min_pool = curriculum_start + num_cl_stages
        widths = np.array([img.shape[1] for img in raw_x])
        keep = [
            i for i, w in enumerate(widths)
            if int(np.sum(np.abs(widths - w) / np.maximum(w, 1) <= width_tolerance)) >= min_pool
        ]
        n_dropped = len(widths) - len(keep)
        if n_dropped:
            print(f"  Val: dropped {n_dropped} outlier systems (pool < {min_pool} at ±{int(width_tolerance*100)}%)")

        self.system_x = [raw_x[i] for i in keep]
        self._raw_y   = [raw_y[i] for i in keep]
        self.system_y = None
        self.w2i = None
        self.i2w = None

    def get_gt(self):
        if self.system_y is None:
            self.system_y = [
                ["<bos>"] + process_text(lines=lines, tokenizer_type=self.tokenizer_type) + ["<eos>"]
                for lines in self._raw_y
            ]
        return self.system_y

    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        if self.system_y is None:
            self.system_y = self.get_gt()

    def set_trainer_data(self, trainer):
        self.trainer = trainer

    def __len__(self):
        return self._dataset_length

    def __getitem__(self, index):
        step = self.trainer.global_step if self.trainer is not None else 0
        max_cl_steps = self.increase_steps * self.num_cl_stages

        if step < max_cl_steps:
            n = (step // self.increase_steps) + self.curriculum_start  # max n for stage
        else:
            n = self.curriculum_start + self.num_cl_stages - 1         # max n in fine-tune

        img, y_tokens, _ = stack_systems(
            self.system_x, self.system_y, n, self.system_height,
            width_tolerance=self.width_tolerance,
        )

        x = convert_img_to_tensor(img)
        y_ids = [self.w2i[tok] for tok in y_tokens if tok in self.w2i]
        y = torch.from_numpy(np.asarray(y_ids, dtype=np.int64))
        return x, y, y, f"val_n{n}"   # decoder_input = y (no teacher forcing)


# ─────────────────────────── datamodule ───────────────────────────────────────

@gin.configurable
class JazzCLDataModule(LightningDataModule):
    """
    DataModule for curriculum full-page training.

    - train: JazzCurriculumDataset  (stacked systems, generated on the fly)
    - val:   JazzStackedValDataset  (stacked systems from val split, on the fly,
                                     always at current stage max n)
    - test:  GrandStaffFullPage     (fixed full pages for final evaluation)
    """

    def __init__(
        self,
        system_data_path: str = "",
        fullpage_data_path: str = "",
        vocab_name: str = "vocab_cl",
        batch_size: int = 1,
        num_workers: int = 4,
        fold: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold

        self.train_set = JazzCurriculumDataset(
            system_data_path=system_data_path,
            fold=fold,
            split="train",
            augment_images=True,
            fullpage_data_path=fullpage_data_path,
        )

        self.val_set = JazzStackedValDataset(
            system_data_path=system_data_path,
            fold=fold,
        )

        self.test_set = GrandStaffFullPage(
            data_path=fullpage_data_path,
            split="test",
            fold=fold,
        )

        w2i, i2w = check_and_retrieveVocabulary(
            [
                self.train_set.get_gt(),
                self.val_set.get_gt(),
                self.test_set.get_gt(),
            ],
            "vocab",
            vocab_name,
        )

        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)

    def train_dataloader(self):
        import torch
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=batch_preparation_img2seq,
        )

    def val_dataloader(self):
        import torch
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=batch_preparation_img2seq,
        )

    def test_dataloader(self):
        import torch
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=batch_preparation_img2seq,
        )
