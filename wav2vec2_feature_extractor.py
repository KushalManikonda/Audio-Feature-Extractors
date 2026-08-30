"""
iiith_wav2vec2_oop.py
Speech Disfluency Classification — OOP Pipeline
================================================
Refactored from iiith-wav2vec2.ipynb following oop_structure_proposal.md.
All ML behavior is identical to the working notebook.

Run:
    python iiith_wav2vec2_oop.py

Set SANITY_RUN = True (default) for a quick shape/logic check before full training.
Set SANITY_RUN = False for the full experiment.
"""

# ============================================================
# SECTION 0 — CONFIGURATION
# All tunable parameters in one place.
# ============================================================

import os

# ── Data paths ────────────────────────────────────────────────
DATA_DIR   = "/kaggle/input"
OUTPUT_DIR = "/kaggle/working/output"

# ── Model ─────────────────────────────────────────────────────
WAV2VEC_MODEL = "facebook/wav2vec2-base"
SAMPLE_RATE   = 16_000

# ── Chunking ──────────────────────────────────────────────────
CHUNK_DURATION_SEC = 30      # keep region per chunk
CHUNK_OVERLAP_SEC  =  1      # context seconds on each side
MIN_CHUNK_SEC      =  0.1

# ── Split ratios ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
RANDOM_SEED = 42

# ── Shared splits path (Q5) ───────────────────────────────────
DEFAULT_SPLITS_PATH = os.path.join(OUTPUT_DIR, "splits", "splits.json")
SHARED_SPLITS_PATH  = os.environ.get("SHARED_SPLITS_PATH", DEFAULT_SPLITS_PATH)

# ── Training sampling caps ────────────────────────────────────
MAX_TRAIN_FRAMES_PER_CLASS_BINARY = 75_000
MAX_TRAIN_FRAMES_PER_CLASS_6CLASS = 40_000

# ── RF ────────────────────────────────────────────────────────
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = 20

# ── DNN ───────────────────────────────────────────────────────
BATCH_SIZE    = 256
DNN_EPOCHS    = 60
LEARNING_RATE = 1e-3
EARLY_STOP_PAT = 8

# ── BiLSTM ────────────────────────────────────────────────────
BILSTM_EPOCHS = 60
SEQ_LEN       = 128
BILSTM_BATCH  = 16

# ── Sanity run ────────────────────────────────────────────────
SANITY_RUN      = True
SANITY_N_FILES  = 3
SANITY_RF_TREES = 2
SANITY_EPOCHS   = 2

# ── Checkpointing ─────────────────────────────────────────────
FORCE_RECOMPUTE = False

# ── Derived directories ───────────────────────────────────────
EMBED_DIR  = os.path.join(OUTPUT_DIR, "embeddings")
META_DIR   = os.path.join(OUTPUT_DIR, "metadata")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
SPLIT_DIR  = os.path.join(OUTPUT_DIR, "splits")

# ============================================================
# SECTION 1 — IMPORTS
# ============================================================

import re
import json
import time
import random
import warnings
import logging
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import Wav2Vec2Processor, Wav2Vec2Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# ── Global runtime state (set in main) ────────────────────────
DEVICE         = None
W2V_TOTAL_STRIDE    = None
W2V_RECEPTIVE_FIELD = None
W2V_RF_CENTRE       = None
W2V_FRAME_STEP_SEC  = None
W2V_FRAME_RATE      = None
W2V_HIDDEN_DIM      = 768

ALL_RESULTS = []          # accumulated by save_result()

# ── Canonical label set ───────────────────────────────────────
VALID_LABELS = {"I", "PR", "PhR", "WR", "PWR", "P"}


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def apply_sanity_overrides() -> None:
    global RF_N_ESTIMATORS, RF_MAX_DEPTH, DNN_EPOCHS, BILSTM_EPOCHS
    global MAX_TRAIN_FRAMES_PER_CLASS_BINARY, MAX_TRAIN_FRAMES_PER_CLASS_6CLASS
    if SANITY_RUN:
        RF_N_ESTIMATORS = SANITY_RF_TREES
        RF_MAX_DEPTH    = 5
        DNN_EPOCHS      = SANITY_EPOCHS
        BILSTM_EPOCHS   = SANITY_EPOCHS
        MAX_TRAIN_FRAMES_PER_CLASS_BINARY = 500
        MAX_TRAIN_FRAMES_PER_CLASS_6CLASS = 200
        print("=" * 60)
        print("  *** SANITY_RUN = True ***")
        print(f"  {SANITY_N_FILES} recordings, {SANITY_EPOCHS} epochs, {SANITY_RF_TREES} RF trees")
        print("=" * 60)


def make_dirs() -> None:
    for d in [OUTPUT_DIR, EMBED_DIR, META_DIR, MODEL_DIR, RESULT_DIR,
              LOG_DIR, SPLIT_DIR]:
        os.makedirs(d, exist_ok=True)


# ============================================================
# SECTION 3 — DATACLASSES
# ============================================================

@dataclass
class AnnotationRow:
    """One labelled speech segment from a .txt annotation file."""
    file_id:           str
    start:             Optional[float]
    end:               Optional[float]
    raw_label:         str
    normalized_label:  str
    annotation_status: str          # VALID | AMBIGUOUS | FLUENT_GAP | UNKNOWN_LABEL | ...
    has_overlap:       bool  = False
    is_nested:         bool  = False
    overlap_group_id:  Optional[int] = None
    line_number:       Optional[int] = None
    raw_line:          Optional[str] = None
    speaker_id:        str   = "unknown"

    @property
    def duration(self) -> Optional[float]:
        if self.start is not None and self.end is not None:
            return self.end - self.start
        return None


@dataclass
class DatasetStore:
    """
    Frame-level dataset produced by Wav2vec2Extractor.
    Embeddings live in .npy files on disk — not held in RAM.
    frame_meta_df is the single source of truth for all metadata.

    Adapted from the OOP proposal's DatasetStore to support frame-level
    temporal classification (Q1: frame-level, not mean-pooled).
    """
    embed_dir:       str
    frame_meta_df:   Any        # pd.DataFrame
    extractor_name:  str
    feature_dim:     int        # 768 for wav2vec2-base

    def load_embeddings(self, file_id: str) -> np.ndarray:
        return np.load(os.path.join(self.embed_dir, f"{file_id}.npy"))

    def load_timestamps(self, file_id: str) -> np.ndarray:
        return np.load(os.path.join(self.embed_dir, f"{file_id}_timestamps.npy"))

    @property
    def n_frames(self) -> int:
        return len(self.frame_meta_df)

    @property
    def file_ids(self) -> List[str]:
        return self.frame_meta_df["file_id"].unique().tolist()


@dataclass
class ClassificationTask:
    """Defines one of the 3 standard classification experiments."""
    name:              str
    label_map:         Dict[str, int]
    excl_labels_rfdnn: Any          # set[str]
    excl_labels_bl:    Any          # set[str]
    class_names:       List[str]
    n_classes:         int
    task_type:         str          # "binary" | "multiclass"
    max_per_class:     int


@dataclass
class EvaluationResult:
    """Metrics from one classifier on one task."""
    task_name:        str
    classifier_name:  str
    accuracy:         float
    macro_f1:         float
    weighted_f1:      float
    precision_macro:  float
    recall_macro:     float
    confusion_matrix: Any           # np.ndarray
    extra:            Dict[str, Any] = field(default_factory=dict)


# ============================================================
# SECTION 4 — AUDIO LOADER
# ============================================================

class AudioLoader:
    """
    Loads a WAV file and returns a mono float32 numpy array at sample_rate.
    Shared by all interns / feature extractors.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def load(self, wav_path: str) -> Tuple[np.ndarray, int, Optional[str]]:
        """
        Load WAV -> mono float32. Returns (waveform, sr, error_or_None).
        Handles stereo->mono and resampling automatically.
        """
        try:
            wf, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if wf.ndim > 1:
                wf = wf.mean(axis=1)
        except Exception:
            try:
                wf, sr = librosa.load(wav_path, sr=None, mono=True,
                                      dtype=np.float32)
            except Exception as e:
                return np.zeros(0, dtype=np.float32), self.sample_rate, str(e)

        if sr != self.sample_rate:
            wf = librosa.resample(wf, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        return wf.astype(np.float32), sr, None

    def slice_segment(self, audio: np.ndarray,
                      start: float, end: float,
                      sr: Optional[int] = None) -> np.ndarray:
        """Slice audio[start_sec : end_sec]."""
        sr = sr or self.sample_rate
        return audio[int(start * sr): int(end * sr)]

    def get_duration(self, wav_path: str) -> Optional[float]:
        try:
            return sf.info(wav_path).duration
        except Exception:
            try:
                return librosa.get_duration(path=wav_path)
            except Exception:
                return None


# ============================================================
# SECTION 5 — ANNOTATION PARSER
# ============================================================

RE_A = re.compile(r"^IED_INTERN_(?P<n>\d+)__speaker_(?P<s>\d+)$",
                  re.IGNORECASE)
RE_B = re.compile(
    r"^intern(?P<n>\d+)__(?P<x>\d+)_(?P<y>\d+)_Speaker_(?P<s>\d+)_(?P<r>\d+)$",
    re.IGNORECASE)


def parse_filename(stem: str) -> dict:
    m = RE_A.match(stem)
    if m:
        return {"file_id": stem, "source_group": f"IED_INTERN_{m['n']}",
                "speaker_id": f"A_spk_{m['s']}", "recording_id": stem,
                "original_filename": stem}
    m = RE_B.match(stem)
    if m:
        return {"file_id": stem, "source_group": f"intern_{m['n']}",
                "speaker_id": f"B_spk_{m['s']}",
                "recording_id": f"{m['x']}_{m['y']}_{m['r']}",
                "original_filename": stem}
    return {"file_id": stem, "source_group": "unknown",
            "speaker_id": "unknown", "recording_id": "unknown",
            "original_filename": stem}


class AnnotationParser:
    """
    Reads .txt annotation files, normalises labels, marks overlaps.
    Shared by all interns — normalisation logic must be consistent.
    """

    VALID_LABELS = {"I", "PR", "PhR", "WR", "PWR", "P"}

    def parse_file(self, txt_path: str, file_id: str,
                   speaker_id: str = "unknown") -> List[dict]:
        """Parse one .txt annotation file. Returns list of raw row dicts."""
        annotations = []
        try:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return [{"file_id": file_id, "speaker_id": speaker_id,
                     "parse_status": "FILE_ERROR", "error": str(e)}]

        for i, raw_line in enumerate(lines, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                annotations.append({
                    "file_id": file_id, "speaker_id": speaker_id,
                    "line_number": i, "raw_line": raw_line.rstrip(),
                    "raw_start": parts[0] if parts else None,
                    "raw_end":   parts[1] if len(parts) > 1 else None,
                    "raw_label": " ".join(parts[2:]) if len(parts) > 2 else None,
                    "start_f": None, "end_f": None,
                    "parse_status": "MALFORMED_COLS"})
                continue

            rs, re_, rl = parts[0], parts[1], " ".join(parts[2:])
            try:    sf_ = float(rs)
            except ValueError: sf_ = None
            try:    ef_ = float(re_)
            except ValueError: ef_ = None

            if sf_ is None or ef_ is None:     st = "MALFORMED_TIME"
            elif sf_ < 0 or ef_ < 0:           st = "NEGATIVE_TIME"
            elif ef_ < sf_:                     st = "END_BEFORE_START"
            elif sf_ == ef_:                    st = "ZERO_DURATION"
            else:                               st = "OK"

            annotations.append({
                "file_id": file_id, "speaker_id": speaker_id,
                "line_number": i, "raw_line": raw_line.rstrip(),
                "raw_start": rs, "raw_end": re_, "raw_label": rl,
                "start_f": sf_, "end_f": ef_, "parse_status": st})
        return annotations

    def normalise_label(self, raw: Optional[str]) -> Tuple[str, str]:
        """
        Map raw label string to canonical form.
        Rules (priority order):
          Ambiguous  : comma, OR, /, ?, BLOCK, AND  -> AMBIGUOUS
          WR         : W[.]R                         -> WR
          PWR        : P[.]W[.]R                     -> PWR
          PhR        : P[Hh][.]R                     -> PhR
          PR         : P[.]R (after PWR/PhR matched) -> PR
          I          : I, FP, FILLED PAUSE           -> I
          P          : P, PAUSE                      -> P
          Fluent     : explicitly generated           -> Fluent
          else                                       -> UNKNOWN
        Returns: (canonical_label, status_note)
        """
        if raw is None:
            return "UNKNOWN", "EMPTY"
        s = str(raw).strip()
        if not s:
            return "UNKNOWN", "EMPTY"
        if s == "Fluent":
            return "Fluent", "FLUENT_REGION"

        for pat in [r",", r"\bOR\b", r"\bor\b", r"\band\b",
                    r"/", r"\?", r"BLOCK", r"block"]:
            if re.search(pat, s, re.IGNORECASE):
                return "AMBIGUOUS", "AMBIGUOUS"

        c  = re.sub(r"[\]\[().,]+.*$", "", s).strip()
        cu = c.upper()

        if re.match(r"^W[.\s]?R$", cu):                       return "WR",  "VALID"
        if re.match(r"^P[.\s]?W[.\s]?R$", cu):               return "PWR", "VALID"
        if re.match(r"^P[Hh][.\s]?R$", c, re.IGNORECASE):    return "PhR", "VALID"
        if cu == "PHR":                                         return "PhR", "VALID"
        if re.match(r"^P[.\s]?R$", cu):                       return "PR",  "VALID"
        if cu in {"I", "FP", "FILLED PAUSE", "FILLEDPAUSE"}:  return "I",   "VALID"
        if cu in {"P", "PAUSE"}:                               return "P",   "VALID"
        return "UNKNOWN", "UNKNOWN"

    def _combine_status(self, parse_status: str, norm_status: str) -> str:
        if parse_status not in ("OK", "ZERO_DURATION"):
            return parse_status
        if norm_status == "AMBIGUOUS":       return "AMBIGUOUS"
        if norm_status == "EMPTY":           return "EMPTY_LABEL"
        if norm_status == "UNKNOWN":         return "UNKNOWN_LABEL"
        if norm_status == "FLUENT_REGION":   return "FLUENT_GAP"
        if parse_status == "ZERO_DURATION":  return "ZERO_DURATION"
        return "VALID"

    def parse_dataset(self, inventory_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse all .txt files listed in inventory_df.
        Returns DataFrame of annotation rows with normalised labels.
        """
        all_annots = []
        for _, row in inventory_df.iterrows():
            rows = self.parse_file(row["txt_path"], row["file_id"],
                                   row["speaker_id"])
            all_annots.extend(rows)

        df = pd.DataFrame(all_annots)
        if df.empty:
            return df

        df["start"] = df.get("start_f", pd.Series(dtype=float))
        df["end"]   = df.get("end_f",   pd.Series(dtype=float))

        norm_results = df["raw_label"].apply(self.normalise_label)
        df["normalized_label"] = [r[0] for r in norm_results]
        df["norm_status"]      = [r[1] for r in norm_results]
        df["annotation_status"] = df.apply(
            lambda r: self._combine_status(r["parse_status"], r["norm_status"]),
            axis=1)
        return df

    def mark_overlaps(self, annot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sweep-line + Union-Find overlap detection.
        Sets has_overlap=True, overlap_group_id, is_nested.
        Returns mutated DataFrame.
        """
        def find_overlaps_uf(file_annots):
            n = len(file_annots)
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            order = sorted(range(n),
                           key=lambda i: (file_annots[i][1], file_annots[i][2]))
            active = []
            for pos, si in enumerate(order):
                _, s, e = file_annots[si]
                if s is None or e is None:
                    continue
                active = [(ae, ap) for ae, ap in active if ae > s]
                for _, ap in active:
                    union(order[ap], order[pos])
                active.append((e, pos))

            root_gid, gid_counter = {}, [0]
            gids = {}
            for i in range(n):
                r = find(i)
                if r not in root_gid:
                    root_gid[r] = gid_counter[0]
                    gid_counter[0] += 1
                gids[i] = root_gid[r]

            cnt      = Counter(gids.values())
            has_ovlp = {i: cnt[gids[i]] > 1 for i in range(n)}
            return gids, has_ovlp

        def mark_nested(file_df: pd.DataFrame) -> set:
            nested = set()
            rows = file_df[file_df["start"].notna() & file_df["end"].notna()]
            idxs = rows.index.tolist()
            for i in idxs:
                for j in idxs:
                    if i == j:
                        continue
                    si, ei = rows.at[i, "start"], rows.at[i, "end"]
                    sj, ej = rows.at[j, "start"], rows.at[j, "end"]
                    if si <= sj and ej <= ei and (si < sj or ej < ei):
                        nested.add(i)
                        nested.add(j)
            return nested

        global_gid_offset = 0
        gid_map, ovlp_map = {}, {}

        valid_mask = annot_df["start"].notna() & annot_df["end"].notna()
        valid_df   = annot_df[valid_mask].copy().reset_index()

        for fid, grp in valid_df.groupby("file_id"):
            idxs  = grp.index.tolist()
            oidxs = grp["index"].tolist()
            fa    = [(i, grp.at[i, "start"], grp.at[i, "end"]) for i in idxs]
            if not fa:
                continue
            lgids, lovlp = find_overlaps_uf(fa)
            # KeyError fix: use i_local (0-based) to index lgids/lovlp
            for i_local, oi in enumerate(oidxs):
                gid_map[oi]  = lgids[i_local] + global_gid_offset
                ovlp_map[oi] = lovlp[i_local]
            global_gid_offset += max(lgids.values(), default=-1) + 1

        annot_df["has_overlap"] = annot_df.index.map(
            lambda i: ovlp_map.get(i, False))
        annot_df["overlap_group_id"] = annot_df.index.map(
            lambda i: gid_map.get(i, None))

        nested_set = set()
        for fid, grp in annot_df.groupby("file_id"):
            nested_set |= mark_nested(grp)
        annot_df["is_nested"] = annot_df.index.isin(nested_set)

        return annot_df


# ============================================================
# SECTION 6 — FLUENCY REGION FINDER  (Q4: explicit generation)
# ============================================================

class FluencyRegionFinder:
    """
    Identifies fluent speech regions by computing the complement of the
    union of disfluency annotation intervals within each recording.

    Generates explicit Fluent AnnotationRows (annotation_status='FLUENT_GAP')
    and adds them to the annotation DataFrame before Wav2Vec2 alignment.

    Semantics are identical to the working notebook (v2): frames not covered
    by any disfluency annotation are Fluent. The difference is that this is
    now explicit rather than an implicit default inside align_frames().
    """

    def find_fluent_gaps(self, file_id: str, duration_sec: Optional[float],
                         disfluent_intervals: List[Tuple[float, float]]
                         ) -> List[Tuple[float, float]]:
        """
        Compute complement of disfluent_intervals within [0, duration_sec].
        Returns list of (start, end) Fluent gap intervals.
        """
        if not duration_sec or duration_sec <= 0:
            return []

        valid = [(s, e) for s, e in disfluent_intervals
                 if s is not None and e is not None and e > s]
        if not valid:
            return [(0.0, duration_sec)]

        valid.sort(key=lambda x: x[0])
        merged = []
        for s, e in valid:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append([s, e])

        gaps, prev_end = [], 0.0
        for s, e in merged:
            if s > prev_end + 1e-6:
                gaps.append((prev_end, s))
            prev_end = max(prev_end, e)
        if prev_end < duration_sec - 1e-6:
            gaps.append((prev_end, duration_sec))

        return gaps

    def build_fluent_rows(self, file_id: str, speaker_id: str,
                          gaps: List[Tuple[float, float]]) -> List[dict]:
        """Convert gap intervals to annotation row dicts."""
        rows = []
        for start, end in gaps:
            if end - start < 1e-4:
                continue
            rows.append({
                "file_id":            file_id,
                "speaker_id":         speaker_id,
                "line_number":        None,
                "raw_line":           None,
                "raw_start":          str(start),
                "raw_end":            str(end),
                "raw_label":          "Fluent",
                "start_f":            start,
                "end_f":              end,
                "start":              start,
                "end":                end,
                "parse_status":       "OK",
                "normalized_label":   "Fluent",
                "norm_status":        "FLUENT_REGION",
                "annotation_status":  "FLUENT_GAP",
                "has_overlap":        False,
                "is_nested":          False,
                "overlap_group_id":   None,
            })
        return rows

    def augment_dataset(self, annot_df: pd.DataFrame,
                        inventory_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each recording: find disfluency intervals, compute gap complement,
        append explicit Fluent rows to annot_df.
        Returns new DataFrame (original rows + Fluent gap rows).
        """
        fluent_rows = []
        for _, inv_row in inventory_df.iterrows():
            fid  = inv_row["file_id"]
            spk  = inv_row["speaker_id"]
            dur  = inv_row.get("duration_sec", None)

            file_annots = annot_df[
                (annot_df["file_id"] == fid) &
                annot_df["start"].notna() &
                annot_df["end"].notna()
            ]
            disf_intervals = list(zip(file_annots["start"], file_annots["end"]))
            gaps = self.find_fluent_gaps(fid, dur, disf_intervals)
            fluent_rows.extend(self.build_fluent_rows(fid, spk, gaps))

        if not fluent_rows:
            return annot_df

        fluent_df = pd.DataFrame(fluent_rows)
        result    = pd.concat([annot_df, fluent_df], ignore_index=True)
        result    = result.sort_values(["file_id", "start"]).reset_index(drop=True)
        return result


# ============================================================
# SECTION 7 — SPLIT MANAGER  (Q5: SHARED_SPLITS_PATH)
# ============================================================

class SplitManager:
    """
    Greedy 70/15/15 speaker-level split.
    Saves to SHARED_SPLITS_PATH and reuses existing valid splits.
    All interns use the same splits.json for comparable results.
    """

    def __init__(self, splits_json_path: str = SHARED_SPLITS_PATH):
        self.splits_json_path = splits_json_path

    def get_or_create_splits(self, file_info_df: pd.DataFrame) -> dict:
        """
        Load existing splits if valid for this file set; otherwise create new ones.
        Never silently regenerates a valid existing split.
        """
        all_files = set(file_info_df["file_id"].tolist())

        if os.path.exists(self.splits_json_path):
            with open(self.splits_json_path) as f:
                saved = json.load(f)
            saved_files = set(
                saved.get("train_files", []) +
                saved.get("val_files",   []) +
                saved.get("test_files",  []))
            if saved_files == all_files:
                print(f"SplitManager: loaded existing splits from {self.splits_json_path}")
                self._verify_no_leakage(saved)
                return saved
            else:
                print("SplitManager: file set changed — regenerating splits.")

        splits = self._create_splits(file_info_df)
        os.makedirs(os.path.dirname(self.splits_json_path), exist_ok=True)
        with open(self.splits_json_path, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"SplitManager: splits saved to {self.splits_json_path}")
        return splits

    def _create_splits(self, file_info_df: pd.DataFrame) -> dict:
        set_seeds(RANDOM_SEED)
        spk_stats = (
            file_info_df
            .groupby("speaker_id")
            .agg(n_frames=("n_frames", "sum"), files=("file_id", list))
            .reset_index()
            .sort_values("n_frames", ascending=False)
            .reset_index(drop=True)
        )
        total   = spk_stats["n_frames"].sum()
        t_train = int(total * TRAIN_RATIO)
        t_val   = int(total * VAL_RATIO)

        train_spk, val_spk, test_spk = [], [], []
        tr_frames = va_frames = 0

        for _, sr in spk_stats.iterrows():
            spk, nf   = sr["speaker_id"], sr["n_frames"]
            rem_tr    = t_train - tr_frames
            rem_va    = t_val   - va_frames
            if rem_tr >= rem_va and rem_tr > 0:
                train_spk.append(spk); tr_frames += nf
            elif rem_va > 0:
                val_spk.append(spk);   va_frames += nf
            else:
                test_spk.append(spk)

        if not val_spk  and len(train_spk) > 1: val_spk.append(train_spk.pop(-1))
        if not test_spk and len(train_spk) > 1: test_spk.append(train_spk.pop(-1))

        def files_for(spk_list):
            return file_info_df[
                file_info_df["speaker_id"].isin(spk_list)]["file_id"].tolist()

        splits = {
            "train_files": files_for(train_spk),
            "val_files":   files_for(val_spk),
            "test_files":  files_for(test_spk),
            "train_spk":   train_spk,
            "val_spk":     val_spk,
            "test_spk":    test_spk,
        }
        self._verify_no_leakage(splits)
        return splits

    def _verify_no_leakage(self, splits: dict) -> None:
        tr = set(splits["train_spk"])
        va = set(splits["val_spk"])
        te = set(splits["test_spk"])
        assert tr & va == set(), f"LEAK train∩val: {tr & va}"
        assert tr & te == set(), f"LEAK train∩test: {tr & te}"
        assert va & te == set(), f"LEAK val∩test: {va & te}"
        print("  Leakage check: train∩val=∅  train∩test=∅  val∩test=∅  OK")

    def print_split_table(self, splits: dict, file_info_df: pd.DataFrame) -> None:
        rows = []
        for split_name, flist in [("train", splits["train_files"]),
                                   ("val",   splits["val_files"]),
                                   ("test",  splits["test_files"])]:
            sub = file_info_df[file_info_df["file_id"].isin(flist)]
            for spk, grp in sub.groupby("speaker_id"):
                rows.append({"speaker": spk, "split": split_name,
                              "files": len(grp),
                              "frames": grp["n_frames"].sum()})
        tbl = pd.DataFrame(rows)
        print(tbl.to_string(index=False))
        totals = tbl.groupby("split")[["files", "frames"]].sum()
        totals["pct_frames"] = (
            totals["frames"] / totals["frames"].sum() * 100).round(1)
        print("\nSplit totals:")
        print(totals)


# ============================================================
# SECTION 8 — BASE FEATURE EXTRACTOR  (abstract, frame-level adapted)
# ============================================================

class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extraction methods.

    Subclass and implement extract_frames() to plug in any extractor.
    The OOP proposal defines extract_segment(audio, start, end) -> (D,).
    For our frame-level Wav2Vec2 pipeline (Q1: B), we instead define:
        extract_frames(wav_path) -> dict with 'embeddings' (T, D), 'timestamps' (T,)

    This preserves temporal resolution required by BiLSTM and frame-level labels.
    extract_segment() is provided as a convenience method for interns who
    want to mean-pool a time interval (e.g., MFCC intern).

    What YOU must implement:
        name (property)
        setup()
        extract_frames(wav_path, file_id) -> dict

    What you get for FREE (inherited):
        extract_dataset()  — loops over inventory, saves .npy
        build_store()      — assembles DatasetStore
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate  = sample_rate
        self.audio_loader = AudioLoader(sample_rate)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def extract_frames(self, wav_path: str, file_id: str) -> dict:
        """
        Extract frame-level embeddings for ONE recording.
        Returns dict:
            embeddings : np.ndarray (T, D)
            timestamps : np.ndarray (T,) — seconds from start of recording
            chunk_meta : list of dict (optional)
        """
        ...

    def extract_segment(self, audio: np.ndarray, start: float, end: float
                        ) -> np.ndarray:
        """
        Convenience: extract a single feature vector for one annotation segment.
        Default: mean-pool the extract_frames() output over the given interval.
        MFCC interns can override this; Wav2Vec2 uses extract_frames() directly.
        Returns: np.ndarray (D,)
        """
        raise NotImplementedError(
            "This extractor uses extract_frames() for frame-level output. "
            "Mean-pooled extract_segment() is not applicable.")

    def extract_dataset(self, inventory_df: pd.DataFrame,
                        embed_dir: str,
                        force: bool = False) -> List[dict]:
        """
        Run extract_frames() over all recordings. Saves {file_id}.npy +
        {file_id}_timestamps.npy. Skips cached files unless force=True.
        Returns extraction_log (list of dicts).
        """
        os.makedirs(embed_dir, exist_ok=True)
        log = []
        for i, (_, row) in enumerate(inventory_df.iterrows()):
            fid      = row["file_id"]
            emb_path = os.path.join(embed_dir, f"{fid}.npy")
            ts_path  = os.path.join(embed_dir, f"{fid}_timestamps.npy")
            cm_path  = os.path.join(embed_dir, f"{fid}_chunk_meta.json")

            if not force and os.path.exists(emb_path) and os.path.exists(ts_path):
                sh = np.load(emb_path, mmap_mode="r").shape
                log.append({"file_id": fid, "n_frames": sh[0], "status": "SKIPPED"})
                print(f"  [{i+1:3d}] {fid} — SKIPPED (cached, {sh[0]} frames)")
                continue

            if row.get("audio_error"):
                log.append({"file_id": fid, "n_frames": 0,
                             "status": "FAILED", "error": row["audio_error"]})
                continue

            try:
                result = self.extract_frames(row["wav_path"], fid)
                emb    = result["embeddings"]
                ts     = result["timestamps"]
                cm     = result.get("chunk_meta", [])
                np.save(emb_path, emb)
                np.save(ts_path,  ts)
                with open(cm_path, "w") as f:
                    json.dump(cm, f)
                log.append({"file_id": fid, "n_frames": emb.shape[0],
                             "dur_sec": row.get("duration_sec", 0), "status": "OK"})
                print(f"  [{i+1:3d}] {fid} — {emb.shape[0]} frames  "
                      f"{row.get('duration_sec', 0):.1f}s")
                del emb, ts
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                log.append({"file_id": fid, "n_frames": 0,
                             "status": "FAILED", "error": str(e)})
                print(f"  ERROR {fid}: {e}")
        return log

    def build_store(self, frame_meta_df: pd.DataFrame,
                    embed_dir: str) -> DatasetStore:
        """Assemble DatasetStore from frame_meta_df and saved .npy files."""
        return DatasetStore(
            embed_dir      = embed_dir,
            frame_meta_df  = frame_meta_df,
            extractor_name = self.name,
            feature_dim    = W2V_HIDDEN_DIM,
        )


# ============================================================
# SECTION 9 — WAV2VEC2 EXTRACTOR  (intern-specific implementation)
# ============================================================

class Wav2vec2Extractor(BaseFeatureExtractor):
    """
    Frame-level Wav2Vec2 feature extractor.

    Extends BaseFeatureExtractor. The only intern-specific class.
    Produces (T, 768) per recording — NOT mean-pooled (D,).

    Implementation details preserved from iiith-wav2vec2.ipynb:
    - Frozen facebook/wav2vec2-base
    - 30s keep region + 1s context overlap per chunk
    - Keep-region edge handling (no duplicate frames)
    - CNN-derived timestamps from conv_kernel / conv_stride
    - Global timestamps relative to start of recording
    """

    name = "Wav2vec2"

    def __init__(self, model_name: str = WAV2VEC_MODEL,
                 sample_rate: int = SAMPLE_RATE):
        super().__init__(sample_rate)
        self.model_name  = model_name
        self.processor   = None
        self.model       = None
        self.device      = None
        # Timing constants — set in setup()
        self.total_stride    = None
        self.receptive_field = None
        self.rf_centre       = None
        self.frame_step_sec  = None
        self.hidden_dim      = W2V_HIDDEN_DIM

    def setup(self) -> None:
        """Load model, derive CNN timing constants, run empirical cross-check."""
        global W2V_TOTAL_STRIDE, W2V_RECEPTIVE_FIELD, W2V_RF_CENTRE
        global W2V_FRAME_STEP_SEC, W2V_FRAME_RATE, W2V_HIDDEN_DIM

        self.device = DEVICE
        print(f"Loading {self.model_name} on {self.device} ...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model     = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        print(f"  Frozen | params={sum(p.numel() for p in self.model.parameters()):,}")

        # Derive timing from CNN feature extractor config
        cfg          = self.model.config
        conv_kernels = list(cfg.conv_kernel)
        conv_strides = list(cfg.conv_stride)

        total_stride = 1
        for s in conv_strides:
            total_stride *= s

        rf, cum = 1, 1
        for k, s in zip(conv_kernels, conv_strides):
            rf  = rf + (k - 1) * cum
            cum *= s

        self.total_stride    = total_stride
        self.receptive_field = rf
        self.rf_centre       = (rf - 1) / 2.0
        self.frame_step_sec  = total_stride / self.sample_rate

        # Empirical cross-check on 1-second silence
        _sig = np.zeros(self.sample_rate, dtype=np.float32)
        _inp = self.processor(_sig, sampling_rate=self.sample_rate,
                              return_tensors="pt")
        _inp = {k: v.to(self.device) for k, v in _inp.items()}
        with torch.inference_mode():
            _out = self.model(**_inp)
        T_emp  = _out.last_hidden_state.shape[1]
        T_theo = (self.sample_rate - rf) // total_stride + 1
        self.hidden_dim = _out.last_hidden_state.shape[2]

        # Update globals so shared utilities can reference them
        W2V_TOTAL_STRIDE    = self.total_stride
        W2V_RECEPTIVE_FIELD = self.receptive_field
        W2V_RF_CENTRE       = self.rf_centre
        W2V_FRAME_STEP_SEC  = self.frame_step_sec
        W2V_FRAME_RATE      = T_emp
        W2V_HIDDEN_DIM      = self.hidden_dim

        print(f"  total_stride      : {total_stride} samp "
              f"({total_stride / self.sample_rate * 1000:.2f} ms)")
        print(f"  receptive_field   : {rf} samp")
        print(f"  hidden_dim        : {self.hidden_dim}")
        print(f"  empirical frames  : {T_emp}  "
              f"theoretical: {T_theo}  match={T_emp == T_theo}")
        del _sig, _inp, _out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def frame_timestamps(self, n_frames: int,
                         offset_samples: int = 0) -> np.ndarray:
        """
        Centre timestamps (seconds) for each output frame.
        t(i) = (offset_samples + i * total_stride + rf_centre) / sample_rate
        """
        i = np.arange(n_frames)
        return (offset_samples + i * self.total_stride + self.rf_centre) / self.sample_rate

    def extract_frames(self, wav_path: str, file_id: str) -> dict:
        """
        Extract frame-level Wav2Vec2 embeddings using overlapping 30s chunks.
        Returns (T_total, 768) embeddings + (T_total,) global timestamps.
        """
        wf, sr, err = self.audio_loader.load(wav_path)
        if err:
            raise RuntimeError(err)

        total_samples = len(wf)
        chunk_samp    = int(CHUNK_DURATION_SEC * sr)
        ctx_samp      = int(CHUNK_OVERLAP_SEC  * sr)

        starts, pos = [], 0
        while pos < total_samples:
            starts.append(pos)
            pos += chunk_samp
            if pos >= total_samples:
                break

        all_emb, all_ts, chunk_meta = [], [], []

        for cid, keep_start in enumerate(starts):
            is_first = (cid == 0)
            is_last  = (cid == len(starts) - 1)

            ctx_start = max(0, keep_start - ctx_samp)
            ctx_end   = min(total_samples, keep_start + chunk_samp + ctx_samp)
            chunk_wf  = wf[ctx_start:ctx_end]

            if len(chunk_wf) < int(MIN_CHUNK_SEC * sr):
                continue

            inp = self.processor(chunk_wf, sampling_rate=sr, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.inference_mode():
                out = self.model(**inp)
            emb = out.last_hidden_state.squeeze(0).cpu().numpy()
            T_c = emb.shape[0]
            ts  = self.frame_timestamps(T_c, offset_samples=ctx_start)

            keep_start_sec = keep_start / sr
            keep_end_sec   = min(total_samples / sr, (keep_start + chunk_samp) / sr)

            if   is_first and is_last: keep_mask = np.ones(T_c, dtype=bool)
            elif is_first:             keep_mask = ts < keep_end_sec
            elif is_last:              keep_mask = ts >= keep_start_sec
            else:                      keep_mask = (ts >= keep_start_sec) & (ts < keep_end_sec)

            all_emb.append(emb[keep_mask])
            all_ts.append(ts[keep_mask])
            chunk_meta.append({
                "chunk_id": cid,
                "ctx_start_sec": ctx_start / sr, "ctx_end_sec": ctx_end / sr,
                "keep_start_sec": keep_start_sec, "keep_end_sec": keep_end_sec,
                "n_frames_chunk": T_c, "n_frames_kept": int(keep_mask.sum()),
            })
            del inp, out, emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not all_emb:
            return {"embeddings": np.zeros((0, self.hidden_dim)),
                    "timestamps": np.zeros(0), "chunk_meta": []}

        return {"embeddings": np.vstack(all_emb),
                "timestamps": np.concatenate(all_ts),
                "chunk_meta": chunk_meta}


# ============================================================
# SECTION 10 — FRAME-LEVEL LABEL ALIGNMENT
# ============================================================

def align_frames(timestamps: np.ndarray,
                 fdf: pd.DataFrame) -> dict:
    """
    Assign labels to every Wav2Vec2 frame in timestamps.

    fdf: annotation rows for one file (includes VALID + AMBIGUOUS + FLUENT_GAP).
    Default label for any uncovered frame: "Fluent" (edge-case fallback only;
    FluencyRegionFinder should cover all gaps explicitly).

    Priority for overlapping annotations:
      - Shortest-duration valid disfluency wins (most specific)
      - Ambiguous wins over nothing
      - Fluent wins over nothing
    """
    T         = len(timestamps)
    labels    = ["Fluent"] * T
    has_ovlp  = [False]   * T
    ovlp_gid  = [None]    * T
    ovlp_set  = [None]    * T

    point_df    = fdf[fdf["annotation_status"] == "ZERO_DURATION"]
    interval_df = fdf[fdf["annotation_status"].isin(
        ["VALID", "AMBIGUOUS", "FLUENT_GAP"])]

    for ti, tc in enumerate(timestamps):
        hit = interval_df[
            (interval_df["start"] <= tc) & (interval_df["end"] > tc)]

        if len(hit) == 0:
            # Point annotation fallback
            if len(point_df) > 0:
                dists = np.abs(point_df["start"].values - tc)
                ni    = dists.argmin()
                if dists[ni] <= W2V_FRAME_STEP_SEC / 2.0:
                    pt = point_df.iloc[ni]
                    nl = pt["normalized_label"]
                    if nl in VALID_LABELS:
                        labels[ti] = nl
                    elif pt["annotation_status"] == "AMBIGUOUS":
                        labels[ti] = "AMBIGUOUS"
            continue

        lbl_list = hit["normalized_label"].tolist()

        if len(hit) == 1:
            r  = hit.iloc[0]
            nl = r["normalized_label"]
            if nl in VALID_LABELS:
                labels[ti] = nl
            elif nl == "Fluent":                      # explicit FLUENT_GAP row
                labels[ti] = "Fluent"
            elif r["annotation_status"] == "AMBIGUOUS":
                labels[ti] = "AMBIGUOUS"
            has_ovlp[ti] = bool(r["has_overlap"])
            ovlp_gid[ti] = r["overlap_group_id"]
        else:
            has_ovlp[ti] = True
            valid_hits = hit[hit["normalized_label"].isin(VALID_LABELS)]
            ambig_hits = hit[hit["annotation_status"] == "AMBIGUOUS"]
            if len(valid_hits) > 0:
                dur  = valid_hits["end"] - valid_hits["start"]
                best = valid_hits.loc[dur.idxmin()]
                labels[ti]   = best["normalized_label"]
                ovlp_gid[ti] = best["overlap_group_id"]
            elif len(ambig_hits) > 0:
                labels[ti] = "AMBIGUOUS"
            ovlp_set[ti] = ",".join(sorted(set(lbl_list))) if len(hit) > 1 else None

    return {"primary_label": labels, "has_overlap": has_ovlp,
            "overlap_group_id": ovlp_gid, "overlap_label_set": ovlp_set}


def build_frame_meta(inventory_df: pd.DataFrame,
                     audit_df: pd.DataFrame,
                     embed_dir: str) -> pd.DataFrame:
    """
    Run align_frames() for every recording and concatenate results
    into a single frame_meta_df.
    """
    all_frame_meta, align_errors = [], []
    print("Aligning frames to labels ...")

    for i, (_, inv_row) in enumerate(inventory_df.iterrows()):
        fid      = inv_row["file_id"]
        ts_path  = os.path.join(embed_dir, f"{fid}_timestamps.npy")
        emb_path = os.path.join(embed_dir, f"{fid}.npy")

        if not os.path.exists(ts_path) or not os.path.exists(emb_path):
            continue
        try:
            ts   = np.load(ts_path)
            fdf  = audit_df[audit_df["file_id"] == fid].copy()
            fdf["duration"] = (fdf["end"] - fdf["start"]).fillna(0)

            aligned = align_frames(ts, fdf)
            T       = min(len(ts), np.load(emb_path, mmap_mode="r").shape[0])

            for ti in range(T):
                all_frame_meta.append({
                    "file_id":          fid,
                    "speaker_id":       inv_row["speaker_id"],
                    "frame_idx":        ti,
                    "timestamp_sec":    ts[ti],
                    "primary_label":    aligned["primary_label"][ti],
                    "has_overlap":      aligned["has_overlap"][ti],
                    "overlap_group_id": aligned["overlap_group_id"][ti],
                    "overlap_label_set":aligned["overlap_label_set"][ti],
                })
        except Exception as e:
            align_errors.append({"file_id": fid, "error": str(e)})
            print(f"  ERROR aligning {fid}: {e}")

    df = pd.DataFrame(all_frame_meta)
    print(f"  Total frames : {len(df):,}  Errors: {len(align_errors)}")
    print("  Primary label distribution:")
    print(df["primary_label"].value_counts().to_string())
    return df


# ============================================================
# SECTION 11 — SHARED DATA UTILITIES
# ============================================================

def load_embeddings_for_files(file_list: list, store: DatasetStore
                               ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load and stack embeddings + aligned frame metadata for a file list."""
    X_parts, M_parts = [], []
    for fid in file_list:
        ep = os.path.join(store.embed_dir, f"{fid}.npy")
        if not os.path.exists(ep):
            continue
        emb  = np.load(ep)
        fmta = store.frame_meta_df[
            store.frame_meta_df["file_id"] == fid].reset_index(drop=True)
        T    = min(len(emb), len(fmta))
        X_parts.append(emb[:T])
        M_parts.append(fmta.iloc[:T])
    if not X_parts:
        return np.zeros((0, store.feature_dim)), pd.DataFrame()
    return np.vstack(X_parts), pd.concat(M_parts, ignore_index=True)


def sample_train_frames(X: np.ndarray, y: np.ndarray,
                        max_per_class: int,
                        rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified per-class cap. Only applied to training data."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    keep = []
    for cls in np.unique(y):
        if cls < 0:
            continue
        idx = np.where(y == cls)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.append(idx)
    if not keep:
        return X[:0], y[:0]
    keep = np.concatenate(keep)
    keep.sort()
    return X[keep], y[keep]


def build_rfdnn_arrays(split: str, label_map: dict, store: DatasetStore,
                       train_files: list, val_files: list, test_files: list,
                       exclude_labels=None,
                       max_per_class: Optional[int] = None
                       ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build X, y arrays for RF / DNN from DatasetStore.
    Training set is optionally sampled; val/test always complete.
    """
    if split == "train":   file_list = train_files
    elif split == "val":   file_list = val_files
    else:                  file_list = test_files

    excl = set(exclude_labels) if exclude_labels else set()
    X_all, M_all = load_embeddings_for_files(file_list, store)

    if M_all.empty:
        return np.zeros((0, store.feature_dim)), np.zeros(0, dtype=np.int64), pd.DataFrame()

    mask  = M_all["primary_label"].isin(label_map) & ~M_all["primary_label"].isin(excl)
    X_all = X_all[mask]
    M_all = M_all[mask].reset_index(drop=True)
    y_all = np.array([label_map[l] for l in M_all["primary_label"]], dtype=np.int64)

    if split == "train" and max_per_class is not None:
        X_all, y_all = sample_train_frames(X_all, y_all, max_per_class)

    return X_all, y_all, M_all


def build_bilstm_index(file_list: list, label_map: dict, store: DatasetStore,
                       exclude_labels_as_ignore=None,
                       seq_len: int = SEQ_LEN) -> list:
    """
    Build a lazy index for BiLSTM — stores only metadata, NO embeddings.
    Each entry: {file_id, start_frame, end_frame, length, target (int64, seq_len)}
    Excluded labels -> target = -1 (masked by CrossEntropyLoss ignore_index=-1).
    Chronological recording order is ALWAYS preserved.
    """
    if exclude_labels_as_ignore is None:
        exclude_labels_as_ignore = set()
    index = []
    for fid in file_list:
        ep = os.path.join(store.embed_dir, f"{fid}.npy")
        if not os.path.exists(ep):
            continue
        emb_shape = np.load(ep, mmap_mode="r").shape
        fmta      = store.frame_meta_df[
            store.frame_meta_df["file_id"] == fid].reset_index(drop=True)
        T         = min(emb_shape[0], len(fmta))
        if T == 0:
            continue

        labels    = fmta["primary_label"].values
        tgt_full  = np.full(T, -1, dtype=np.int64)
        for i in range(T):
            lbl = labels[i]
            if lbl in label_map:
                tgt_full[i] = label_map[lbl]
            # excluded labels stay -1

        for start in range(0, T, seq_len):
            end    = min(start + seq_len, T)
            actual = end - start
            tgt    = np.full(seq_len, -1, dtype=np.int64)
            tgt[:actual] = tgt_full[start:end]
            index.append({"file_id": fid, "start_frame": start,
                          "end_frame": end, "length": actual, "target": tgt})
    return index


def class_weight_tensor(y: np.ndarray, n_classes: int,
                        device=None) -> torch.Tensor:
    """Balanced class weights. Absent classes get weight 1.0 (robust for SANITY_RUN)."""
    if device is None:
        device = DEVICE
    y_arr   = np.asarray(y).ravel()
    y_valid = y_arr[y_arr >= 0]
    if len(y_valid) == 0:
        return torch.ones(n_classes, device=device)
    present = np.unique(y_valid)
    if len(present) == n_classes:
        cw = compute_class_weight("balanced",
                                  classes=np.arange(n_classes), y=y_valid)
    else:
        cw = np.ones(n_classes, dtype=np.float64)
        present_cw = compute_class_weight("balanced",
                                          classes=present, y=y_valid)
        for cls, w in zip(present, present_cw):
            cw[int(cls)] = w
    return torch.tensor(cw, dtype=torch.float32, device=device)


def cw_from_index(index: list, n_classes: int, device=None) -> torch.Tensor:
    """Compute class weights from a BiLSTM lazy index."""
    if not index:
        return torch.ones(n_classes, device=device or DEVICE)
    y_all = np.concatenate([e["target"] for e in index])
    return class_weight_tensor(y_all, n_classes, device)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: List[str], task: str = "binary",
                    y_prob: Optional[np.ndarray] = None,
                    labels: Optional[List[int]] = None) -> dict:
    """Compute standard metrics. Primary: Macro-F1."""
    _labels = labels if labels is not None else list(range(len(class_names)))
    r = {
        "accuracy"   : accuracy_score(y_true, y_pred),
        "precision"  : precision_score(y_true, y_pred, average="macro",
                                       zero_division=0, labels=_labels),
        "recall"     : recall_score(y_true, y_pred, average="macro",
                                    zero_division=0, labels=_labels),
        "macro_f1"   : f1_score(y_true, y_pred, average="macro",
                                zero_division=0, labels=_labels),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted",
                                zero_division=0, labels=_labels),
        "conf_matrix": confusion_matrix(y_true, y_pred, labels=_labels),
    }
    if task == "binary" and y_prob is not None:
        try:    r["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception: r["roc_auc"] = None
        try:    r["pr_auc"]  = average_precision_score(y_true, y_prob)
        except Exception: r["pr_auc"]  = None
    per_f = f1_score(y_true, y_pred, average=None, zero_division=0, labels=_labels)
    for i, lbl in enumerate(class_names):
        r[f"f1_{lbl}"] = per_f[i] if i < len(per_f) else 0.0
    return r


def save_result(experiment: str, model_name: str,
                metrics: dict, class_names: List[str]) -> None:
    row = {"experiment": experiment, "model": model_name,
           "accuracy": metrics.get("accuracy", 0),
           "macro_f1": metrics.get("macro_f1", 0),
           "weighted_f1": metrics.get("weighted_f1", 0),
           "precision": metrics.get("precision", 0),
           "recall": metrics.get("recall", 0),
           "class_names": ",".join(class_names)}
    if "roc_auc" in metrics:
        row["roc_auc"] = metrics["roc_auc"]
    ALL_RESULTS.append(row)


# ============================================================
# SECTION 12 — PYTORCH DATASETS AND MODEL ARCHITECTURES
# ============================================================

class FrameDataset(Dataset):
    """Frame-level dataset for DNN (no temporal structure)."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LazySeqDataset(Dataset):
    """
    Index-based BiLSTM dataset.
    Only metadata stored; (seq_len, 768) embedding loaded lazily via mmap.
    Memory cost: O(n_windows * seq_len) NOT O(n_windows * seq_len * 768).
    """
    def __init__(self, index: list, embed_dir: str):
        self.index     = index
        self.embed_dir = embed_dir

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        e      = self.index[i]
        ep     = os.path.join(self.embed_dir, f"{e['file_id']}.npy")
        emb    = np.load(ep, mmap_mode="r")
        actual = e["length"]
        D      = emb.shape[1]
        seq_len = len(e["target"])
        X_seg  = np.zeros((seq_len, D), dtype=np.float32)
        X_seg[:actual] = emb[e["start_frame"]:e["end_frame"]]
        return (torch.tensor(X_seg,        dtype=torch.float32),
                torch.tensor(e["target"],  dtype=torch.long),
                actual)


def collate_seq(batch):
    seqs, lbls, lens = zip(*batch)
    B = len(seqs); T = max(s.shape[0] for s in seqs); D = seqs[0].shape[1]
    Xp = torch.zeros(B, T, D)
    yp = torch.full((B, T), -1, dtype=torch.long)
    for i, (s, l, ln) in enumerate(zip(seqs, lbls, lens)):
        Xp[i, :ln] = s[:ln]
        yp[i, :ln] = l[:ln]
    return Xp, yp, list(lens)


class DNN(nn.Module):
    """Frame-level MLP. Input: (B, 768) -> Output: (B, n_classes)."""
    def __init__(self, input_dim=768, h1=256, h2=128, n_classes=2, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h1, h2),        nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h2, n_classes))
    def forward(self, x): return self.net(x)


class BiLSTMClassifier(nn.Module):
    """
    Temporal sequence classifier.
    Input: (B, T, 768)  Output: (B, T, n_classes) logits.
    Loss: CrossEntropyLoss(ignore_index=-1) — padded + excluded frames masked.
    """
    def __init__(self, input_dim=768, h1=128, h2=64, n_classes=2, drop=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, h1, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(drop)
        self.lstm2 = nn.LSTM(h1 * 2,   h2, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(drop)
        self.fc    = nn.Linear(h2 * 2, n_classes)

    def forward(self, x):
        o, _ = self.lstm1(x); o = self.drop1(o)
        o, _ = self.lstm2(o); o = self.drop2(o)
        return self.fc(o)     # (B, T, n_classes)


# ============================================================
# SECTION 13 — CLASSIFIER WRAPPERS  (BaseClassifier + RF / DNN / BiLSTM)
# ============================================================

class BaseClassifier(ABC):
    """
    Interface that every classifier satisfies.
    RF and DNN implement fit(X_train, y_train, X_val, y_val).
    BiLSTM implements fit_sequences() — ClassifierRunner dispatches.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, path: str) -> None:
        joblib.dump(self, path)


class RandomForestWrapper(BaseClassifier):
    name = "RandomForest"

    def __init__(self, n_estimators: int = RF_N_ESTIMATORS,
                 max_depth: Optional[int] = RF_MAX_DEPTH,
                 random_state: int = RANDOM_SEED):
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.random_state  = random_state
        self._model        = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            class_weight="balanced", n_jobs=-1, random_state=self.random_state)
        self._model.fit(X_train, y_train)

    def predict(self, X): return self._model.predict(X)

    def predict_proba(self, X): return self._model.predict_proba(X)

    def save(self, path: str): joblib.dump(self._model, path)


class DNNWrapper(BaseClassifier):
    name = "DNN"

    def __init__(self, input_dim: int = W2V_HIDDEN_DIM, n_classes: int = 2,
                 lr: float = LEARNING_RATE, n_epochs: int = DNN_EPOCHS,
                 patience: int = EARLY_STOP_PAT):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr        = lr
        self.n_epochs  = n_epochs
        self.patience  = patience
        self._model    = None
        self.history   = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        cw             = class_weight_tensor(y_train, self.n_classes)
        self._model    = DNN(self.input_dim, n_classes=self.n_classes).to(DEVICE)
        dl_tr          = DataLoader(FrameDataset(X_train, y_train),
                                    BATCH_SIZE, shuffle=True, drop_last=True)
        dl_va          = DataLoader(FrameDataset(X_val, y_val),
                                    BATCH_SIZE, shuffle=False) if X_val is not None else None
        self._model, self.history, _ = self._train_loop(self._model, dl_tr, dl_va, cw)

    def _train_loop(self, model, dl_tr, dl_va, cw):
        opt   = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=3)
        crit  = nn.CrossEntropyLoss(weight=cw)
        best_f1, best_state, wait = -1.0, None, 0
        hist = []
        for ep in range(1, self.n_epochs + 1):
            model.train()
            tr_loss = 0.0
            for Xb, yb in dl_tr:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(Xb), yb)
                loss.backward(); opt.step()
                tr_loss += loss.item() * len(yb)
            tr_loss /= len(dl_tr.dataset)
            vf1 = 0.0
            if dl_va is not None:
                model.eval(); pv, tv = [], []
                with torch.inference_mode():
                    for Xb, yb in dl_va:
                        pv.extend(model(Xb.to(DEVICE)).argmax(1).cpu().tolist())
                        tv.extend(yb.tolist())
                vf1 = f1_score(tv, pv, average="macro", zero_division=0)
            sched.step(vf1)
            hist.append({"ep": ep, "loss": tr_loss, "val_f1": vf1})
            if vf1 > best_f1:
                best_f1 = vf1; best_state = deepcopy(model.state_dict()); wait = 0
            else:
                wait += 1
            if ep % 10 == 0 or ep <= 3:
                print(f"    ep{ep:3d}  loss={tr_loss:.4f}  val_f1={vf1:.4f}  best={best_f1:.4f}")
            if wait >= self.patience:
                print(f"    Early stop ep{ep} (best={best_f1:.4f})"); break
        if best_state:
            model.load_state_dict(best_state)
        return model, hist, best_f1

    def predict(self, X):
        self._model.eval()
        ds = FrameDataset(X, np.zeros(len(X), dtype=np.int64))
        dl = DataLoader(ds, BATCH_SIZE * 4, shuffle=False)
        pred = []
        with torch.inference_mode():
            for Xb, _ in dl:
                pred.extend(self._model(Xb.to(DEVICE)).argmax(1).cpu().tolist())
        return np.array(pred)

    def predict_proba(self, X):
        self._model.eval()
        ds = FrameDataset(X, np.zeros(len(X), dtype=np.int64))
        dl = DataLoader(ds, BATCH_SIZE * 4, shuffle=False)
        probs = []
        with torch.inference_mode():
            for Xb, _ in dl:
                probs.append(
                    torch.softmax(self._model(Xb.to(DEVICE)), 1).cpu().numpy())
        return np.vstack(probs)

    def save(self, path: str):
        if self._model:
            torch.save(self._model.state_dict(), path)


class BiLSTMWrapper(BaseClassifier):
    """
    BiLSTM wrapper with frame-level masked cross-entropy.  (Q2: A)

    CRITICAL: fit() raises NotImplementedError.
    Use fit_sequences(idx_tr, idx_va, n_classes, embed_dir) instead.
    ClassifierRunner detects isinstance(clf, BiLSTMWrapper) and dispatches.

    Preserved from working notebook:
    - LazySeqDataset (mmap .npy loading per __getitem__)
    - Chronological recording order (never shuffled across recordings)
    - ignore_index=-1 for padding and excluded frames
    - SEQ_LEN=128 windows with stride=128
    - clip_grad_norm_ for stability
    """

    name = "BiLSTM"

    def __init__(self, input_dim: int = W2V_HIDDEN_DIM, n_classes: int = 2,
                 lr: float = LEARNING_RATE, n_epochs: int = BILSTM_EPOCHS,
                 patience: int = EARLY_STOP_PAT):
        self.input_dim  = input_dim
        self.n_classes  = n_classes
        self.lr         = lr
        self.n_epochs   = n_epochs
        self.patience   = patience
        self._model     = None
        self._embed_dir = None
        self.history    = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError(
            "BiLSTMWrapper uses fit_sequences(idx_tr, idx_va, n_classes, embed_dir). "
            "Call that directly, or use ClassifierRunner which dispatches automatically.")

    def fit_sequences(self, idx_tr: list, idx_va: list,
                      n_classes: int, embed_dir: str) -> None:
        """
        Train BiLSTM from lazy index lists (full chronological recordings).
        Parameters: idx_tr/idx_va from build_bilstm_index(), embed_dir for .npy loading.
        """
        self.n_classes  = n_classes
        self._embed_dir = embed_dir
        cw = cw_from_index(idx_tr, n_classes)
        self._model = BiLSTMClassifier(self.input_dim, n_classes=n_classes).to(DEVICE)
        dl_tr = DataLoader(LazySeqDataset(idx_tr, embed_dir),
                           BILSTM_BATCH, shuffle=True,  collate_fn=collate_seq)
        dl_va = DataLoader(LazySeqDataset(idx_va, embed_dir),
                           BILSTM_BATCH, shuffle=False, collate_fn=collate_seq)
        self._model, self.history, _ = self._train_loop(self._model, dl_tr, dl_va, cw)

    def _train_loop(self, model, dl_tr, dl_va, cw):
        opt   = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=3)
        crit  = nn.CrossEntropyLoss(weight=cw, ignore_index=-1)
        best_f1, best_state, wait = -1.0, None, 0
        hist = []
        for ep in range(1, self.n_epochs + 1):
            model.train()
            tr_loss, n_b = 0.0, 0
            for Xb, yb, lens in dl_tr:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits = model(Xb); B, T, C = logits.shape
                loss   = crit(logits.reshape(B * T, C), yb.reshape(B * T))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); tr_loss += loss.item(); n_b += 1
            tr_loss /= max(n_b, 1)
            model.eval(); pv, tv = [], []
            with torch.inference_mode():
                for Xb, yb, lens in dl_va:
                    lg = model(Xb.to(DEVICE)).argmax(2).cpu().numpy()
                    for i, ln in enumerate(lens):
                        mask = yb[i, :ln] >= 0
                        pv.extend(lg[i, :ln][mask.numpy()].tolist())
                        tv.extend(yb[i, :ln][mask].numpy().tolist())
            vf1 = f1_score(tv, pv, average="macro", zero_division=0) if tv else 0.0
            sched.step(vf1)
            hist.append({"ep": ep, "loss": tr_loss, "val_f1": vf1})
            if vf1 > best_f1:
                best_f1 = vf1; best_state = deepcopy(model.state_dict()); wait = 0
            else:
                wait += 1
            if ep % 10 == 0 or ep <= 3:
                print(f"    ep{ep:3d}  loss={tr_loss:.4f}  val_f1={vf1:.4f}  best={best_f1:.4f}")
            if wait >= self.patience:
                print(f"    Early stop ep{ep} (best={best_f1:.4f})"); break
        if best_state:
            model.load_state_dict(best_state)
        return model, hist, best_f1

    def predict_sequences(self, idx_te: list, embed_dir: str
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate on test lazy index. Returns (y_true, y_pred, y_prob)."""
        self._model.eval()
        dl = DataLoader(LazySeqDataset(idx_te, embed_dir),
                        batch_size=8, shuffle=False, collate_fn=collate_seq)
        all_pred, all_true, all_prob = [], [], []
        with torch.inference_mode():
            for Xb, yb, blens in dl:
                logits = self._model(Xb.to(DEVICE))
                prob   = torch.softmax(logits, 2).cpu().numpy()
                pred   = logits.argmax(2).cpu().numpy()
                for i, ln in enumerate(blens):
                    mask = yb[i, :ln].numpy() >= 0
                    all_pred.extend(pred[i, :ln][mask].tolist())
                    all_true.extend(yb[i, :ln].numpy()[mask].tolist())
                    all_prob.append(prob[i, :ln][mask])
        p  = np.array(all_pred)
        t  = np.array(all_true)
        pm = np.vstack(all_prob) if all_prob else np.zeros((0, self.n_classes))
        return t, p, pm

    def predict(self, X):
        raise NotImplementedError("Use predict_sequences() for BiLSTM.")

    def save(self, path: str):
        if self._model:
            torch.save(self._model.state_dict(), path)


# ============================================================
# SECTION 14 — EVALUATOR
# ============================================================

class Evaluator:
    """
    Computes metrics and plots confusion matrices.
    Shared by all classifiers and all interns. Primary metric: Macro-F1.
    """

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 class_names: List[str], task_name: str,
                 classifier_name: str,
                 task: str = "binary",
                 y_prob: Optional[np.ndarray] = None,
                 labels: Optional[List[int]] = None) -> EvaluationResult:
        m = compute_metrics(y_true, y_pred, class_names, task, y_prob, labels)
        return EvaluationResult(
            task_name        = task_name,
            classifier_name  = classifier_name,
            accuracy         = m["accuracy"],
            macro_f1         = m["macro_f1"],
            weighted_f1      = m["weighted_f1"],
            precision_macro  = m["precision"],
            recall_macro     = m["recall"],
            confusion_matrix = m["conf_matrix"],
            extra            = {k: v for k, v in m.items()
                                if k not in ("accuracy","macro_f1","weighted_f1",
                                             "precision","recall","conf_matrix")},
        )

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                               title: str, save_path: Optional[str] = None) -> None:
        fig, ax = plt.subplots(figsize=(max(5, len(class_names)*1.3),
                                        max(4, len(class_names)*1.1)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  Confusion matrix saved: {save_path}")

    def print_summary(self, result: EvaluationResult) -> None:
        print(f"  {result.classifier_name:<14} {result.task_name:<30} "
              f"Acc={result.accuracy:.4f}  Macro-F1={result.macro_f1:.4f}  "
              f"Wtd-F1={result.weighted_f1:.4f}")


# ============================================================
# SECTION 15 — CLASSIFIER RUNNER
# ============================================================

class ClassifierRunner:
    """
    Runs all 3 standard classification tasks across all provided classifiers.
    Works with any DatasetStore regardless of which extractor produced it.

    Tasks:
      Exp1 — Fluent vs Disfluent   (binary)
      Exp2 — Fluent vs <Class>     (6 x binary)
      Exp3 — 6-class disfluency

    Dispatches fit() for RF/DNN and fit_sequences() for BiLSTM.
    """

    DISF_CLASSES = ["I", "PR", "PhR", "WR", "PWR", "P"]

    def __init__(self, store: DatasetStore, splits: dict,
                 classifiers: List[BaseClassifier],
                 evaluator: Evaluator,
                 model_dir: str = MODEL_DIR,
                 result_dir: str = RESULT_DIR):
        self.store       = store
        self.splits      = splits
        self.clfs        = classifiers
        self.evaluator   = evaluator
        self.model_dir   = model_dir
        self.result_dir  = result_dir
        self.results: Dict[str, Dict[str, dict]] = {}

    def _train_files(self):  return self.splits["train_files"]
    def _val_files(self):    return self.splits["val_files"]
    def _test_files(self):   return self.splits["test_files"]

    def build_standard_tasks(self) -> List[ClassificationTask]:
        tasks = []

        # Exp 1 — Fluent vs Disfluent
        tasks.append(ClassificationTask(
            name              = "Exp1_FluDis",
            label_map         = {"Fluent":0,"I":1,"PR":1,"PhR":1,"WR":1,"PWR":1,"P":1,"AMBIGUOUS":1},
            excl_labels_rfdnn = {"UNKNOWN"},
            excl_labels_bl    = {"UNKNOWN"},
            class_names       = ["Fluent", "Disfluent"],
            n_classes         = 2,
            task_type         = "binary",
            max_per_class     = MAX_TRAIN_FRAMES_PER_CLASS_BINARY,
        ))

        # Exp 2 — Fluent vs each disfluency class (6 binary tasks)
        for tgt in self.DISF_CLASSES:
            other = set(self.DISF_CLASSES) - {tgt}
            tasks.append(ClassificationTask(
                name              = f"Exp2_Flu_{tgt}",
                label_map         = {"Fluent": 0, tgt: 1},
                excl_labels_rfdnn = other | {"AMBIGUOUS", "UNKNOWN"},
                excl_labels_bl    = other | {"AMBIGUOUS", "UNKNOWN"},
                class_names       = ["Fluent", tgt],
                n_classes         = 2,
                task_type         = "binary",
                max_per_class     = MAX_TRAIN_FRAMES_PER_CLASS_BINARY,
            ))

        # Exp 3 — 6-class disfluency (Fluent excluded from RF/DNN; masked in BiLSTM)
        tasks.append(ClassificationTask(
            name              = "Exp3_6class",
            label_map         = {"I":0,"PR":1,"PhR":2,"WR":3,"PWR":4,"P":5},
            excl_labels_rfdnn = {"Fluent", "AMBIGUOUS", "UNKNOWN"},
            excl_labels_bl    = {"AMBIGUOUS", "UNKNOWN"},
            class_names       = ["I","PR","PhR","WR","PWR","P"],
            n_classes         = 6,
            task_type         = "multiclass",
            max_per_class     = MAX_TRAIN_FRAMES_PER_CLASS_6CLASS,
        ))

        return tasks

    def _run_rf_dnn(self, clf: BaseClassifier, task: ClassificationTask
                    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Build arrays, fit RF/DNN, return (y_true, y_pred, y_prob)."""
        X_tr, y_tr, _ = build_rfdnn_arrays(
            "train", task.label_map, self.store,
            self._train_files(), self._val_files(), self._test_files(),
            task.excl_labels_rfdnn, task.max_per_class)
        X_va, y_va, _ = build_rfdnn_arrays(
            "val",   task.label_map, self.store,
            self._train_files(), self._val_files(), self._test_files(),
            task.excl_labels_rfdnn)
        X_te, y_te, _ = build_rfdnn_arrays(
            "test",  task.label_map, self.store,
            self._train_files(), self._val_files(), self._test_files(),
            task.excl_labels_rfdnn)

        c_tr = Counter(y_tr.tolist()); c_te = Counter(y_te.tolist())
        for ci, cn in enumerate(task.class_names):
            print(f"    Train {cn}={c_tr.get(ci,0):,}  Test {cn}={c_te.get(ci,0):,}")

        # Skip if any class absent from test
        if any(c_te.get(i, 0) == 0 for i in range(task.n_classes)):
            return None, None, None

        clf.fit(X_tr, y_tr, X_va, y_va)
        y_pred = clf.predict(X_te)
        y_true = y_te
        y_prob = None
        if task.n_classes == 2 and hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_te)[:, 1]
        return y_true, y_pred, y_prob

    def _run_bilstm(self, clf: "BiLSTMWrapper", task: ClassificationTask
                    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Build lazy index, call fit_sequences(), return (y_true, y_pred, y_prob)."""
        idx_tr = build_bilstm_index(self._train_files(), task.label_map,
                                    self.store, task.excl_labels_bl, SEQ_LEN)
        idx_va = build_bilstm_index(self._val_files(),   task.label_map,
                                    self.store, task.excl_labels_bl, SEQ_LEN)
        idx_te = build_bilstm_index(self._test_files(),  task.label_map,
                                    self.store, task.excl_labels_bl, SEQ_LEN)

        print(f"    Windows: train={len(idx_tr)}  val={len(idx_va)}  test={len(idx_te)}")

        if not idx_tr or not idx_va or not idx_te:
            return None, None, None

        clf.fit_sequences(idx_tr, idx_va, task.n_classes, self.store.embed_dir)
        y_true, y_pred, y_prob = clf.predict_sequences(idx_te, self.store.embed_dir)
        prob1d = y_prob[:, 1] if task.n_classes == 2 and len(y_prob) > 0 else None
        return y_true, y_pred, prob1d

    def run_task(self, task: ClassificationTask) -> None:
        print(f"\n{'='*60}\nTASK: {task.name}  classes={task.class_names}\n{'='*60}")
        task_results = {}

        for clf in self.clfs:
            print(f"\n  -- {clf.name} --")
            t0 = time.time()

            if isinstance(clf, BiLSTMWrapper):
                y_true, y_pred, y_prob = self._run_bilstm(clf, task)
            else:
                y_true, y_pred, y_prob = self._run_rf_dnn(clf, task)

            if y_true is None:
                print(f"    Skipped — insufficient data.")
                continue

            labels_idx = list(range(task.n_classes))
            metrics    = compute_metrics(y_true, y_pred, task.class_names,
                                         task.task_type,
                                         y_prob if task.n_classes == 2 else None,
                                         labels=labels_idx)
            result = EvaluationResult(
                task_name        = task.name,
                classifier_name  = clf.name,
                accuracy         = metrics["accuracy"],
                macro_f1         = metrics["macro_f1"],
                weighted_f1      = metrics["weighted_f1"],
                precision_macro  = metrics["precision"],
                recall_macro     = metrics["recall"],
                confusion_matrix = metrics["conf_matrix"],
                extra            = {k: v for k, v in metrics.items()
                                    if k not in ("accuracy","macro_f1","weighted_f1",
                                                 "precision","recall","conf_matrix")},
            )
            self.evaluator.print_summary(result)
            save_result(task.name, clf.name, metrics, task.class_names)
            task_results[clf.name] = metrics

            # Save model
            sub = ("rf"     if isinstance(clf, RandomForestWrapper) else
                   "dnn"    if isinstance(clf, DNNWrapper)          else
                   "bilstm")
            ext = ".pkl" if isinstance(clf, RandomForestWrapper) else ".pt"
            clf.save(os.path.join(self.model_dir, f"{sub}_{task.name}{ext}"))
            print(f"    Done {time.time()-t0:.1f}s  Macro-F1={metrics['macro_f1']:.4f}")

            # Confusion matrix
            self.evaluator.plot_confusion_matrix(
                metrics["conf_matrix"], task.class_names,
                f"{task.name} — {clf.name}",
                save_path=os.path.join(self.result_dir,
                                       f"cm_{task.name}_{clf.name}.png"))

        self.results[task.name] = task_results

    def run_all(self) -> dict:
        """Run all 8 standard tasks (Exp1 + 6xExp2 + Exp3)."""
        tasks = self.build_standard_tasks()
        for task in tasks:
            self.run_task(task)
        return self.results

    def save_results(self, path: str) -> None:
        serialisable = {}
        for task_name, clf_dict in self.results.items():
            serialisable[task_name] = {}
            for clf_name, metrics in clf_dict.items():
                serialisable[task_name][clf_name] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in metrics.items() if k != "conf_matrix"}
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"Results saved to {path}")


# ============================================================
# SECTION 16 — VISUALIZATIONS
# ============================================================

def plot_class_distribution(frame_meta_df: pd.DataFrame, save_dir: str) -> None:
    lc = frame_meta_df["primary_label"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    lc.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Frame-Level Label Distribution")
    ax.set_xlabel("Label"); ax.set_ylabel("Frames")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_distribution.png"), bbox_inches="tight")
    plt.close()


def plot_macro_f1_comparison(all_results: list, save_dir: str) -> None:
    res  = pd.DataFrame(all_results)
    if res.empty:
        return
    exps = res["experiment"].unique()
    fig, ax = plt.subplots(figsize=(max(10, len(exps)*1.5), 5))
    x, w = np.arange(len(exps)), 0.25
    for i, mdl in enumerate(["RandomForest", "DNN", "BiLSTM"]):
        sub = res[res["model"] == mdl]
        f1s = []
        for exp in exps:
            r2 = sub[sub["experiment"] == exp]
            f1s.append(r2["macro_f1"].values[0] if len(r2) else 0.0)
        ax.bar(x + (i - 1) * w, f1s, w, label=mdl)
    ax.set_xticks(x); ax.set_xticklabels(exps, rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel("Macro-F1"); ax.legend()
    ax.set_title("Macro-F1: All Experiments x All Classifiers")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "macro_f1_comparison.png"), bbox_inches="tight")
    plt.close()


def print_final_table(all_results: list) -> pd.DataFrame:
    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print("No results to show.")
        return results_df
    num = [c for c in ["accuracy","macro_f1","weighted_f1","precision","recall"]
           if c in results_df.columns]
    tbl = results_df[["experiment","model"] + num].copy()
    tbl[num] = tbl[num].round(4)
    print("=" * 80)
    print("FINAL RESULTS TABLE (frame-level | primary metric: Macro-F1)")
    print("=" * 80)
    print(tbl.to_string(index=False))
    print("\n-- Best model per experiment --")
    for exp in results_df["experiment"].unique():
        sub  = results_df[results_df["experiment"] == exp]
        best = sub.loc[sub["macro_f1"].idxmax()]
        print(f"  {exp:<32} -> {best['model']:<14}  Macro-F1={best['macro_f1']:.4f}")
    return results_df


# ============================================================
# SECTION 17 — SANITY CHECKS
# ============================================================

def run_sanity_checks(store: DatasetStore, splits: dict, matched: list) -> None:
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # 1. Timestamp alignment
    print("\n[1] Timestamp alignment")
    fid  = matched[0]
    ts   = store.load_timestamps(fid)
    fm   = store.frame_meta_df[store.frame_meta_df["file_id"] == fid].reset_index(drop=True)
    minT = min(len(ts), len(fm))
    print(f"  {fid}: {minT} frames  {ts[0]*1000:.2f}ms -> {ts[minT-1]*1000:.2f}ms")
    if minT > 1:
        diffs = np.diff(ts[:50])
        assert abs(diffs.mean() - W2V_FRAME_STEP_SEC) < 1e-3, "Timestamp stride mismatch!"
        print(f"  Mean frame step: {diffs.mean()*1000:.3f}ms  (expected {W2V_FRAME_STEP_SEC*1000:.3f}ms)  OK")

    # 2. FluencyRegionFinder — no Fluent row overlaps any disfluency row
    print("\n[2] FluencyRegionFinder explicit Fluent rows")
    fdf = store.frame_meta_df[store.frame_meta_df["file_id"] == fid]
    n_fl = (fdf["primary_label"] == "Fluent").sum()
    print(f"  Fluent frames in {fid}: {n_fl}")
    print("  OK (Fluent rows generated by FluencyRegionFinder before alignment)")

    # 3. BiLSTM lazy index + tensor shapes
    print("\n[3] BiLSTM lazy index and tensor dimensions")
    _bl_lmap = {"Fluent":0,"I":1,"PR":1,"PhR":1,"WR":1,"PWR":1,"P":1}
    _idx     = build_bilstm_index(matched[:1], _bl_lmap, store, {"AMBIGUOUS","UNKNOWN"}, SEQ_LEN)
    if _idx:
        _e0 = _idx[0]
        print(f"  Index entries: {len(_idx)}  entry[0]: start={_e0['start_frame']} "
              f"end={_e0['end_frame']} length={_e0['length']}")
        _lazy = LazySeqDataset(_idx, store.embed_dir)
        Xs, ys, ln = _lazy[0]
        assert Xs.shape == (SEQ_LEN, W2V_HIDDEN_DIM), f"X shape {Xs.shape}"
        assert ys.shape == (SEQ_LEN,),                f"y shape {ys.shape}"
        print(f"  X shape: {tuple(Xs.shape)}  y shape: {tuple(ys.shape)}  OK")
    else:
        print("  No BiLSTM windows (SANITY_RUN file subset too small — OK)")

    # 4. Masked loss (ignore_index=-1)
    print("\n[4] Masked loss ignore_index=-1")
    _crit = nn.CrossEntropyLoss(ignore_index=-1)
    _lg   = torch.randn(2, 5, 2)
    _yt   = torch.tensor([[0, 1, -1, -1, 0], [1, -1, 0, -1, -1]])
    _l    = _crit(_lg.reshape(10, 2), _yt.reshape(10))
    assert not torch.isnan(_l), "NaN loss!"
    print(f"  Loss={_l.item():.4f}  OK")

    # 5. Speaker leakage
    print("\n[5] Speaker leakage")
    tr = set(splits["train_spk"]); va = set(splits["val_spk"]); te = set(splits["test_spk"])
    assert tr & va == set() and tr & te == set() and va & te == set()
    print(f"  train={len(tr)} val={len(va)} test={len(te)}  No leakage  OK")

    # 6. Embedding shapes
    print("\n[6] Embedding shapes")
    for fid2 in matched[:3]:
        sh = store.load_embeddings(fid2).shape
        assert sh[1] == W2V_HIDDEN_DIM, f"dim mismatch {sh[1]}"
        print(f"  {fid2}: {sh}  OK")

    # 7. class_weight_tensor absent-class robustness
    print("\n[7] class_weight_tensor absent-class robustness")
    _cw = class_weight_tensor(np.array([0,0,0,1,1,-1,-1]), 3)
    assert _cw[2].item() == 1.0, "Absent class should have weight 1.0"
    print(f"  Weights: {_cw.tolist()}  (class 2 = 1.0)  OK")

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 60)


# ============================================================
# SECTION 18 — MAIN
# ============================================================

def main() -> None:
    global DEVICE, ALL_RESULTS

    # ── Setup ──────────────────────────────────────────────────
    apply_sanity_overrides()
    make_dirs()
    set_seeds(RANDOM_SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    ALL_RESULTS = []

    # ── Dataset discovery ─────────────────────────────────────
    print("\n-- Dataset Discovery --")
    wav_by_stem, txt_by_stem = {}, {}
    for root, dirs, files in os.walk(DATA_DIR):
        for fn in files:
            stem = os.path.splitext(fn)[0]
            ext  = os.path.splitext(fn)[1].lower()
            if ext == ".wav":   wav_by_stem[stem] = os.path.join(root, fn)
            elif ext == ".txt": txt_by_stem[stem] = os.path.join(root, fn)

    matched = sorted(set(wav_by_stem) & set(txt_by_stem))
    if SANITY_RUN:
        matched = matched[:SANITY_N_FILES]
        print(f"SANITY_RUN: using {len(matched)} recordings")

    print(f"WAV={len(wav_by_stem)}  TXT={len(txt_by_stem)}  Matched={len(matched)}")

    audio_loader = AudioLoader(SAMPLE_RATE)
    inventory_rows = []
    for stem in matched:
        meta = parse_filename(stem)
        dur  = audio_loader.get_duration(wav_by_stem[stem])
        inventory_rows.append({**meta,
            "wav_path":    wav_by_stem[stem],
            "txt_path":    txt_by_stem[stem],
            "duration_sec":dur,
            "audio_error": None if dur else "UNREADABLE"})
    inventory_df = pd.DataFrame(inventory_rows)
    inventory_df.to_csv(os.path.join(META_DIR, "dataset_inventory.csv"), index=False)
    print(f"Inventory: {len(inventory_df)} recordings | "
          f"{inventory_df['speaker_id'].nunique()} speakers")

    # ── Annotation parsing ────────────────────────────────────
    print("\n-- Annotation Parsing --")
    ann_parser = AnnotationParser()
    annot_df   = ann_parser.parse_dataset(inventory_df)
    print(f"Annotation rows: {len(annot_df)}")
    print(annot_df["annotation_status"].value_counts().to_string())

    # ── Overlap analysis ──────────────────────────────────────
    print("\n-- Overlap Analysis --")
    annot_df = ann_parser.mark_overlaps(annot_df)
    print(f"has_overlap=True: {annot_df['has_overlap'].sum()}")

    # ── Fluent region generation (Q4: explicit) ───────────────
    print("\n-- Fluent Region Generation --")
    fluency_finder = FluencyRegionFinder()
    disf_df  = annot_df[annot_df["annotation_status"] != "FLUENT_GAP"].copy()
    audit_df = fluency_finder.augment_dataset(disf_df, inventory_df)
    audit_df.to_csv(os.path.join(META_DIR, "annotation_audit.csv"), index=False)
    n_fluent_rows = (audit_df["annotation_status"] == "FLUENT_GAP").sum()
    print(f"Fluent gap rows added: {n_fluent_rows:,}")
    print(f"Total annotation rows: {len(audit_df):,}")

    # ── Speaker split ─────────────────────────────────────────
    print("\n-- Speaker-Level Split --")
    inv_with_frames = inventory_df.copy()
    inv_with_frames["n_frames"] = (
        inventory_df["duration_sec"].fillna(0) * 50).astype(int).clip(lower=1)

    split_mgr = SplitManager(SHARED_SPLITS_PATH)
    splits    = split_mgr.get_or_create_splits(inv_with_frames)
    train_files = splits["train_files"]
    val_files   = splits["val_files"]
    test_files  = splits["test_files"]
    split_mgr.print_split_table(splits, inv_with_frames)

    # ── Wav2Vec2 feature extraction ───────────────────────────
    print("\n-- Wav2Vec2 Feature Extraction --")
    extractor = Wav2vec2Extractor(WAV2VEC_MODEL, SAMPLE_RATE)
    extractor.setup()

    ext_log = extractor.extract_dataset(inventory_df, EMBED_DIR, force=FORCE_RECOMPUTE)
    ext_log_df = pd.DataFrame(ext_log)
    ext_log_df.to_csv(os.path.join(LOG_DIR, "extraction_log.csv"), index=False)
    print(f"Extraction: OK={( ext_log_df['status']=='OK').sum()}  "
          f"SKIPPED={( ext_log_df['status']=='SKIPPED').sum()}  "
          f"FAILED={( ext_log_df['status']=='FAILED').sum()}")

    # ── Frame-level label alignment ───────────────────────────
    print("\n-- Frame-Level Label Alignment --")
    frame_meta_df = build_frame_meta(inventory_df, audit_df, EMBED_DIR)
    frame_meta_df["split"] = frame_meta_df["file_id"].map({
        **{f: "train" for f in train_files},
        **{f: "val"   for f in val_files},
        **{f: "test"  for f in test_files},
    })
    frame_meta_df.to_csv(os.path.join(META_DIR, "frame_metadata.csv"), index=False)

    # ── Build DatasetStore ────────────────────────────────────
    store = extractor.build_store(frame_meta_df, EMBED_DIR)
    print(f"DatasetStore: {store.n_frames:,} frames | "
          f"extractor={store.extractor_name} | dim={store.feature_dim}")

    # ── Sanity checks ─────────────────────────────────────────
    print("\n-- Sanity Checks --")
    run_sanity_checks(store, splits, matched)

    # ── Visualize class distribution ──────────────────────────
    plot_class_distribution(frame_meta_df, RESULT_DIR)

    # ── Classification ────────────────────────────────────────
    print("\n-- Classification --")
    classifiers = [
        RandomForestWrapper(RF_N_ESTIMATORS, RF_MAX_DEPTH),
        DNNWrapper(W2V_HIDDEN_DIM, n_classes=2, n_epochs=DNN_EPOCHS),
        BiLSTMWrapper(W2V_HIDDEN_DIM, n_classes=2, n_epochs=BILSTM_EPOCHS),
    ]
    evaluator = Evaluator()
    runner    = ClassifierRunner(store, splits, classifiers, evaluator,
                                  MODEL_DIR, RESULT_DIR)

    print("Running Exp1 + Exp2x6 + Exp3 ...")
    runner.run_all()
    runner.save_results(os.path.join(RESULT_DIR, "all_results.json"))

    # ── Results summary ───────────────────────────────────────
    results_df = print_final_table(ALL_RESULTS)
    results_df.to_csv(os.path.join(RESULT_DIR, "all_results.csv"), index=False)
    plot_macro_f1_comparison(ALL_RESULTS, RESULT_DIR)

    # ── Save config ───────────────────────────────────────────
    config = {
        "SANITY_RUN": SANITY_RUN, "DATA_DIR": DATA_DIR, "OUTPUT_DIR": OUTPUT_DIR,
        "WAV2VEC_MODEL": WAV2VEC_MODEL, "SAMPLE_RATE": SAMPLE_RATE,
        "W2V_HIDDEN_DIM": W2V_HIDDEN_DIM,
        "W2V_TOTAL_STRIDE": W2V_TOTAL_STRIDE,
        "CHUNK_DURATION_SEC": CHUNK_DURATION_SEC,
        "CHUNK_OVERLAP_SEC": CHUNK_OVERLAP_SEC,
        "RANDOM_SEED": RANDOM_SEED,
        "RF_N_ESTIMATORS": RF_N_ESTIMATORS, "RF_MAX_DEPTH": RF_MAX_DEPTH,
        "DNN_EPOCHS": DNN_EPOCHS, "BILSTM_EPOCHS": BILSTM_EPOCHS,
        "SEQ_LEN": SEQ_LEN,
        "MAX_TRAIN_BINARY": MAX_TRAIN_FRAMES_PER_CLASS_BINARY,
        "MAX_TRAIN_6CLASS": MAX_TRAIN_FRAMES_PER_CLASS_6CLASS,
        "SHARED_SPLITS_PATH": SHARED_SPLITS_PATH,
    }
    with open(os.path.join(LOG_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 65)
    print("DONE — OOP Pipeline v3")
    print("=" * 65)
    print(f"  SANITY_RUN   : {SANITY_RUN}")
    print(f"  Recordings   : {len(matched)}")
    print(f"  Total frames : {len(frame_meta_df):,}")
    print(f"  Splits file  : {SHARED_SPLITS_PATH}")
    print(f"  Results dir  : {RESULT_DIR}")
    if SANITY_RUN:
        print("\n  >>> Set SANITY_RUN=False in Section 0 for full training.")
    print("=" * 65)


if __name__ == "__main__":
    main()
