"""
IED Disfluency Dataset — Prosodic + Acoustic OOP Pipeline
===========================================================

Refactored to follow the same single-file OOP structure as the
Wav2Vec2 pipeline, while replacing Wav2Vec2 embeddings with the
17-D Prosodic + Acoustic feature representation.

Run:
    python ied_prosodic_acoustic_oop_wav2vec_style.py

Pipeline:
    WAV + TXT
        -> ProsodicAcousticExtractor
        -> standardized 17-D features
        -> DatasetStore
        -> speaker-level split
        -> RF / DNN / BiLSTM
        -> Weighted-F1-first evaluation
"""

from __future__ import annotations

import csv
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
from scipy.signal import bilinear, find_peaks, hilbert, lfilter, savgol_filter
from scipy.stats import skew
from gammatone.gtgram import gtgram

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_DIR = Path(r"/content/drive/MyDrive/IED Dataset")
OUTPUT_DIR = Path(r"/content/drive/MyDrive/prosodic_acoustic_output_safe")
SAMPLE_RATE = 16_000

# Use 1 for a quick test. Change to None for all files.
MAX_FILES = 1

# Segmentation is done in chunks to keep RAM manageable.
CHUNK_DURATION = 30.0
CHUNK_OVERLAP = 0.5

# ============================================================================
# LABEL NORMALISATION
# ============================================================================

VALID_LABELS = {"I", "PR", "PhR", "WR", "PWR", "P", "Fluent"}

LABEL_MAP: Dict[str, str] = {
    "W.R": "WR", "W.R.": "WR", "WR": "WR", "wr": "WR",
    "PWR": "PWR", "PW.R": "PWR", "P.W.R": "PWR", "PWr": "PWR",
    "PHR": "PhR", "PH.R": "PhR", "Ph.r": "PhR", "Ph.R": "PhR", "PhR": "PhR",
    "PR": "PR", "Pr": "PR", "P.R": "PR", "P.r": "PR",
    "I": "I", "i": "I",
    "P": "P",
    "Fluent": "Fluent",
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AnnotationRow:
    file: str
    start: float
    end: float
    label: str
    duration: float
    raw_label: str
    source_line: int
    has_overlap: bool = False


@dataclass
class FeatureRecord:
    file: str
    start: float
    end: float
    label: str
    duration: float
    has_overlap: bool
    features: np.ndarray


@dataclass
class DatasetStore:
    X: np.ndarray
    labels: np.ndarray
    files: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    durations: np.ndarray
    has_overlap: np.ndarray
    feature_names: List[str]
    extractor_name: str

# ============================================================================
# AUDIO LOADING
# ============================================================================

class AudioLoader:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    def load(self, wav_path: Path) -> np.ndarray:
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)

        if audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        else:
            audio = audio[:, 0]

        if sr != self.sample_rate:
            raise ValueError(
                f"Expected {self.sample_rate} Hz, got {sr} Hz in {wav_path.name}"
            )

        return audio

# ============================================================================
# ANNOTATION PARSER
# ============================================================================

class AnnotationParser:
    def _normalise_label(self, raw_label: str) -> Optional[str]:
        label = raw_label.strip()
        label = re.sub(r"\([^)]*\)", "", label).strip()
        mapped = LABEL_MAP.get(label)
        if mapped is not None:
            return mapped
        return label if label in VALID_LABELS else None

    def parse_file(self, txt_path: Path) -> List[AnnotationRow]:
        rows: List[AnnotationRow] = []

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for source_line, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = re.split(r"\s+", line)
                if len(parts) < 3:
                    continue

                try:
                    start = float(parts[0])
                    end = float(parts[1])
                except ValueError:
                    continue

                if end <= start:
                    continue

                raw_label = str(parts[2]).strip()
                raw_parts = [p.strip() for p in raw_label.split(",") if p.strip()]

                for raw_part in raw_parts:
                    label = self._normalise_label(raw_part)
                    if label is None:
                        continue

                    rows.append(
                        AnnotationRow(
                            file=txt_path.stem,
                            start=start,
                            end=end,
                            label=label,
                            duration=end - start,
                            raw_label=raw_part,
                            source_line=source_line,
                        )
                    )

        return rows

    def parse_dataset(self, dataset_dir: Path) -> List[AnnotationRow]:
        rows: List[AnnotationRow] = []
        txt_paths = sorted(dataset_dir.rglob("*.txt"))

        print(f"Found TXT files: {len(txt_paths)}", flush=True)

        for i, txt_path in enumerate(txt_paths, 1):
            rows.extend(self.parse_file(txt_path))
            if i == 1 or i % 25 == 0 or i == len(txt_paths):
                print(
                    f"  Parsed annotations: {i}/{len(txt_paths)}",
                    flush=True,
                )

        return rows

# ============================================================================
# SEGMENTATION
# ============================================================================

def compute_gammatone(
    audio_signal: np.ndarray,
    sr: int,
    channels: int = 20,
    f_min: int = 100,
    window_time: float = 0.025,
    hop_time: float = 0.001,
) -> np.ndarray:
    return gtgram(audio_signal, sr, window_time, hop_time, channels, f_min)


def compute_sonority(
    gammatone_features: np.ndarray,
    fs_env: int = 1000,
    center_freq: float = 5,
    bandwidth: float = 6,
) -> np.ndarray:
    amplitude = np.sqrt(np.maximum(gammatone_features, 0))

    m = 1.0
    omega0 = 2 * np.pi * center_freq
    k = m * omega0**2
    d = 2 * np.pi * bandwidth * m

    bz, az = bilinear([1.0], [m, d, k], fs=fs_env)

    osc = np.zeros_like(amplitude)
    for ch in range(amplitude.shape[0]):
        osc[ch] = lfilter(bz, az, amplitude[ch])

    osc_amp = np.zeros_like(osc)
    for ch in range(osc.shape[0]):
        osc_amp[ch] = np.abs(hilbert(osc[ch]))

    channel_energy = np.mean(osc_amp**2, axis=1)
    top10 = np.argsort(channel_energy)[-10:]

    sonority = np.sum(np.log10(osc_amp[top10] + 1e-8), axis=0)
    sonority -= sonority.min()

    if sonority.max() > 0:
        sonority /= sonority.max()

    return sonority


def detect_syllable_regions(
    sonority: np.ndarray,
    fs_env: int = 1000,
) -> pd.DataFrame:
    window = min(101, len(sonority))
    if window % 2 == 0:
        window -= 1

    if window >= 5:
        smooth = savgol_filter(sonority, window, 3)
    else:
        smooth = sonority.copy()

    smooth = np.clip(smooth, 0, 1)

    peaks, _ = find_peaks(
        smooth,
        distance=int(0.08 * fs_env),
        prominence=0.04,
    )

    valleys, _ = find_peaks(
        -smooth,
        distance=int(0.06 * fs_env),
        prominence=0.02,
    )

    boundaries = [v for v in valleys if np.any(peaks < v)]
    boundary_times = np.asarray(boundaries) / fs_env

    rows = []
    for i in range(len(boundary_times) - 1):
        start = float(boundary_times[i])
        end = float(boundary_times[i + 1])
        duration = end - start

        if duration < 0.08:
            continue

        rows.append(
            {
                "syllable_id": len(rows) + 1,
                "start": start,
                "end": end,
                "duration": duration,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["syllable_id", "start", "end", "duration"],
    )

# ============================================================================
# PRECOMPUTED FILE-LEVEL ACOUSTIC SIGNALS
# ============================================================================

class FileAcousticCache:
    """Compute expensive acoustic signals once per WAV and reuse them."""

    def __init__(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
        self.audio = audio
        self.sr = sample_rate

        # One Praat pitch calculation for the whole file.
        snd = parselmouth.Sound(audio, sampling_frequency=sample_rate)
        pitch = snd.to_pitch(
            time_step=0.005,
            pitch_floor=75,
            pitch_ceiling=500,
        )

        self.f0_values = pitch.selected_array["frequency"].astype(np.float32)
        self.f0_values[self.f0_values == 0] = np.nan
        self.f0_times = pitch.xs().astype(np.float64)

        # One RMS calculation for the whole file.
        self.rms_hop = 160
        self.rms_frame = 320
        self.rms = librosa.feature.rms(
            y=audio,
            frame_length=self.rms_frame,
            hop_length=self.rms_hop,
            center=True,
        )[0].astype(np.float32)

        self.energy_times = (
            np.arange(len(self.rms)) * self.rms_hop / sample_rate
        )

        max_rms = float(np.max(self.rms)) if len(self.rms) else 0.0
        self.rms_norm = (
            self.rms / max_rms if max_rms > 0 else self.rms.copy()
        )

    def f0_features(self, start: float, end: float) -> List[float]:
        mask = (self.f0_times >= start) & (self.f0_times < end)
        f0 = self.f0_values[mask]
        f0 = f0[np.isfinite(f0)]

        if len(f0):
            values = [
                np.mean(f0),
                np.std(f0),
                np.min(f0),
                np.max(f0),
                skew(f0) if len(f0) > 2 else 0.0,
            ]
        else:
            values = [0.0] * 5

        if len(f0) >= 2:
            x = np.arange(len(f0))
            tilt = np.polyfit(x, f0, 1)[0]
            fd = np.diff(f0)
            values.extend([
                tilt,
                np.mean(fd),
                np.std(fd),
                skew(fd) if len(fd) > 2 else 0.0,
            ])
        else:
            values.extend([0.0] * 4)

        return [float(v) for v in values]

    def energy_features(self, start: float, end: float) -> List[float]:
        mask = (self.energy_times >= start) & (self.energy_times < end)
        energy = self.rms_norm[mask]

        if len(energy):
            values = [
                np.mean(energy),
                np.std(energy),
                skew(energy) if len(energy) > 2 else 0.0,
            ]
        else:
            values = [0.0] * 3

        if len(energy) >= 2:
            ed = np.diff(energy)
            values.extend([
                np.mean(ed),
                np.std(ed),
                skew(ed) if len(ed) > 2 else 0.0,
            ])
        else:
            values.extend([0.0] * 3)

        return [float(v) for v in values]

    def pause_after(self, end: float) -> float:
        """Estimate the longest low-energy run after the region, up to 2 s."""
        start_idx = int(np.searchsorted(self.energy_times, end, side="left"))
        if start_idx >= len(self.rms_norm):
            return 0.0

        max_frames = int(2.0 * self.sr / self.rms_hop) + 2
        following = self.rms_norm[start_idx:start_idx + max_frames]

        if len(following) == 0:
            return 0.0

        max_rms = float(np.max(following))
        if max_rms <= 0:
            return 0.0

        pause_mask = following < max_rms * 0.10

        frames = 0
        best_frames = 0
        for is_pause in pause_mask:
            if is_pause:
                frames += 1
                best_frames = max(best_frames, frames)
            else:
                frames = 0

        return float(best_frames * self.rms_hop / self.sr)

# ============================================================================
# PROSODIC FEATURE EXTRACTOR
# ============================================================================

class ProsodicAcousticExtractor:
    name = "ProsodicAcoustic"

    FEATURE_NAMES = [
        "duration",
        "f0_mean", "f0_std", "f0_min", "f0_max", "f0_skew",
        "f0_tilt", "f0_tilt_mean", "f0_tilt_std", "f0_tilt_skew",
        "energy_mean", "energy_std", "energy_skew",
        "energy_tilt_mean", "energy_tilt_std", "energy_tilt_skew",
        "pause_after",
    ]

    def extract_segment(
        self,
        acoustic: FileAcousticCache,
        start: float,
        end: float,
    ) -> np.ndarray:
        duration = max(0.0, float(end - start))
        f0 = acoustic.f0_features(start, end)
        energy = acoustic.energy_features(start, end)
        pause = acoustic.pause_after(end)

        vector = np.asarray(
            [duration, *f0, *energy, pause],
            dtype=np.float32,
        )

        if len(vector) != len(self.FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(self.FEATURE_NAMES)} features, got {len(vector)}"
            )

        return vector

# ============================================================================
# LABEL ASSIGNMENT
# ============================================================================

def assign_label(region_start: float, region_end: float, annotations: List[AnnotationRow]):
    best = None
    best_overlap = 0.0

    for ann in annotations:
        if ann.end <= region_start:
            continue
        if ann.start >= region_end:
            break

        overlap = max(
            0.0,
            min(region_end, ann.end) - max(region_start, ann.start),
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best = ann

    duration = max(0.0, region_end - region_start)
    ratio = best_overlap / duration if duration > 0 else 0.0

    if best is None:
        return "Fluent", np.nan, np.nan, 0.0, 0.0

    return (
        best.label,
        best.start,
        best.end,
        best_overlap,
        ratio,
    )

# ============================================================================
# SAVE
# ============================================================================

def save_outputs(
    records: List[FeatureRecord],
    output_dir: Path,
    failures: List[dict],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    records = sorted(records, key=lambda r: (r.file, r.start, r.end))

    X = np.stack([r.features for r in records]).astype(np.float32)

    files = np.array([r.file for r in records])
    starts = np.array([r.start for r in records], dtype=np.float64)
    ends = np.array([r.end for r in records], dtype=np.float64)
    durations = np.array([r.duration for r in records], dtype=np.float32)
    labels = np.array([r.label for r in records])
    overlap = np.array([r.has_overlap for r in records], dtype=bool)

    npz_path = output_dir / "prosodic_acoustic_features.npz"
    csv_path = output_dir / "prosodic_acoustic_metadata.csv"
    failures_path = output_dir / "prosodic_acoustic_failures.csv"

    np.savez_compressed(
        npz_path,
        X=X,
        labels=labels,
        files=files,
        starts=starts,
        ends=ends,
        durations=durations,
        has_overlap=overlap,
        feature_names=np.array(ProsodicAcousticExtractor.FEATURE_NAMES),
        extractor_name=np.array("ProsodicAcoustic"),
    )

    metadata = pd.DataFrame({
        "idx": np.arange(len(records)),
        "file": files,
        "start": starts,
        "end": ends,
        "label": labels,
        "duration": durations,
        "has_overlap": overlap,
    })

    # Add the actual feature values to metadata too.
    feature_df = pd.DataFrame(
        X,
        columns=ProsodicAcousticExtractor.FEATURE_NAMES,
    )
    metadata = pd.concat([metadata, feature_df], axis=1)
    metadata.to_csv(csv_path, index=False)

    pd.DataFrame(
        failures,
        columns=["file", "start", "end", "label", "reason"],
    ).to_csv(failures_path, index=False)

    print(f"Saved features -> {npz_path}", flush=True)
    print(f"Saved metadata -> {csv_path}", flush=True)
    print(f"Saved failures -> {failures_path}", flush=True)

    return X, labels, files

# ============================================================================
# MAIN
# ============================================================================

def extract_features_main():
    t0 = time.time()

    print("=" * 72, flush=True)
    print("  IED PROSODIC ACOUSTIC FEATURE EXTRACTION", flush=True)
    print("=" * 72, flush=True)
    print(f"Dataset : {DATASET_DIR}", flush=True)
    print(f"Output  : {OUTPUT_DIR}", flush=True)
    print(f"MAX_FILES: {MAX_FILES}", flush=True)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    wav_paths = {p.stem: p for p in DATASET_DIR.rglob("*.wav")}
    txt_paths = {p.stem: p for p in DATASET_DIR.rglob("*.txt")}
    common_ids = sorted(set(wav_paths) & set(txt_paths))

    print(f"WAV files found: {len(wav_paths)}", flush=True)
    print(f"TXT files found: {len(txt_paths)}", flush=True)
    print(f"Matched WAV + TXT: {len(common_ids)}", flush=True)

    if MAX_FILES is not None:
        common_ids = common_ids[:MAX_FILES]
        print(f"TEST MODE: processing only {len(common_ids)} file(s)", flush=True)

    parser = AnnotationParser()
    annotations_by_file: Dict[str, List[AnnotationRow]] = {}

    print("\n[Step 1] Parsing selected annotations ...", flush=True)
    for stem in common_ids:
        rows = parser.parse_file(txt_paths[stem])
        rows.sort(key=lambda x: (x.start, x.end))
        annotations_by_file[stem] = rows

    print(
        "Annotation rows in selected files:",
        sum(len(v) for v in annotations_by_file.values()),
        flush=True,
    )

    extractor = ProsodicAcousticExtractor()
    loader = AudioLoader(SAMPLE_RATE)

    records: List[FeatureRecord] = []
    failures: List[dict] = []

    print("\n[Step 2] Extracting features ...", flush=True)

    for fi, stem in enumerate(common_ids, start=1):
        file_start_time = time.time()
        wav_path = wav_paths[stem]

        print(
            f"\n[{fi}/{len(common_ids)}] {wav_path.name}",
            flush=True,
        )

        try:
            audio = loader.load(wav_path)
            audio_duration = len(audio) / SAMPLE_RATE
            print(f"  Audio duration: {audio_duration:.2f}s", flush=True)

            # Compute F0 + RMS once for the entire file.
            print("  Computing file-level F0 + RMS ...", flush=True)
            acoustic = FileAcousticCache(audio, SAMPLE_RATE)
            print("  ✓ F0 + RMS ready", flush=True)

            file_annotations = annotations_by_file[stem]
            file_records_start = len(records)

            chunk_start = 0.0
            step = CHUNK_DURATION - CHUNK_OVERLAP

            chunk_no = 0
            total_chunks = max(1, int(np.ceil(audio_duration / step)))

            while chunk_start < audio_duration:
                chunk_no += 1
                chunk_end = min(chunk_start + CHUNK_DURATION, audio_duration)

                print(
                    f"  Chunk {chunk_no}/{total_chunks}: "
                    f"{chunk_start:.1f}-{chunk_end:.1f}s",
                    flush=True,
                )

                sample_start = int(chunk_start * SAMPLE_RATE)
                sample_end = int(chunk_end * SAMPLE_RATE)
                audio_chunk = audio[sample_start:sample_end]

                if len(audio_chunk) < int(0.1 * SAMPLE_RATE):
                    break

                g = compute_gammatone(audio_chunk, SAMPLE_RATE)
                s = compute_sonority(g)
                regions = detect_syllable_regions(s)

                if len(regions) == 0:
                    chunk_start += step
                    continue

                regions["start"] += chunk_start
                regions["end"] += chunk_start
                regions["duration"] = regions["end"] - regions["start"]

                # Keep only the core portion of overlapping chunks.
                core_start = (
                    chunk_start
                    if chunk_start == 0
                    else chunk_start + CHUNK_OVERLAP
                )

                regions = regions[
                    regions["start"] >= core_start
                ].copy()

                if chunk_end < audio_duration:
                    regions = regions[
                        regions["end"] <= chunk_end
                    ].copy()

                for _, region in regions.iterrows():
                    r_start = float(region["start"])
                    r_end = float(region["end"])

                    vec = extractor.extract_segment(
                        acoustic,
                        r_start,
                        r_end,
                    )

                    label, ann_start, ann_end, overlap_duration, overlap_ratio = assign_label(
                        r_start,
                        r_end,
                        file_annotations,
                    )

                    records.append(
                        FeatureRecord(
                            file=stem,
                            start=r_start,
                            end=r_end,
                            label=label,
                            duration=r_end - r_start,
                            has_overlap=overlap_duration > 0,
                            features=vec,
                        )
                    )

                print(
                    f"    Regions in chunk: {len(regions)} | "
                    f"file total: {len(records) - file_records_start}",
                    flush=True,
                )

                chunk_start += step

            file_count = len(records) - file_records_start
            elapsed = time.time() - file_start_time
            print(
                f"  ✓ Completed: {file_count} regions in {elapsed:.1f}s",
                flush=True,
            )

        except Exception as exc:
            print(
                f"  ✗ FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )
            failures.append({
                "file": stem,
                "start": 0.0,
                "end": 0.0,
                "label": "",
                "reason": repr(exc),
            })

    if not records:
        raise RuntimeError("No features extracted.")

    print("\n[Step 3] Removing duplicate regions ...", flush=True)
    unique_records: List[FeatureRecord] = []
    seen = set()

    for record in records:
        key = (
            record.file,
            round(record.start, 6),
            round(record.end, 6),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_records.append(record)

    records = unique_records
    print(f"Unique regions: {len(records)}", flush=True)

    print("\n[Step 4] Saving standardized output ...", flush=True)
    X, labels, files = save_outputs(
        records,
        OUTPUT_DIR,
        failures,
    )

    print("\n" + "=" * 72, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(f"Feature matrix : {X.shape} (N x D)", flush=True)
    print(f"Feature dim    : {X.shape[1]}", flush=True)
    print(f"Feature names  : {ProsodicAcousticExtractor.FEATURE_NAMES}", flush=True)
    print(f"NaN            : {np.isnan(X).any()}", flush=True)
    print(f"Inf            : {np.isinf(X).any()}", flush=True)
    print(f"Files          : {len(set(files.tolist()))}", flush=True)
    print(f"Failures       : {len(failures)}", flush=True)

    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:", flush=True)
    for label, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {label:<10} {count:>7}", flush=True)

    print(f"Total runtime : {time.time() - t0:.1f}s", flush=True)
    print("=" * 72, flush=True)

# ==========================================================================
# SECTION 10 — CLASSIFIERS
# ==========================================================================

"""
ied_prosodic_acoustic_classifier.py
Speech Disfluency Classification — OOP Pipeline
================================================
Prosodic + Acoustic region/syllable feature classifier.

Input:
    /content/prosodic_acoustic_output_safe/prosodic_acoustic_features.npz

Feature dimension:
    17

Classifiers:
    Random Forest
    DNN
    BiLSTM

Experiments:
    Exp1 — Fluent vs Disfluent
    Exp2 — Fluent vs each disfluency class (6 binary tasks)
    Exp3 — 6-class disfluency

Primary metric:
    Weighted-F1

SANITY_RUN=True is used first to verify shapes, splits, labels and model logic.
Set SANITY_RUN = True only after the sanity run passes.
"""

# ============================================================
# SECTION 0 — CONFIGURATION
# ============================================================

import os

NPZ_PATH = "/content/drive/MyDrive/prosodic_acoustic_output_safe/prosodic_acoustic_features_fixed.npz"
OUTPUT_DIR = "/content/drive/MyDrive/prosodic_acoustic_classifier_output"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
RANDOM_SEED = 42

# Reuse an externally supplied shared split when desired.
DEFAULT_SPLITS_PATH = os.path.join(OUTPUT_DIR, "splits", "splits.json")
SHARED_SPLITS_PATH = os.environ.get("SHARED_SPLITS_PATH", DEFAULT_SPLITS_PATH)

# Training caps: only training data are capped.
MAX_TRAIN_SAMPLES_PER_CLASS_BINARY = 75_000
MAX_TRAIN_SAMPLES_PER_CLASS_6CLASS = 40_000

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 20

# DNN
BATCH_SIZE = 256
DNN_EPOCHS = 60
LEARNING_RATE = 1e-3
EARLY_STOP_PAT = 8

# BiLSTM
BILSTM_EPOCHS = 60
SEQ_LEN = 128
BILSTM_BATCH = 16

# Sanity run: True = quick check; False = full experiments.
SANITY_RUN = False
SANITY_N_FILES = 3
SANITY_RF_TREES = 2
SANITY_EPOCHS = 2

FORCE_RECOMPUTE = False

META_DIR   = os.path.join(OUTPUT_DIR, "metadata")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
SPLIT_DIR  = os.path.join(OUTPUT_DIR, "splits")

FEATURE_DIM = 17

FEATURE_NAMES = [
    "duration",
    "f0_mean", "f0_std", "f0_min", "f0_max", "f0_skew",
    "f0_tilt", "f0_tilt_mean", "f0_tilt_std", "f0_tilt_skew",
    "energy_mean", "energy_std", "energy_skew",
    "energy_tilt_mean", "energy_tilt_std", "energy_tilt_skew",
    "pause_after",
]

DISF_CLASSES = ["I", "PR", "PhR", "WR", "PWR", "P"]

ALL_RESULTS = []


# ============================================================
# SECTION 1 — IMPORTS
# ============================================================

import json
import time
import random
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DEVICE = None


# ============================================================
# SECTION 2 — REPRODUCIBILITY / DIRECTORIES
# ============================================================

def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def apply_sanity_overrides() -> None:
    global RF_N_ESTIMATORS, RF_MAX_DEPTH, DNN_EPOCHS, BILSTM_EPOCHS
    global MAX_TRAIN_SAMPLES_PER_CLASS_BINARY, MAX_TRAIN_SAMPLES_PER_CLASS_6CLASS

    if SANITY_RUN:
        RF_N_ESTIMATORS = SANITY_RF_TREES
        RF_MAX_DEPTH = 5
        DNN_EPOCHS = SANITY_EPOCHS
        BILSTM_EPOCHS = SANITY_EPOCHS
        MAX_TRAIN_SAMPLES_PER_CLASS_BINARY = 500
        MAX_TRAIN_SAMPLES_PER_CLASS_6CLASS = 200

        print("=" * 65)
        print("  *** SANITY_RUN = True ***")
        print(f"  {SANITY_N_FILES} recordings, {SANITY_EPOCHS} epochs, "
              f"{SANITY_RF_TREES} RF trees")
        print("  Primary metric: Weighted-F1")
        print("=" * 65)


def make_dirs() -> None:
    for d in [OUTPUT_DIR, META_DIR, MODEL_DIR, RESULT_DIR, LOG_DIR, SPLIT_DIR]:
        os.makedirs(d, exist_ok=True)


# ============================================================
# SECTION 3 — DATACLASSES
# ============================================================

@dataclass
class DatasetStore:
    X: np.ndarray
    labels: np.ndarray
    files: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    durations: np.ndarray
    has_overlap: np.ndarray
    feature_names: List[str]
    extractor_name: str

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    @property
    def feature_dim(self) -> int:
        return self.X.shape[1]

    @property
    def file_ids(self) -> List[str]:
        return pd.Series(self.files).unique().tolist()


@dataclass
class ClassificationTask:
    name: str
    label_map: Dict[str, int]
    excl_labels_rfdnn: Any
    excl_labels_bl: Any
    class_names: List[str]
    n_classes: int
    task_type: str
    max_per_class: int


@dataclass
class EvaluationResult:
    task_name: str
    classifier_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    precision_macro: float
    recall_macro: float
    confusion_matrix: Any
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# SECTION 4 — NPZ DATA LOADER
# ============================================================

class ProsodicAcousticLoader:
    """
    Loads the already-generated standardized 17-D Prosodic Acoustic NPZ.
    No feature extraction is performed here.
    """

    REQUIRED_KEYS = {
        "X", "labels", "files", "starts", "ends",
        "durations", "has_overlap", "feature_names", "extractor_name"
    }

    def load(self, path: str = NPZ_PATH) -> DatasetStore:
        if not os.path.exists(path):
            raise FileNotFoundError(f"NPZ not found: {path}")

        data = np.load(path, allow_pickle=True)
        missing = self.REQUIRED_KEYS - set(data.files)
        if missing:
            raise KeyError(f"Missing NPZ keys: {sorted(missing)}")

        X = np.asarray(data["X"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=object)
        files = np.asarray(data["files"], dtype=object)
        starts = np.asarray(data["starts"], dtype=np.float32)
        ends = np.asarray(data["ends"], dtype=np.float32)
        durations = np.asarray(data["durations"], dtype=np.float32)
        has_overlap = np.asarray(data["has_overlap"], dtype=bool)
        feature_names = [str(x) for x in data["feature_names"].tolist()]
        extractor_name = str(data["extractor_name"].item())

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.shape}")
        if X.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected {FEATURE_DIM} features, got {X.shape[1]}"
            )

        n = len(labels)
        lengths = {
            "labels": len(labels), "files": len(files),
            "starts": len(starts), "ends": len(ends),
            "durations": len(durations), "has_overlap": len(has_overlap),
            "X": len(X)
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"NPZ arrays have inconsistent lengths: {lengths}")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values.")

        if feature_names != FEATURE_NAMES:
            print("WARNING: Feature names differ from the expected 17-D schema.")
            print("NPZ:", feature_names)
            print("Expected:", FEATURE_NAMES)

        return DatasetStore(
            X=X,
            labels=labels,
            files=files,
            starts=starts,
            ends=ends,
            durations=durations,
            has_overlap=has_overlap,
            feature_names=feature_names,
            extractor_name=extractor_name,
        )


# ============================================================
# SECTION 5 — FILE / SPEAKER IDENTIFICATION
# ============================================================

def parse_speaker_id(file_id: str) -> str:
    """
    Match the filename conventions used by the shared Wav2Vec2 pipeline.

    A:
        IED_INTERN_1__speaker_1
        -> A_spk_1

    B:
        intern1__x_y_Speaker_1_r
        -> B_spk_1

    Unknown naming falls back to the recording itself.
    """
    import re

    m = re.match(
        r"^IED_INTERN_(?P<n>\d+)__speaker_(?P<s>\d+)$",
        str(file_id), re.IGNORECASE
    )
    if m:
        return f"A_spk_{m.group('s')}"

    m = re.match(
        r"^intern(?P<n>\d+)__(?P<x>\d+)_(?P<y>\d+)_Speaker_(?P<s>\d+)_(?P<r>\d+)$",
        str(file_id), re.IGNORECASE
    )
    if m:
        return f"B_spk_{m.group('s')}"

    return f"unknown_spk_{file_id}"


def build_file_info(store: DatasetStore) -> pd.DataFrame:
    df = pd.DataFrame({
        "file_id": store.files.astype(str),
        "label": store.labels.astype(str),
    })

    file_info = (
        df.groupby("file_id")
        .agg(
            n_samples=("label", "size"),
            n_labels=("label", "nunique"),
        )
        .reset_index()
    )
    file_info["speaker_id"] = file_info["file_id"].map(parse_speaker_id)
    return file_info


# ============================================================
# SECTION 6 — SPEAKER-LEVEL SPLIT MANAGER
# ============================================================

class SplitManager:
    """
    Greedy 70/15/15 speaker-level split.
    Existing valid splits are reused.
    """

    def __init__(self, splits_json_path: str = SHARED_SPLITS_PATH):
        self.splits_json_path = splits_json_path

    def get_or_create_splits(self, file_info_df: pd.DataFrame) -> dict:
        all_files = set(file_info_df["file_id"].tolist())

        if os.path.exists(self.splits_json_path):
            with open(self.splits_json_path) as f:
                saved = json.load(f)

            saved_files = set(
                saved.get("train_files", []) +
                saved.get("val_files", []) +
                saved.get("test_files", [])
            )

            if saved_files == all_files:
                print(f"SplitManager: loaded existing splits from "
                      f"{self.splits_json_path}")
                self._verify_no_leakage(saved)
                return saved

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
            file_info_df.groupby("speaker_id")
            .agg(
                n_samples=("n_samples", "sum"),
                files=("file_id", list)
            )
            .reset_index()
            .sort_values("n_samples", ascending=False)
            .reset_index(drop=True)
        )

        total = int(spk_stats["n_samples"].sum())
        target_train = int(total * TRAIN_RATIO)
        target_val = int(total * VAL_RATIO)

        train_spk, val_spk, test_spk = [], [], []
        train_n = val_n = 0

        for _, row in spk_stats.iterrows():
            spk = row["speaker_id"]
            n = int(row["n_samples"])

            rem_train = target_train - train_n
            rem_val = target_val - val_n

            if rem_train >= rem_val and rem_train > 0:
                train_spk.append(spk)
                train_n += n
            elif rem_val > 0:
                val_spk.append(spk)
                val_n += n
            else:
                test_spk.append(spk)

        if not val_spk and len(train_spk) > 1:
            val_spk.append(train_spk.pop(-1))
        if not test_spk and len(train_spk) > 1:
            test_spk.append(train_spk.pop(-1))

        def files_for(spks):
            return file_info_df[
                file_info_df["speaker_id"].isin(spks)
            ]["file_id"].tolist()

        splits = {
            "train_files": files_for(train_spk),
            "val_files": files_for(val_spk),
            "test_files": files_for(test_spk),
            "train_spk": train_spk,
            "val_spk": val_spk,
            "test_spk": test_spk,
        }

        self._verify_no_leakage(splits)
        return splits

    def _verify_no_leakage(self, splits: dict) -> None:
        tr = set(splits["train_spk"])
        va = set(splits["val_spk"])
        te = set(splits["test_spk"])

        assert not (tr & va), f"LEAK train∩val: {tr & va}"
        assert not (tr & te), f"LEAK train∩test: {tr & te}"
        assert not (va & te), f"LEAK val∩test: {va & te}"

        print("  Leakage check: train∩val=∅  train∩test=∅  "
              "val∩test=∅  OK")

    def print_split_table(self, splits: dict,
                          file_info_df: pd.DataFrame) -> None:
        rows = []

        for split_name, flist in [
            ("train", splits["train_files"]),
            ("val", splits["val_files"]),
            ("test", splits["test_files"])
        ]:
            sub = file_info_df[file_info_df["file_id"].isin(flist)]
            for spk, grp in sub.groupby("speaker_id"):
                rows.append({
                    "speaker": spk,
                    "split": split_name,
                    "files": len(grp),
                    "samples": grp["n_samples"].sum()
                })

        tbl = pd.DataFrame(rows)
        if tbl.empty:
            print("No split table available.")
            return

        print(tbl.to_string(index=False))
        totals = tbl.groupby("split")[["files", "samples"]].sum()
        totals["pct_samples"] = (
            totals["samples"] / totals["samples"].sum() * 100
        ).round(1)

        print("\nSplit totals:")
        print(totals)


# ============================================================
# SECTION 7 — DATA ARRAY UTILITIES
# ============================================================

def sample_train_data(X, y, max_per_class, rng=None):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    keep = []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.append(idx)

    if not keep:
        return X[:0], y[:0]

    keep = np.concatenate(keep)
    keep.sort()
    return X[keep], y[keep]


def rows_for_files(store: DatasetStore, file_list: list):
    file_set = set(map(str, file_list))
    mask = np.array([str(f) in file_set for f in store.files], dtype=bool)
    return mask


def build_rfdnn_arrays(
    store: DatasetStore,
    split: str,
    splits: dict,
    label_map: dict,
    exclude_labels=None,
    max_per_class: Optional[int] = None,
):
    if split == "train":
        files = splits["train_files"]
    elif split == "val":
        files = splits["val_files"]
    else:
        files = splits["test_files"]

    mask = rows_for_files(store, files)

    labels = store.labels.astype(str)
    allowed = labels.copy()

    if exclude_labels:
        allowed_mask = ~np.isin(allowed, list(exclude_labels))
        mask &= allowed_mask

    selected_labels = labels[mask]
    label_mask = np.array([lbl in label_map for lbl in selected_labels])
    mask_indices = np.where(mask)[0][label_mask]

    X = store.X[mask_indices]
    y_str = labels[mask_indices]
    y = np.array([label_map[lbl] for lbl in y_str], dtype=np.int64)

    if split == "train" and max_per_class is not None:
        X, y = sample_train_data(X, y, max_per_class)

    return X, y, mask_indices


def build_bilstm_index(
    store: DatasetStore,
    file_list: list,
    label_map: dict,
    exclude_labels_as_ignore=None,
    seq_len: int = SEQ_LEN,
):
    """
    Chronological sequence windows within each recording.
    The NPZ rows are already ordered by file/region from extraction.
    """

    if exclude_labels_as_ignore is None:
        exclude_labels_as_ignore = set()

    index = []
    file_set = set(map(str, file_list))
    all_files = pd.Series(store.files.astype(str)).unique().tolist()

    for fid in all_files:
        if fid not in file_set:
            continue

        row_idx = np.where(store.files.astype(str) == fid)[0]
        if len(row_idx) == 0:
            continue

        labels = store.labels[row_idx].astype(str)
        tgt_full = np.full(len(row_idx), -1, dtype=np.int64)

        for i, lbl in enumerate(labels):
            if lbl in label_map and lbl not in exclude_labels_as_ignore:
                tgt_full[i] = label_map[lbl]

        for start in range(0, len(row_idx), seq_len):
            end = min(start + seq_len, len(row_idx))
            actual = end - start

            target = np.full(seq_len, -1, dtype=np.int64)
            target[:actual] = tgt_full[start:end]

            index.append({
                "file_id": fid,
                "row_indices": row_idx[start:end].astype(np.int64),
                "length": actual,
                "target": target,
            })

    return index


def class_weight_tensor(y, n_classes, device=None):
    """Compute balanced weights for exactly n_classes classes."""
    if device is None:
        device = DEVICE

    y = np.asarray(y).ravel().astype(np.int64, copy=False)
    y = y[y >= 0]

    if len(y) == 0:
        return torch.ones(n_classes, dtype=torch.float32, device=device)

    invalid = y[y >= n_classes]
    if len(invalid):
        raise ValueError(
            f"Labels {np.unique(invalid).tolist()} are invalid for "
            f"a {n_classes}-class model."
        )

    present = np.unique(y)
    cw = np.ones(n_classes, dtype=np.float64)

    present_cw = compute_class_weight(
        class_weight="balanced",
        classes=present,
        y=y
    )

    for cls, weight in zip(present, present_cw):
        cw[int(cls)] = float(weight)

    return torch.tensor(cw, dtype=torch.float32, device=device)


def cw_from_index(index, n_classes, device=None):
    if not index:
        return torch.ones(n_classes, device=device or DEVICE)

    y = np.concatenate([e["target"] for e in index])
    return class_weight_tensor(y, n_classes, device)


# ============================================================
# SECTION 8 — METRICS
# ============================================================

def compute_metrics(
    y_true,
    y_pred,
    class_names,
    task="binary",
    y_prob=None,
    labels=None,
):
    """
    Weighted-F1 is the PRIMARY metric for this Prosodic Acoustic experiment.
    Macro-F1, accuracy, precision and recall are still reported.
    """

    _labels = labels if labels is not None else list(range(len(class_names)))

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="macro",
            zero_division=0, labels=_labels
        ),
        "recall": recall_score(
            y_true, y_pred, average="macro",
            zero_division=0, labels=_labels
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro",
            zero_division=0, labels=_labels
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted",
            zero_division=0, labels=_labels
        ),
        "conf_matrix": confusion_matrix(
            y_true, y_pred, labels=_labels
        ),
    }

    if task == "binary" and y_prob is not None:
        try:
            result["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            result["roc_auc"] = None

        try:
            result["pr_auc"] = average_precision_score(y_true, y_prob)
        except Exception:
            result["pr_auc"] = None

    per_f1 = f1_score(
        y_true, y_pred, average=None,
        zero_division=0, labels=_labels
    )

    for i, lbl in enumerate(class_names):
        result[f"f1_{lbl}"] = (
            float(per_f1[i]) if i < len(per_f1) else 0.0
        )

    return result


def save_result(experiment, model_name, metrics, class_names):
    row = {
        "experiment": experiment,
        "model": model_name,
        "accuracy": metrics.get("accuracy", 0),
        "macro_f1": metrics.get("macro_f1", 0),
        "weighted_f1": metrics.get("weighted_f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "class_names": ",".join(class_names),
    }

    if "roc_auc" in metrics:
        row["roc_auc"] = metrics["roc_auc"]
    if "pr_auc" in metrics:
        row["pr_auc"] = metrics["pr_auc"]

    ALL_RESULTS.append(row)


# ============================================================
# SECTION 9 — PYTORCH DATASETS
# ============================================================

class FrameDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class ProsodicSeqDataset(Dataset):
    def __init__(self, index, store: DatasetStore):
        self.index = index
        self.store = store

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e = self.index[i]
        actual = e["length"]

        X_seg = np.zeros(
            (SEQ_LEN, self.store.feature_dim),
            dtype=np.float32
        )

        rows = e["row_indices"]
        X_seg[:actual] = self.store.X[rows]

        return (
            torch.tensor(X_seg, dtype=torch.float32),
            torch.tensor(e["target"], dtype=torch.long),
            actual,
        )


def collate_seq(batch):
    seqs, labels, lens = zip(*batch)

    B = len(seqs)
    T = max(s.shape[0] for s in seqs)
    D = seqs[0].shape[1]

    Xp = torch.zeros(B, T, D)
    yp = torch.full((B, T), -1, dtype=torch.long)

    for i, (seq, lbl, ln) in enumerate(zip(seqs, labels, lens)):
        Xp[i, :ln] = seq[:ln]
        yp[i, :ln] = lbl[:ln]

    return Xp, yp, list(lens)


# ============================================================
# SECTION 10 — MODEL ARCHITECTURES
# ============================================================

class DNN(nn.Module):
    """
    Input: 17-D Prosodic Acoustic vector.
    Same basic DNN structure as the provided Wav2Vec2 classifier,
    with only the input dimension changed from 768 -> 17.
    """

    def __init__(
        self,
        input_dim=FEATURE_DIM,
        h1=256,
        h2=128,
        n_classes=2,
        drop=0.3,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class BiLSTMClassifier(nn.Module):
    """
    Input: (B, T, 17)
    Output: (B, T, n_classes)
    """

    def __init__(
        self,
        input_dim=FEATURE_DIM,
        h1=128,
        h2=64,
        n_classes=2,
        drop=0.3,
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_dim, h1,
            batch_first=True,
            bidirectional=True
        )
        self.drop1 = nn.Dropout(drop)

        self.lstm2 = nn.LSTM(
            h1 * 2, h2,
            batch_first=True,
            bidirectional=True
        )
        self.drop2 = nn.Dropout(drop)

        self.fc = nn.Linear(h2 * 2, n_classes)

    def forward(self, x):
        o, _ = self.lstm1(x)
        o = self.drop1(o)

        o, _ = self.lstm2(o)
        o = self.drop2(o)

        return self.fc(o)


# ============================================================
# SECTION 11 — CLASSIFIER WRAPPERS
# ============================================================

class BaseClassifier(ABC):

    @property
    @abstractmethod
    def name(self):
        ...

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    def save(self, path):
        joblib.dump(self, path)


class RandomForestWrapper(BaseClassifier):
    name = "RandomForest"

    def __init__(
        self,
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_SEED,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )
        self._model.fit(X_train, y_train)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def save(self, path):
        joblib.dump(self._model, path)


class DNNWrapper(BaseClassifier):
    name = "DNN"

    def __init__(
        self,
        input_dim=FEATURE_DIM,
        n_classes=2,
        lr=LEARNING_RATE,
        n_epochs=DNN_EPOCHS,
        patience=EARLY_STOP_PAT,
    ):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self._model = None
        self.history = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        cw = class_weight_tensor(y_train, self.n_classes)

        self._model = DNN(
            self.input_dim,
            n_classes=self.n_classes
        ).to(DEVICE)

        dl_tr = DataLoader(
            FrameDataset(X_train, y_train),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=(len(X_train) >= BATCH_SIZE),
        )

        dl_va = (
            DataLoader(
                FrameDataset(X_val, y_val),
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
            if X_val is not None and len(X_val)
            else None
        )

        self._model, self.history, _ = self._train_loop(
            self._model, dl_tr, dl_va, cw
        )

    def _train_loop(self, model, dl_tr, dl_va, cw):
        opt = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, "max", factor=0.5, patience=3
        )
        crit = nn.CrossEntropyLoss(weight=cw)

        best_f1 = -1.0
        best_state = None
        wait = 0
        hist = []

        for ep in range(1, self.n_epochs + 1):
            model.train()
            tr_loss = 0.0
            n_seen = 0

            for Xb, yb in dl_tr:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

                opt.zero_grad()
                loss = crit(model(Xb), yb)
                loss.backward()
                opt.step()

                tr_loss += loss.item() * len(yb)
                n_seen += len(yb)

            tr_loss /= max(n_seen, 1)

            wf1 = 0.0

            if dl_va is not None:
                model.eval()
                pv, tv = [], []

                with torch.inference_mode():
                    for Xb, yb in dl_va:
                        pred = model(Xb.to(DEVICE)).argmax(1)
                        pv.extend(pred.cpu().tolist())
                        tv.extend(yb.tolist())

                wf1 = f1_score(
                    tv, pv,
                    average="weighted",
                    zero_division=0
                )

            # PRIMARY METRIC: Weighted-F1
            sched.step(wf1)

            hist.append({
                "ep": ep,
                "loss": tr_loss,
                "val_weighted_f1": wf1
            })

            if wf1 > best_f1:
                best_f1 = wf1
                best_state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1

            if ep % 10 == 0 or ep <= 3:
                print(
                    f"    ep{ep:3d}  loss={tr_loss:.4f}  "
                    f"val_weighted_f1={wf1:.4f}  best={best_f1:.4f}"
                )

            if wait >= self.patience:
                print(
                    f"    Early stop ep{ep} "
                    f"(best Weighted-F1={best_f1:.4f})"
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, hist, best_f1

    def predict(self, X):
        self._model.eval()

        ds = FrameDataset(
            X,
            np.zeros(len(X), dtype=np.int64)
        )
        dl = DataLoader(
            ds,
            batch_size=BATCH_SIZE * 4,
            shuffle=False
        )

        pred = []

        with torch.inference_mode():
            for Xb, _ in dl:
                pred.extend(
                    self._model(Xb.to(DEVICE))
                    .argmax(1)
                    .cpu()
                    .tolist()
                )

        return np.array(pred)

    def predict_proba(self, X):
        self._model.eval()

        ds = FrameDataset(
            X,
            np.zeros(len(X), dtype=np.int64)
        )
        dl = DataLoader(
            ds,
            batch_size=BATCH_SIZE * 4,
            shuffle=False
        )

        probs = []

        with torch.inference_mode():
            for Xb, _ in dl:
                probs.append(
                    torch.softmax(
                        self._model(Xb.to(DEVICE)), 1
                    ).cpu().numpy()
                )

        return np.vstack(probs)

    def save(self, path):
        if self._model is not None:
            torch.save(self._model.state_dict(), path)


class BiLSTMWrapper(BaseClassifier):
    name = "BiLSTM"

    def __init__(
        self,
        input_dim=FEATURE_DIM,
        n_classes=2,
        lr=LEARNING_RATE,
        n_epochs=BILSTM_EPOCHS,
        patience=EARLY_STOP_PAT,
    ):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self._model = None
        self.history = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError(
            "Use fit_sequences() for BiLSTM."
        )

    def fit_sequences(self, idx_tr, idx_va, store, n_classes):
        self.n_classes = n_classes

        cw = cw_from_index(idx_tr, n_classes)

        self._model = BiLSTMClassifier(
            self.input_dim,
            n_classes=n_classes
        ).to(DEVICE)

        dl_tr = DataLoader(
            ProsodicSeqDataset(idx_tr, store),
            batch_size=BILSTM_BATCH,
            shuffle=True,
            collate_fn=collate_seq,
        )

        dl_va = DataLoader(
            ProsodicSeqDataset(idx_va, store),
            batch_size=BILSTM_BATCH,
            shuffle=False,
            collate_fn=collate_seq,
        )

        self._model, self.history, _ = self._train_loop(
            self._model, dl_tr, dl_va, cw
        )

    def _train_loop(self, model, dl_tr, dl_va, cw):
        opt = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )

        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, "max", factor=0.5, patience=3
        )

        crit = nn.CrossEntropyLoss(
            weight=cw,
            ignore_index=-1
        )

        best_f1 = -1.0
        best_state = None
        wait = 0
        hist = []

        for ep in range(1, self.n_epochs + 1):
            model.train()
            tr_loss = 0.0
            n_batches = 0

            for Xb, yb, lens in dl_tr:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

                opt.zero_grad()

                logits = model(Xb)
                B, T, C = logits.shape

                loss = crit(
                    logits.reshape(B * T, C),
                    yb.reshape(B * T)
                )

                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )

                opt.step()

                tr_loss += loss.item()
                n_batches += 1

            tr_loss /= max(n_batches, 1)

            model.eval()
            pv, tv = [], []

            with torch.inference_mode():
                for Xb, yb, lens in dl_va:
                    logits = model(Xb.to(DEVICE))
                    pred = logits.argmax(2).cpu().numpy()

                    for i, ln in enumerate(lens):
                        true = yb[i, :ln].numpy()
                        mask = true >= 0

                        tv.extend(true[mask].tolist())
                        pv.extend(pred[i, :ln][mask].tolist())

            wf1 = (
                f1_score(
                    tv, pv,
                    average="weighted",
                    zero_division=0
                )
                if tv else 0.0
            )

            # PRIMARY METRIC: Weighted-F1
            sched.step(wf1)

            hist.append({
                "ep": ep,
                "loss": tr_loss,
                "val_weighted_f1": wf1
            })

            if wf1 > best_f1:
                best_f1 = wf1
                best_state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1

            if ep % 10 == 0 or ep <= 3:
                print(
                    f"    ep{ep:3d}  loss={tr_loss:.4f}  "
                    f"val_weighted_f1={wf1:.4f}  best={best_f1:.4f}"
                )

            if wait >= self.patience:
                print(
                    f"    Early stop ep{ep} "
                    f"(best Weighted-F1={best_f1:.4f})"
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, hist, best_f1

    def predict_sequences(self, idx_te, store):
        self._model.eval()

        dl = DataLoader(
            ProsodicSeqDataset(idx_te, store),
            batch_size=8,
            shuffle=False,
            collate_fn=collate_seq,
        )

        all_pred = []
        all_true = []
        all_prob = []

        with torch.inference_mode():
            for Xb, yb, lens in dl:
                logits = self._model(Xb.to(DEVICE))
                prob = torch.softmax(logits, 2).cpu().numpy()
                pred = logits.argmax(2).cpu().numpy()

                for i, ln in enumerate(lens):
                    true = yb[i, :ln].numpy()
                    mask = true >= 0

                    all_true.extend(true[mask].tolist())
                    all_pred.extend(
                        pred[i, :ln][mask].tolist()
                    )
                    all_prob.append(
                        prob[i, :ln][mask]
                    )

        y_true = np.array(all_true)
        y_pred = np.array(all_pred)

        y_prob = (
            np.vstack(all_prob)
            if all_prob else
            np.zeros((0, self.n_classes))
        )

        return y_true, y_pred, y_prob

    def predict(self, X):
        raise NotImplementedError(
            "Use predict_sequences() for BiLSTM."
        )

    def save(self, path):
        if self._model is not None:
            torch.save(
                self._model.state_dict(),
                path
            )


# ============================================================
# SECTION 12 — EVALUATOR
# ============================================================

class Evaluator:
    """
    Shared evaluator.
    PRIMARY METRIC: Weighted-F1.
    """

    def evaluate(
        self,
        y_true,
        y_pred,
        class_names,
        task_name,
        classifier_name,
        task="binary",
        y_prob=None,
        labels=None,
    ):
        metrics = compute_metrics(
            y_true,
            y_pred,
            class_names,
            task,
            y_prob,
            labels
        )

        return EvaluationResult(
            task_name=task_name,
            classifier_name=classifier_name,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            weighted_f1=metrics["weighted_f1"],
            precision_macro=metrics["precision"],
            recall_macro=metrics["recall"],
            confusion_matrix=metrics["conf_matrix"],
            extra={
                k: v for k, v in metrics.items()
                if k not in (
                    "accuracy",
                    "macro_f1",
                    "weighted_f1",
                    "precision",
                    "recall",
                    "conf_matrix",
                )
            },
        )

    def plot_confusion_matrix(
        self,
        cm,
        class_names,
        title,
        save_path=None
    ):
        fig, ax = plt.subplots(
            figsize=(
                max(5, len(class_names) * 1.3),
                max(4, len(class_names) * 1.1)
            )
        )

        im = ax.imshow(cm)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center",
                    va="center"
                )

        fig.colorbar(im, ax=ax)
        plt.tight_layout()

        if save_path:
            plt.savefig(
                save_path,
                dpi=110,
                bbox_inches="tight"
            )

        plt.close()

    def print_summary(self, result):
        print(
            f"  {result.classifier_name:<14} "
            f"{result.task_name:<30} "
            f"Acc={result.accuracy:.4f}  "
            f"Macro-F1={result.macro_f1:.4f}  "
            f"WEIGHTED-F1={result.weighted_f1:.4f} ⭐"
        )


# ============================================================
# SECTION 13 — CLASSIFIER RUNNER
# ============================================================

class ClassifierRunner:

    def __init__(
        self,
        store,
        splits,
        classifiers,
        evaluator,
        model_dir=MODEL_DIR,
        result_dir=RESULT_DIR,
    ):
        self.store = store
        self.splits = splits
        self.clfs = classifiers
        self.evaluator = evaluator
        self.model_dir = model_dir
        self.result_dir = result_dir
        self.results = {}

    def build_standard_tasks(self):
        tasks = []

        # ----------------------------------------------------
        # Exp1 — Fluent vs Disfluent
        # ----------------------------------------------------
        tasks.append(
            ClassificationTask(
                name="Exp1_FluDis",
                label_map={
                    "Fluent": 0,
                    "I": 1, "PR": 1, "PhR": 1,
                    "WR": 1, "PWR": 1, "P": 1,
                },
                excl_labels_rfdnn={"AMBIGUOUS", "UNKNOWN"},
                excl_labels_bl={"AMBIGUOUS", "UNKNOWN"},
                class_names=["Fluent", "Disfluent"],
                n_classes=2,
                task_type="binary",
                max_per_class=MAX_TRAIN_SAMPLES_PER_CLASS_BINARY,
            )
        )

        # ----------------------------------------------------
        # Exp2 — Fluent vs each disfluency class
        # ----------------------------------------------------
        for target in DISF_CLASSES:
            other = set(DISF_CLASSES) - {target}

            tasks.append(
                ClassificationTask(
                    name=f"Exp2_Flu_{target}",
                    label_map={
                        "Fluent": 0,
                        target: 1,
                    },
                    excl_labels_rfdnn=(
                        other | {"AMBIGUOUS", "UNKNOWN"}
                    ),
                    excl_labels_bl=(
                        other | {"AMBIGUOUS", "UNKNOWN"}
                    ),
                    class_names=["Fluent", target],
                    n_classes=2,
                    task_type="binary",
                    max_per_class=MAX_TRAIN_SAMPLES_PER_CLASS_BINARY,
                )
            )

        # ----------------------------------------------------
        # Exp3 — Six-class disfluency
        # ----------------------------------------------------
        tasks.append(
            ClassificationTask(
                name="Exp3_6class",
                label_map={
                    "I": 0,
                    "PR": 1,
                    "PhR": 2,
                    "WR": 3,
                    "PWR": 4,
                    "P": 5,
                },
                excl_labels_rfdnn={
                    "Fluent", "AMBIGUOUS", "UNKNOWN"
                },
                excl_labels_bl={
                    "Fluent", "AMBIGUOUS", "UNKNOWN"
                },
                class_names=[
                    "I", "PR", "PhR", "WR", "PWR", "P"
                ],
                n_classes=6,
                task_type="multiclass",
                max_per_class=MAX_TRAIN_SAMPLES_PER_CLASS_6CLASS,
            )
        )

        return tasks

    def _run_rf_dnn(self, clf, task):
        X_tr, y_tr, _ = build_rfdnn_arrays(
            self.store,
            "train",
            self.splits,
            task.label_map,
            task.excl_labels_rfdnn,
            task.max_per_class,
        )

        X_va, y_va, _ = build_rfdnn_arrays(
            self.store,
            "val",
            self.splits,
            task.label_map,
            task.excl_labels_rfdnn,
        )

        X_te, y_te, _ = build_rfdnn_arrays(
            self.store,
            "test",
            self.splits,
            task.label_map,
            task.excl_labels_rfdnn,
        )

        c_tr = Counter(y_tr.tolist())
        c_te = Counter(y_te.tolist())

        for ci, cn in enumerate(task.class_names):
            print(
                f"    Train {cn}={c_tr.get(ci, 0):,}  "
                f"Test {cn}={c_te.get(ci, 0):,}"
            )

        if any(
            c_te.get(i, 0) == 0
            for i in range(task.n_classes)
        ):
            return None, None, None

        # DNN is reused across experiments. Its output dimension must
        # match the current task: 2 classes for Exp1/Exp2 and 6 for Exp3.
        if isinstance(clf, DNNWrapper):
            clf.n_classes = task.n_classes

        clf.fit(X_tr, y_tr, X_va, y_va)

        y_pred = clf.predict(X_te)
        y_prob = None

        if task.n_classes == 2 and hasattr(
            clf, "predict_proba"
        ):
            y_prob = clf.predict_proba(X_te)[:, 1]

        return y_te, y_pred, y_prob

    def _run_bilstm(self, clf, task):
        idx_tr = build_bilstm_index(
            self.store,
            self.splits["train_files"],
            task.label_map,
            task.excl_labels_bl,
            SEQ_LEN,
        )

        idx_va = build_bilstm_index(
            self.store,
            self.splits["val_files"],
            task.label_map,
            task.excl_labels_bl,
            SEQ_LEN,
        )

        idx_te = build_bilstm_index(
            self.store,
            self.splits["test_files"],
            task.label_map,
            task.excl_labels_bl,
            SEQ_LEN,
        )

        print(
            f"    Windows: train={len(idx_tr)}  "
            f"val={len(idx_va)}  test={len(idx_te)}"
        )

        if not idx_tr or not idx_va or not idx_te:
            return None, None, None

        clf.fit_sequences(
            idx_tr,
            idx_va,
            self.store,
            task.n_classes,
        )

        y_true, y_pred, y_prob = clf.predict_sequences(
            idx_te,
            self.store,
        )

        prob1d = (
            y_prob[:, 1]
            if task.n_classes == 2 and len(y_prob)
            else None
        )

        return y_true, y_pred, prob1d

    def run_task(self, task):
        print(
            f"\n{'=' * 65}\n"
            f"TASK: {task.name}  classes={task.class_names}\n"
            f"{'=' * 65}"
        )

        task_results = {}

        for clf in self.clfs:
            print(f"\n  -- {clf.name} --")
            t0 = time.time()

            if isinstance(clf, BiLSTMWrapper):
                y_true, y_pred, y_prob = self._run_bilstm(
                    clf, task
                )
            else:
                y_true, y_pred, y_prob = self._run_rf_dnn(
                    clf, task
                )

            if y_true is None:
                print("    Skipped — insufficient class/split data.")
                continue

            labels_idx = list(range(task.n_classes))

            metrics = compute_metrics(
                y_true,
                y_pred,
                task.class_names,
                task.task_type,
                y_prob if task.n_classes == 2 else None,
                labels=labels_idx,
            )

            result = EvaluationResult(
                task_name=task.name,
                classifier_name=clf.name,
                accuracy=metrics["accuracy"],
                macro_f1=metrics["macro_f1"],
                weighted_f1=metrics["weighted_f1"],
                precision_macro=metrics["precision"],
                recall_macro=metrics["recall"],
                confusion_matrix=metrics["conf_matrix"],
                extra={
                    k: v for k, v in metrics.items()
                    if k not in (
                        "accuracy",
                        "macro_f1",
                        "weighted_f1",
                        "precision",
                        "recall",
                        "conf_matrix",
                    )
                },
            )

            self.evaluator.print_summary(result)

            save_result(
                task.name,
                clf.name,
                metrics,
                task.class_names
            )

            task_results[clf.name] = metrics

            if isinstance(clf, RandomForestWrapper):
                sub = "rf"
                ext = ".pkl"
            elif isinstance(clf, DNNWrapper):
                sub = "dnn"
                ext = ".pt"
            else:
                sub = "bilstm"
                ext = ".pt"

            clf.save(
                os.path.join(
                    self.model_dir,
                    f"{sub}_{task.name}{ext}"
                )
            )

            self.evaluator.plot_confusion_matrix(
                metrics["conf_matrix"],
                task.class_names,
                f"{task.name} — {clf.name}",
                save_path=os.path.join(
                    self.result_dir,
                    f"cm_{task.name}_{clf.name}.png"
                ),
            )

            print(
                f"    Done {time.time() - t0:.1f}s  "
                f"WEIGHTED-F1={metrics['weighted_f1']:.4f}"
            )

        self.results[task.name] = task_results

    def run_all(self):
        tasks = self.build_standard_tasks()

        for task in tasks:
            self.run_task(task)

        return self.results

    def save_results(self, path):
        serialisable = {}

        for task_name, clf_dict in self.results.items():
            serialisable[task_name] = {}

            for clf_name, metrics in clf_dict.items():
                serialisable[task_name][clf_name] = {
                    k: (
                        v.tolist()
                        if isinstance(v, np.ndarray)
                        else v
                    )
                    for k, v in metrics.items()
                    if k != "conf_matrix"
                }

        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)

        print(f"Results saved to {path}")


# ============================================================
# SECTION 14 — VISUALIZATIONS / FINAL TABLE
# ============================================================

def plot_class_distribution(store, save_dir):
    counts = pd.Series(
        store.labels.astype(str)
    ).value_counts()

    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot(kind="bar", ax=ax)

    ax.set_title(
        "Prosodic Acoustic Region-Level Label Distribution"
    )
    ax.set_xlabel("Label")
    ax.set_ylabel("Samples")

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height()):,}",
            (
                p.get_x() + p.get_width() / 2,
                p.get_height()
            ),
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir,
            "class_distribution.png"
        ),
        bbox_inches="tight"
    )
    plt.close()


def plot_weighted_f1_comparison(all_results, save_dir):
    res = pd.DataFrame(all_results)

    if res.empty:
        return

    experiments = res["experiment"].unique()

    fig, ax = plt.subplots(
        figsize=(max(10, len(experiments) * 1.5), 5)
    )

    x = np.arange(len(experiments))
    width = 0.25

    for i, model in enumerate(
        ["RandomForest", "DNN", "BiLSTM"]
    ):
        sub = res[res["model"] == model]

        values = []

        for exp in experiments:
            r = sub[sub["experiment"] == exp]
            values.append(
                r["weighted_f1"].values[0]
                if len(r)
                else 0.0
            )

        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=model
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        experiments,
        rotation=40,
        ha="right",
        fontsize=7
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Weighted-F1")
    ax.set_title(
        "Weighted-F1 — All Experiments × All Classifiers"
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir,
            "weighted_f1_comparison.png"
        ),
        bbox_inches="tight"
    )
    plt.close()


def print_final_table(all_results):
    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        print("No results to show.")
        return results_df

    cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "precision",
        "recall",
    ]

    tbl = results_df[
        ["experiment", "model"] + cols
    ].copy()

    tbl[cols] = tbl[cols].round(4)

    print("=" * 95)
    print(
        "FINAL RESULTS TABLE "
        "(region-level | PRIMARY METRIC: Weighted-F1 ⭐)"
    )
    print("=" * 95)
    print(tbl.to_string(index=False))

    print("\n-- Best model per experiment by Weighted-F1 --")

    for exp in results_df["experiment"].unique():
        sub = results_df[
            results_df["experiment"] == exp
        ]

        best = sub.loc[
            sub["weighted_f1"].idxmax()
        ]

        print(
            f"  {exp:<32} -> "
            f"{best['model']:<14}  "
            f"Weighted-F1={best['weighted_f1']:.4f}"
        )

    return results_df


# ============================================================
# SECTION 15 — SANITY CHECKS
# ============================================================

def run_sanity_checks(store, splits, file_info):
    print("=" * 65)
    print("SANITY CHECKS")
    print("=" * 65)

    # 1. Feature matrix
    print("\n[1] Feature matrix")
    print(f"  X shape: {store.X.shape}")
    assert store.X.ndim == 2
    assert store.X.shape[1] == FEATURE_DIM
    assert np.isfinite(store.X).all()
    print("  17-D float32 features, no NaN/Inf — OK")

    # 2. Labels
    print("\n[2] Label set")
    counts = pd.Series(
        store.labels.astype(str)
    ).value_counts()

    print(counts.to_string())

    expected = {
        "Fluent", "I", "PR", "PhR",
        "WR", "PWR", "P"
    }

    assert set(counts.index).issubset(expected)
    print("  Labels match standardized class vocabulary — OK")

    # 3. Files
    print("\n[3] File inventory")
    print(f"  Recordings: {len(file_info)}")
    print(f"  Speakers:   {file_info['speaker_id'].nunique()}")
    assert len(file_info) > 0
    print("  OK")

    # 4. Speaker leakage
    print("\n[4] Speaker leakage")
    tr = set(splits["train_spk"])
    va = set(splits["val_spk"])
    te = set(splits["test_spk"])

    assert not tr & va
    assert not tr & te
    assert not va & te

    print(
        f"  train={len(tr)}  val={len(va)}  test={len(te)}"
    )
    print("  No speaker leakage — OK")

    # 5. Standard task construction
    print("\n[5] Classification tasks")
    dummy_runner = ClassifierRunner(
        store,
        splits,
        [],
        Evaluator()
    )

    tasks = dummy_runner.build_standard_tasks()

    print(f"  Number of tasks: {len(tasks)}")
    print(
        "  " +
        ", ".join(t.name for t in tasks)
    )

    assert len(tasks) == 8
    print("  Exp1 + 6×Exp2 + Exp3 — OK")

    # 6. DNN input
    print("\n[6] DNN input shape")
    dnn = DNN(
        input_dim=FEATURE_DIM,
        n_classes=2
    )

    xb = torch.randn(4, FEATURE_DIM)
    out = dnn(xb)

    print(
        f"  Input: {tuple(xb.shape)}  "
        f"Output: {tuple(out.shape)}"
    )

    assert out.shape == (4, 2)
    print("  17-D → DNN — OK")

    # 7. BiLSTM input
    print("\n[7] BiLSTM input shape")
    bl = BiLSTMClassifier(
        input_dim=FEATURE_DIM,
        n_classes=2
    )

    xb = torch.randn(2, SEQ_LEN, FEATURE_DIM)
    out = bl(xb)

    print(
        f"  Input: {tuple(xb.shape)}  "
        f"Output: {tuple(out.shape)}"
    )

    assert out.shape == (2, SEQ_LEN, 2)
    print("  (B,128,17) → BiLSTM → (B,128,2) — OK")

    # 8. Weighted-F1
    print("\n[8] Weighted-F1 metric")
    yt = np.array([0, 0, 0, 1, 1])
    yp = np.array([0, 0, 1, 1, 1])

    m = compute_metrics(
        yt,
        yp,
        ["Fluent", "Disfluent"],
        task="binary",
        labels=[0, 1]
    )

    print(
        f"  Accuracy={m['accuracy']:.4f}  "
        f"Macro-F1={m['macro_f1']:.4f}  "
        f"Weighted-F1={m['weighted_f1']:.4f}"
    )

    assert "weighted_f1" in m
    print("  Weighted-F1 available as primary metric — OK")

    print("\n" + "=" * 65)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 65)


# ============================================================
# SECTION 16 — MAIN
# ============================================================

def run_classification():
    global DEVICE, ALL_RESULTS

    apply_sanity_overrides()
    make_dirs()
    set_seeds(RANDOM_SEED)

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {DEVICE}")

    ALL_RESULTS = []

    # --------------------------------------------------------
    # Load standardized NPZ
    # --------------------------------------------------------
    print("\n-- Prosodic Acoustic Dataset --")

    loader = ProsodicAcousticLoader()
    store = loader.load(NPZ_PATH)

    print(
        f"Samples       : {store.n_samples:,}"
    )
    print(
        f"Feature dim    : {store.feature_dim}"
    )
    print(
        f"Extractor      : {store.extractor_name}"
    )
    print(
        f"Recordings     : {len(store.file_ids)}"
    )

    print("\nFeature names:")
    print(store.feature_names)

    print("\nLabel distribution:")
    print(
        pd.Series(
            store.labels.astype(str)
        ).value_counts().to_string()
    )

    # --------------------------------------------------------
    # File inventory
    # --------------------------------------------------------
    file_info = build_file_info(store)

    file_info.to_csv(
        os.path.join(
            META_DIR,
            "file_inventory.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Speaker-level split
    # --------------------------------------------------------
    print("\n-- Speaker-Level Split --")

    split_mgr = SplitManager(
        SHARED_SPLITS_PATH
    )

    splits = split_mgr.get_or_create_splits(
        file_info
    )

    split_mgr.print_split_table(
        splits,
        file_info
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------
    print("\n-- Sanity Checks --")
    run_sanity_checks(
        store,
        splits,
        file_info
    )

    plot_class_distribution(
        store,
        RESULT_DIR
    )

    # --------------------------------------------------------
    # Important: SANITY_RUN stops before training.
    # --------------------------------------------------------
    if SANITY_RUN:
        print("\n" + "=" * 65)
        print("SANITY RUN COMPLETE — NO FULL TRAINING STARTED")
        print("=" * 65)
        print("If all checks passed, set:")
        print("    SANITY_RUN = False")
        print("and run again for the full experiment.")
        print("=" * 65)
        return

    # --------------------------------------------------------
    # Full classification
    # --------------------------------------------------------
    print("\n-- Classification --")

    classifiers = [
        RandomForestWrapper(
            RF_N_ESTIMATORS,
            RF_MAX_DEPTH
        ),
        DNNWrapper(
            FEATURE_DIM,
            n_classes=2,
            n_epochs=DNN_EPOCHS
        ),
        BiLSTMWrapper(
            FEATURE_DIM,
            n_classes=2,
            n_epochs=BILSTM_EPOCHS
        ),
    ]

    evaluator = Evaluator()

    runner = ClassifierRunner(
        store,
        splits,
        classifiers,
        evaluator,
        MODEL_DIR,
        RESULT_DIR,
    )

    print(
        "Running Exp1 + Exp2×6 + Exp3 ..."
    )

    runner.run_all()

    runner.save_results(
        os.path.join(
            RESULT_DIR,
            "all_results.json"
        )
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------
    results_df = print_final_table(
        ALL_RESULTS
    )

    results_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "all_results.csv"
        ),
        index=False
    )

    plot_weighted_f1_comparison(
        ALL_RESULTS,
        RESULT_DIR
    )

    # --------------------------------------------------------
    # Save config
    # --------------------------------------------------------
    config = {
        "SANITY_RUN": SANITY_RUN,
        "NPZ_PATH": NPZ_PATH,
        "OUTPUT_DIR": OUTPUT_DIR,
        "FEATURE_DIM": FEATURE_DIM,
        "FEATURE_NAMES": FEATURE_NAMES,
        "RANDOM_SEED": RANDOM_SEED,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "RF_N_ESTIMATORS": RF_N_ESTIMATORS,
        "RF_MAX_DEPTH": RF_MAX_DEPTH,
        "DNN_EPOCHS": DNN_EPOCHS,
        "BILSTM_EPOCHS": BILSTM_EPOCHS,
        "SEQ_LEN": SEQ_LEN,
        "MAX_TRAIN_BINARY": MAX_TRAIN_SAMPLES_PER_CLASS_BINARY,
        "MAX_TRAIN_6CLASS": MAX_TRAIN_SAMPLES_PER_CLASS_6CLASS,
        "PRIMARY_METRIC": "weighted_f1",
        "SHARED_SPLITS_PATH": SHARED_SPLITS_PATH,
    }

    with open(
        os.path.join(
            LOG_DIR,
            "config.json"
        ),
        "w"
    ) as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 65)
    print("DONE — Prosodic Acoustic OOP Classifier")
    print("=" * 65)
    print(f"Recordings   : {len(store.file_ids)}")
    print(f"Samples      : {store.n_samples:,}")
    print(f"Feature dim  : {store.feature_dim}")
    print(f"Primary metric: Weighted-F1")
    print(f"Results dir  : {RESULT_DIR}")
    print("=" * 65)




# ============================================================================
# SECTION 14 — EXTRACTOR -> CLASSIFIER BRIDGE
# ============================================================================

def prepare_classifier_npz():
    """
    Validate the extractor output and create the NaN-free NPZ consumed by
    the classifier. Only non-finite feature values are replaced with 0.0.
    """
    source = Path("/content/drive/MyDrive/prosodic_acoustic_output_safe/prosodic_acoustic_features.npz")
    fixed = Path("/content/drive/MyDrive/prosodic_acoustic_output_safe/prosodic_acoustic_features_fixed.npz")

    if not source.exists():
        raise FileNotFoundError(f"Extractor output not found: {source}")

    data = np.load(source, allow_pickle=True)
    required = {
        "X", "labels", "files", "starts", "ends",
        "durations", "has_overlap", "feature_names", "extractor_name"
    }
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"Extractor NPZ is missing keys: {sorted(missing)}")

    arrays = {k: data[k] for k in data.files}
    X = np.asarray(arrays["X"], dtype=np.float32)

    if X.ndim != 2 or X.shape[1] != 17:
        raise ValueError(f"Expected extractor matrix (N, 17), got {X.shape}")

    bad = ~np.isfinite(X)
    bad_count = int(bad.sum())
    if bad_count:
        print(f"  Replacing {bad_count:,} NaN/Inf feature values with 0.0")
        X = X.copy()
        X[bad] = 0.0

    arrays["X"] = X
    np.savez_compressed(fixed, **arrays)

    print(f"  Classifier-ready NPZ -> {fixed}")
    print(f"  Shape: {X.shape} | NaN: {np.isnan(X).sum()} | Inf: {np.isinf(X).sum()}")
    return fixed


# ============================================================================
# SECTION 15 — MAIN PIPELINE
# ============================================================================

def main():
    """
    Complete OOP pipeline, following the Wav2Vec2-style single-file layout:

        WAV + TXT
          -> Prosodic/Acoustic feature extraction
          -> standardized 17-D NPZ
          -> NaN/Inf validation
          -> speaker-level split
          -> RF + DNN + BiLSTM
          -> Weighted-F1-first evaluation + saved results
    """
    print("\n" + "=" * 78)
    print("  IED PROSODIC + ACOUSTIC COMPLETE OOP PIPELINE")
    print("=" * 78)
    print("  Stage 1: Feature Extraction")
    print("  Stage 2: Classification")
    print("  Primary metric: Weighted-F1")
    print("=" * 78 + "\n")

    extract_features_main()
    prepare_classifier_npz()
    run_classification()


if __name__ == "__main__":
    main()
