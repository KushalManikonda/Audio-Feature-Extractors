"""
whisper_feature_extractor.py
============================
IED Disfluency Dataset — Whisper Encoder Embedding Extraction
Follows the shared OOP design agreed by all feature-extractor interns.

Author  : Partha Kesav Reddy Chundi
Extractor : Whisper (medium) — mean-pooled encoder hidden states
Feature dim : 1024

Pipeline
--------
1.  AnnotationParser  — parse + normalise all .txt label files
2.  FluencyRegionFinder — mine fluent gaps; sample to balance dataset
3.  WhisperExtractor  — load model, run encoder, mean-pool per segment
4.  Save              — .npz (features) + .csv (metadata)

Output
------
  <OUTPUT_DIR>/whisper_features.npz
  <OUTPUT_DIR>/whisper_metadata.csv
  <OUTPUT_DIR>/whisper_failures.csv

Usage
-----
  python whisper_feature_extractor.py
"""

# ─── Standard library ─────────────────────────────────────────────────────────
import csv
import json
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import soundfile as sf
import torch
import whisper

# ==============================================================================
#  CONFIGURATION  — change paths here, nowhere else
# ==============================================================================

DATASET_DIR         = Path(r"e:\Whisper\IED Dataset")
OUTPUT_DIR          = Path(r"e:\Whisper\output")
MODEL_NAME          = "medium"          # whisper model size
MODEL_CACHE         = r"e:\whisper_models"
SAMPLE_RATE         = 16_000
FLUENT_CHUNK_SEC    = 1.0               # duration of each fluent chunk (seconds)
FLUENT_SAMPLE_SEED  = 42               # reproducibility seed for fluent sampling
TARGET_FLUENT_COUNT = 4170             # how many fluent segments to sample

VALID_LABELS = {"I", "PR", "PhR", "WR", "PWR", "P"}

# All observed non-standard label variants → canonical label
LABEL_MAP: Dict[str, str] = {
    "W.R": "WR",   "W.R.": "WR",  "WR": "WR",   "wr": "WR",
    "PWR": "PWR",  "PW.R": "PWR", "P.W.R": "PWR","PWr": "PWR",
    "PHR": "PhR",  "PH.R": "PhR", "Ph.r": "PhR", "Ph.R": "PhR", "PhR": "PhR",
    "PR":  "PR",   "Pr":   "PR",  "P.R":  "PR",  "P.r": "PR",
    "I":   "I",    "i":    "I",
    "P":   "P",
}


# ==============================================================================
#  DATACLASSES  — shared value objects used by every intern's extractor
# ==============================================================================

@dataclass
class AnnotationRow:
    """One labelled speech segment from a .txt annotation file."""
    file:        str
    start:       float
    end:         float
    label:       str          # canonical label: I, PR, PhR, WR, PWR, P, Fluent
    duration:    float
    raw_label:   str          # original label before normalisation
    source_line: int          # line number in the .txt file
    has_overlap: bool = False # True if this segment time-overlaps another


@dataclass
class FeatureRecord:
    """
    Result of running ANY feature extractor on ONE segment.
    The 'features' field is a 1-D array of shape (D,).
    D varies per method: 39 for MFCC, 1024 for Whisper, etc.
    """
    file:        str
    start:       float
    end:         float
    label:       str
    duration:    float
    has_overlap: bool
    features:    np.ndarray   # shape (D,)


@dataclass
class DatasetStore:
    """
    The complete extracted dataset ready for classification.
    Produced by BaseFeatureExtractor.extract_dataset().
    All interns output this same structure so classifiers are plug-and-play.
    """
    X:              np.ndarray    # (N, D)  all feature vectors
    labels:         np.ndarray    # (N,)    string labels
    files:          np.ndarray    # (N,)    source file stem per row
    starts:         np.ndarray    # (N,)    segment start times  [float64]
    ends:           np.ndarray    # (N,)    segment end times    [float64]
    durations:      np.ndarray    # (N,)    segment durations    [float32]
    has_overlap:    np.ndarray    # (N,)    bool flag
    feature_dim:    int           # D
    extractor_name: str           # e.g. "Whisper", "MFCC"


# ==============================================================================
#  AudioLoader
# ==============================================================================

class AudioLoader:
    """
    Loads a WAV file and returns a mono float32 numpy array.
    Handles stereo → mono by averaging channels.
    Used internally by BaseFeatureExtractor.
    """

    def __init__(self, sample_rate: int = 16_000):
        self.sample_rate = sample_rate

    def load(self, wav_path: Path) -> np.ndarray:
        """
        Load WAV → mono float32 array of shape (n_samples,).
        Raises ValueError if sample rate doesn't match config.
        """
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1)   # stereo → mono
        else:
            audio = audio[:, 0]
        if sr != self.sample_rate:
            raise ValueError(
                f"Expected {self.sample_rate} Hz, got {sr} Hz in {wav_path.name}"
            )
        return audio

    def slice_segment(self, audio: np.ndarray, start: float, end: float) -> np.ndarray:
        """
        Slice audio[start_sec : end_sec].
        Returns np.ndarray of shape (n_segment_samples,).
        """
        s = int(start * self.sample_rate)
        e = int(end   * self.sample_rate)
        return audio[s:e].copy()


# ==============================================================================
#  AnnotationParser
# ==============================================================================

class AnnotationParser:
    """
    Reads .txt annotation files, normalises labels, and marks overlapping
    segments.  Output is a list of AnnotationRow dataclass instances.

    Normalisation decisions (carried over from Stage 1):
      - Non-standard spellings   → mapped via LABEL_MAP
      - Parenthetical qualifiers → stripped (e.g. PR(Delta) → PR)
      - Compound labels (PR,I)   → split into two separate rows
      - Zero/negative duration   → dropped
    """

    def __init__(
        self,
        valid_labels: set = VALID_LABELS,
        label_map: Dict[str, str] = LABEL_MAP,
    ):
        self.valid_labels = valid_labels
        self.label_map    = label_map

    # ── public ────────────────────────────────────────────────────────────────

    def parse_dataset(self, dataset_dir: Path) -> List[AnnotationRow]:
        """
        Parse ALL .txt files in dataset_dir.
        Returns a flat List[AnnotationRow] for the entire dataset.
        """
        txt_files = sorted(dataset_dir.glob("*.txt"))
        all_rows: List[AnnotationRow] = []

        for txt_path in txt_files:
            rows, _ = self.parse_file(txt_path)
            all_rows.extend(rows)

        # Group by file and mark overlaps within each file
        by_file: Dict[str, List[AnnotationRow]] = {}
        for r in all_rows:
            by_file.setdefault(r.file, []).append(r)
        for file_rows in by_file.values():
            self.mark_overlaps(file_rows)

        print(f"[AnnotationParser] Parsed {len(all_rows)} rows from {len(txt_files)} files.")
        return all_rows

    def parse_file(
        self, txt_path: Path
    ) -> Tuple[List[AnnotationRow], List[str]]:
        """
        Parse one .txt annotation file.
        Returns (List[AnnotationRow], List[str] warnings).
        """
        rows: List[AnnotationRow] = []
        warnings: List[str] = []

        with open(txt_path, encoding="utf-8", errors="replace") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 3:
                    warnings.append(f"Line {line_no}: too few columns — skipped")
                    continue

                start_raw, end_raw, label_raw = parts[0], parts[1], parts[2]

                # Parse timestamps
                try:
                    start = float(start_raw)
                    end   = float(end_raw)
                except ValueError:
                    warnings.append(f"Line {line_no}: bad timestamp — skipped")
                    continue

                duration = round(end - start, 6)
                if duration <= 0:
                    warnings.append(f"Line {line_no}: zero/negative duration — dropped")
                    continue

                # Normalise label
                result, note = self.normalise_label(label_raw)
                if note:
                    warnings.append(f"Line {line_no}: {note}")

                # Compound label (e.g. "PR,I") → emit one row per sub-label
                if isinstance(result, list):
                    for canon in result:
                        rows.append(AnnotationRow(
                            file=txt_path.stem, start=start, end=end,
                            label=canon, duration=duration,
                            raw_label=label_raw, source_line=line_no,
                        ))
                    continue

                if result is None:
                    warnings.append(
                        f"Line {line_no}: unknown label '{label_raw}' — skipped"
                    )
                    continue

                rows.append(AnnotationRow(
                    file=txt_path.stem, start=start, end=end,
                    label=result, duration=duration,
                    raw_label=label_raw, source_line=line_no,
                ))

        return rows, warnings

    def normalise_label(self, raw: str):
        """
        Map a raw label string to its canonical form.
        Returns (canonical_label_or_list, note_string).
        Returns (None, note) for truly unknown labels.
        """
        raw = raw.strip().strip("]").strip("[").strip()

        # Split compound labels like "PR,I"
        if "," in raw:
            parts = [p.strip() for p in raw.split(",")]
            normed = []
            for p in parts:
                n, _ = self.normalise_label(p)
                if n:
                    normed.append(n)
            return normed, f"split compound '{raw}'"

        # Strip parenthetical qualifiers: (Delta), (d), (delta)
        paren_match = re.search(r"\(([^)]+)\)", raw)
        if paren_match:
            raw_base = raw[: paren_match.start()].strip()
        else:
            raw_base = raw

        # Exact map
        if raw_base in self.label_map:
            canon = self.label_map[raw_base]
            note  = f"normalised '{raw}' → '{canon}'" if raw != canon else ""
            return canon, note

        # Case-insensitive map
        raw_upper = raw_base.upper()
        for k, v in self.label_map.items():
            if k.upper() == raw_upper:
                return v, f"normalised (case) '{raw}' → '{v}'"

        return None, f"UNKNOWN '{raw}'"

    def mark_overlaps(self, rows: List[AnnotationRow]) -> None:
        """
        Set has_overlap=True on any AnnotationRow whose time interval
        overlaps another row in the same file.  Mutates in-place.
        """
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                ov_start = max(a.start, b.start)
                ov_end   = min(a.end,   b.end)
                if ov_end > ov_start:
                    rows[i].has_overlap = True
                    rows[j].has_overlap = True


# ==============================================================================
#  FluencyRegionFinder
# ==============================================================================

class FluencyRegionFinder:
    """
    Identifies fluent (non-disfluent) speech regions by finding gaps
    between annotated disfluency intervals, then creates AnnotationRows
    with label='Fluent' to balance the dataset.
    """

    def __init__(
        self,
        dataset_dir: Path,
        chunk_duration: float = FLUENT_CHUNK_SEC,
        target_count: int = TARGET_FLUENT_COUNT,
        random_seed: int = FLUENT_SAMPLE_SEED,
    ):
        self.dataset_dir    = dataset_dir
        self.chunk_duration = chunk_duration
        self.target_count   = target_count
        self.random_seed    = random_seed

    def find_and_sample(
        self, disfluent_rows: List[AnnotationRow]
    ) -> List[AnnotationRow]:
        """
        Main entry point.
        1. Build a map of disfluent intervals per file.
        2. Find gap regions in each file.
        3. Sample up to target_count fluent chunks.
        Returns List[AnnotationRow] with label='Fluent'.
        """
        # Group disfluent intervals by file
        file_to_intervals: Dict[str, List[Tuple[float, float]]] = {}
        for r in disfluent_rows:
            file_to_intervals.setdefault(r.file, []).append((r.start, r.end))

        candidates: List[AnnotationRow] = []
        for stem, intervals in file_to_intervals.items():
            wav_path = self.dataset_dir / f"{stem}.wav"
            if not wav_path.exists():
                continue
            try:
                info = sf.info(str(wav_path))
                total_dur = info.frames / info.samplerate
            except Exception as exc:
                print(f"[FluencyRegionFinder] Cannot read {stem}.wav: {exc}")
                continue

            chunks = self._find_fluent_chunks(stem, total_dur, sorted(intervals))
            candidates.extend(chunks)

        print(f"[FluencyRegionFinder] Found {len(candidates)} fluent candidate chunks.")

        if len(candidates) > self.target_count:
            random.seed(self.random_seed)
            candidates = random.sample(candidates, self.target_count)

        print(f"[FluencyRegionFinder] Sampled {len(candidates)} fluent segments.")
        return candidates

    # ── private ───────────────────────────────────────────────────────────────

    def _find_fluent_chunks(
        self,
        stem: str,
        total_duration: float,
        disfluent_intervals: List[Tuple[float, float]],
    ) -> List[AnnotationRow]:
        """
        Walk the timeline, collect gaps between disfluent intervals,
        and slice each gap into non-overlapping chunks of chunk_duration.
        """
        chunks: List[AnnotationRow] = []
        curr_time = 0.0

        for (s, e) in disfluent_intervals:
            gap = s - curr_time
            if gap >= self.chunk_duration:
                n = int(gap // self.chunk_duration)
                for i in range(n):
                    cs = curr_time + i * self.chunk_duration
                    chunks.append(self._make_fluent_row(stem, cs))
            curr_time = max(curr_time, e)

        # Tail gap after last disfluent segment
        gap = total_duration - curr_time
        if gap >= self.chunk_duration:
            n = int(gap // self.chunk_duration)
            for i in range(n):
                cs = curr_time + i * self.chunk_duration
                chunks.append(self._make_fluent_row(stem, cs))

        return chunks

    def _make_fluent_row(self, stem: str, chunk_start: float) -> AnnotationRow:
        return AnnotationRow(
            file=stem,
            start=chunk_start,
            end=chunk_start + self.chunk_duration,
            label="Fluent",
            duration=self.chunk_duration,
            raw_label="Fluent",
            source_line=-1,
            has_overlap=False,
        )


# ==============================================================================
#  BaseFeatureExtractor  — Abstract base class every intern implements
# ==============================================================================

class BaseFeatureExtractor(ABC):
    """
    Abstract base class for ALL feature extraction methods.

    To add a new extractor (MFCC, Wav2vec2, Conformer, etc.):
      1. Subclass BaseFeatureExtractor
      2. Set the `name` property
      3. Implement `setup()`
      4. Implement `extract_segment(audio, start, end) → np.ndarray (D,)`

    Everything else — file-level looping, error handling, saving, loading —
    is inherited from this class and is identical across all interns.
    """

    def __init__(self, sample_rate: int = 16_000):
        self.sample_rate  = sample_rate
        self.audio_loader = AudioLoader(sample_rate)

    # ── MUST OVERRIDE ─────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short identifier for this extractor.
        Used in console output and output filenames.
        Examples: "Whisper", "MFCC", "Wav2vec2", "Conformer"
        """
        ...

    @abstractmethod
    def setup(self) -> None:
        """
        Load model weights, initialise GPU, set eval mode, etc.
        Called ONCE before extract_dataset().
        For methods with no model (e.g. MFCC), this can be a no-op: pass
        """
        ...

    @abstractmethod
    def extract_segment(
        self, audio: np.ndarray, start: float, end: float
    ) -> np.ndarray:
        """
        Extract a fixed-size feature vector for ONE audio segment.

        Parameters
        ----------
        audio : np.ndarray, shape (n_samples,)
            Full mono float32 audio of one WAV file.
        start : float
            Segment start in seconds.
        end : float
            Segment end in seconds.

        Returns
        -------
        np.ndarray, shape (D,)
            A 1-D feature vector. D can be anything the method produces.
            Must NOT contain NaN or Inf.
        """
        ...

    # ── INHERITED — do NOT override ───────────────────────────────────────────

    def extract_dataset(
        self,
        rows: List[AnnotationRow],
        dataset_dir: Path,
    ) -> DatasetStore:
        """
        Run extract_segment() over all AnnotationRows.
        Loads each WAV file once, then processes all its segments.
        Skips failed segments gracefully and logs them.

        Returns: DatasetStore
        """
        # Group rows by file
        file_order: List[str] = []
        by_file: Dict[str, List[AnnotationRow]] = {}
        for r in rows:
            if r.file not in by_file:
                file_order.append(r.file)
                by_file[r.file] = []
            by_file[r.file].append(r)

        records: List[FeatureRecord] = []
        failures: List[dict] = []

        total_files = len(file_order)
        t_start = time.time()

        for fi, stem in enumerate(file_order, start=1):
            wav_path = dataset_dir / f"{stem}.wav"
            file_rows = by_file[stem]

            # Load audio once per file
            try:
                audio = self.audio_loader.load(wav_path)
            except Exception as exc:
                print(f"  [{fi}/{total_files}] SKIP FILE {stem}: {exc}")
                for r in file_rows:
                    failures.append(dict(
                        file=stem, start=r.start, end=r.end,
                        label=r.label, reason=f"audio load failed: {exc}",
                    ))
                continue

            ok, fail = 0, 0
            for r in file_rows:
                try:
                    vec = self.extract_segment(audio, r.start, r.end)
                    if np.isnan(vec).any() or np.isinf(vec).any():
                        raise ValueError("NaN/Inf in feature vector")
                    records.append(FeatureRecord(
                        file=r.file, start=r.start, end=r.end,
                        label=r.label, duration=r.duration,
                        has_overlap=r.has_overlap, features=vec,
                    ))
                    ok += 1
                except Exception as exc:
                    failures.append(dict(
                        file=stem, start=r.start, end=r.end,
                        label=r.label, reason=str(exc),
                    ))
                    fail += 1

            elapsed = time.time() - t_start
            rate = len(records) / elapsed if elapsed > 0 else 0
            print(
                f"  [{fi:>3}/{total_files}] {stem}  "
                f"ok={ok}  fail={fail}  total={len(records)}  "
                f"rate={rate:.1f} seg/s"
            )
            sys.stdout.flush()

        if not records:
            raise RuntimeError("No features extracted — check dataset path and audio files.")

        store = DatasetStore(
            X              = np.stack([r.features  for r in records]).astype(np.float32),
            labels         = np.array([r.label      for r in records]),
            files          = np.array([r.file       for r in records]),
            starts         = np.array([r.start      for r in records], dtype=np.float64),
            ends           = np.array([r.end        for r in records], dtype=np.float64),
            durations      = np.array([r.duration   for r in records], dtype=np.float32),
            has_overlap    = np.array([r.has_overlap for r in records], dtype=bool),
            feature_dim    = records[0].features.shape[0],
            extractor_name = self.name,
        )

        print(f"\n[{self.name}] Extracted {store.X.shape[0]} segments, "
              f"dim={store.feature_dim}, failures={len(failures)}")
        self._failures = failures   # stash for save()
        return store

    def save(
        self,
        store: DatasetStore,
        npz_path: Path,
        csv_path: Path,
        failures_csv_path: Optional[Path] = None,
    ) -> None:
        """
        Save DatasetStore to .npz (features) + .csv (metadata).
        Optionally save a failures CSV.
        Standard format shared by all interns.
        """
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            npz_path,
            embeddings  = store.X,
            labels      = store.labels,
            files       = store.files,
            starts      = store.starts,
            ends        = store.ends,
            durations   = store.durations,
            has_overlap = store.has_overlap,
        )
        print(f"[{self.name}] Saved features → {npz_path}  "
              f"({npz_path.stat().st_size / 1e6:.1f} MB)")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["idx", "file", "start", "end",
                               "label", "duration", "has_overlap"]
            )
            w.writeheader()
            for i in range(len(store.files)):
                w.writerow(dict(
                    idx=i, file=store.files[i],
                    start=store.starts[i], end=store.ends[i],
                    label=store.labels[i], duration=store.durations[i],
                    has_overlap=store.has_overlap[i],
                ))
        print(f"[{self.name}] Saved metadata  → {csv_path}")

        if failures_csv_path is not None:
            failures = getattr(self, "_failures", [])
            if failures:
                with open(failures_csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(
                        f, fieldnames=["file", "start", "end", "label", "reason"]
                    )
                    w.writeheader()
                    w.writerows(failures)
                print(f"[{self.name}] Saved failures  → {failures_csv_path}  "
                      f"({len(failures)} entries)")

    @staticmethod
    def load(npz_path: Path, csv_path: Path) -> DatasetStore:
        """
        Load a previously saved DatasetStore from .npz + .csv.
        """
        data = np.load(npz_path, allow_pickle=True)
        return DatasetStore(
            X              = data["embeddings"],
            labels         = data["labels"],
            files          = data["files"],
            starts         = data["starts"],
            ends           = data["ends"],
            durations      = data["durations"],
            has_overlap    = data["has_overlap"],
            feature_dim    = int(data["embeddings"].shape[1]),
            extractor_name = "loaded",
        )


# ==============================================================================
#  WhisperExtractor  — Concrete implementation for Whisper embeddings
# ==============================================================================

class WhisperExtractor(BaseFeatureExtractor):
    """
    Extracts mean-pooled Whisper encoder hidden states for each audio segment.

    Method
    ------
    For each segment [start, end]:
      1. Slice raw audio → segment
      2. Compute actual encoder frames:  n_frames = (n_samples // HOP) // STRIDE
      3. Pad/trim to exactly 30 s  (Whisper positional embedding is fixed to 1500 frames)
      4. Compute log-mel spectrogram  → (80, 3000)
      5. Run Whisper encoder          → (1, 1500, 1024)
      6. Slice [:n_frames_actual]     → (n_frames_actual, 1024)
      7. Mean-pool over time          → (1024,)   ← the feature vector

    Why mean-pool?
      Segments have variable duration (0.1 s – 34 s), so T varies.
      Mean-pooling collapses (T, 1024) → (1024,): a fixed-size embedding.
    """

    name = "Whisper"

    # Whisper internal constants
    _MEL_HOP       = 160    # mel spectrogram hop = 10 ms  @ 16 kHz
    _ENCODER_STRIDE = 2     # encoder conv stride → one frame per 320 samples (20 ms hop → 40 ms/frame)

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_dir: str  = MODEL_CACHE,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(sample_rate=sample_rate)
        self.model_name = model_name
        self.cache_dir  = cache_dir
        self._model     = None
        self._device    = None

    # ── BaseFeatureExtractor interface ────────────────────────────────────────

    def setup(self) -> None:
        """Load Whisper model onto GPU (or CPU if CUDA unavailable)."""
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_label = (
            torch.cuda.get_device_name(0) if self._device == "cuda" else "CPU"
        )
        print(f"[WhisperExtractor] Device: {self._device.upper()}  ({gpu_label})")
        print(f"[WhisperExtractor] Loading whisper-{self.model_name} …")
        t0 = time.time()
        self._model = whisper.load_model(
            self.model_name, device=self._device, download_root=self.cache_dir
        )
        self._model.eval()
        print(f"[WhisperExtractor] Loaded in {time.time() - t0:.1f} s  "
              f"| dim={self._model.dims.n_audio_state}  "
              f"layers={self._model.dims.n_audio_layer}")

    def extract_segment(
        self, audio: np.ndarray, start: float, end: float
    ) -> np.ndarray:
        """
        Extract a (1024,) mean-pooled Whisper encoder embedding for one segment.
        """
        if self._model is None:
            raise RuntimeError("Call setup() before extract_segment().")

        s = int(start * self.sample_rate)
        e = int(end   * self.sample_rate)
        segment = audio[s:e].copy()

        if len(segment) == 0:
            raise ValueError(f"Empty segment [{start:.3f}, {end:.3f}]")

        # Number of encoder frames the real segment maps to
        n_raw           = e - s
        n_mel_frames    = max(1, n_raw // self._MEL_HOP)
        n_frames_actual = max(1, n_mel_frames // self._ENCODER_STRIDE)

        # Pad to 30 s so encoder positional embedding matches
        segment_30s = whisper.pad_or_trim(segment)          # → (480_000,)

        mel = whisper.log_mel_spectrogram(
            segment_30s, n_mels=self._model.dims.n_mels
        ).to(self._device)                                   # → (80, 3000)

        with torch.no_grad():
            enc_out = self._model.encoder(mel.unsqueeze(0)) # → (1, 1500, 1024)

        # Slice to actual segment frames, then mean-pool
        enc_seq   = enc_out[0, :n_frames_actual, :].cpu().float().numpy()  # (T, 1024)
        embedding = enc_seq.mean(axis=0)                                    # (1024,)

        return embedding


# ==============================================================================
#  MAIN PIPELINE
# ==============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("  WHISPER FEATURE EXTRACTION PIPELINE")
    print("=" * 70)

    # ── Step 1: Parse disfluency annotations ──────────────────────────────────
    print("\n[Step 1] Parsing annotations …")
    parser        = AnnotationParser()
    disf_rows     = parser.parse_dataset(DATASET_DIR)

    # ── Step 2: Find and sample fluent regions ────────────────────────────────
    print("\n[Step 2] Finding fluent regions …")
    finder        = FluencyRegionFinder(DATASET_DIR)
    fluent_rows   = finder.find_and_sample(disf_rows)

    all_rows      = disf_rows + fluent_rows
    print(f"  Total segments (disfluent + fluent): {len(all_rows)}")

    # ── Step 3: Extract Whisper embeddings ────────────────────────────────────
    print("\n[Step 3] Extracting Whisper embeddings …")
    extractor = WhisperExtractor()
    extractor.setup()
    store = extractor.extract_dataset(all_rows, DATASET_DIR)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    print("\n[Step 4] Saving output …")
    extractor.save(
        store,
        npz_path       = OUTPUT_DIR / "whisper_features.npz",
        csv_path       = OUTPUT_DIR / "whisper_metadata.csv",
        failures_csv_path = OUTPUT_DIR / "whisper_failures.csv",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    label_counts = Counter(store.labels.tolist())

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Extractor       : {store.extractor_name}")
    print(f"  Feature matrix  : {store.X.shape}  (N × D)")
    print(f"  Feature dim (D) : {store.feature_dim}")
    print(f"  dtype           : {store.X.dtype}")
    print(f"  NaN             : {np.isnan(store.X).any()}")
    print(f"  Inf             : {np.isinf(store.X).any()}")
    print(f"  Label distribution:")
    for lbl in sorted(label_counts, key=lambda x: -label_counts[x]):
        pct = 100 * label_counts[lbl] / len(store.labels)
        print(f"    {lbl:<10}  {label_counts[lbl]:>5}  ({pct:.1f}%)")
    print()
    print(f"  Output NPZ  → {OUTPUT_DIR / 'whisper_features.npz'}")
    print(f"  Output CSV  → {OUTPUT_DIR / 'whisper_metadata.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
