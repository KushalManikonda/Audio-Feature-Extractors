"""
IED Disfluency Dataset — Prosodic + Acoustic Feature Extraction
================================================================
Standardized feature-extractor implementation for the IED project.

Follows the shared extractor layout used by the other interns:
    Annotation parsing -> segmentation -> feature extraction ->
    DatasetStore -> NPZ + metadata CSV + failures CSV

Feature vector (D=21):
    duration
    f0_mean, f0_std, f0_min, f0_max, f0_skew
    f0_tilt, f0_tilt_mean, f0_tilt_std, f0_tilt_skew
    energy_mean, energy_std, energy_skew
    energy_tilt_mean, energy_tilt_std, energy_tilt_skew
    pause_after

The segmentation logic is retained from the validated prosodic pipeline:
WAV -> Gammatone -> sonority -> syllable-like regions.

Run:
    python ied_prosodic_acoustic_extractor.py
"""

from __future__ import annotations

import csv
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import parselmouth
import soundfile as sf
from scipy.signal import bilinear, find_peaks, hilbert, lfilter, savgol_filter
from scipy.stats import skew
from gammatone.gtgram import gtgram

warnings.filterwarnings("ignore")

# ============================================================================== 
# CONFIGURATION — change paths here, nowhere else
# ============================================================================== 

DATASET_DIR = Path(r"e:\Whisper\IED Dataset")
OUTPUT_DIR = Path(r"e:\Whisper\output")
SAMPLE_RATE = 16_000

# Processing in chunks prevents large recordings from consuming excessive RAM.
CHUNK_DURATION = 30.0
CHUNK_OVERLAP = 0.5

# ============================================================================== 
# LABEL NORMALISATION
# ============================================================================== 

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

# ============================================================================== 
# DATACLASSES — shared output structure
# ============================================================================== 


@dataclass
class AnnotationRow:
    """One labelled interval from a TXT annotation file."""

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
    """One fixed-size prosodic/acoustic feature vector."""

    file: str
    start: float
    end: float
    label: str
    duration: float
    has_overlap: bool
    features: np.ndarray


@dataclass
class DatasetStore:
    """Complete extracted dataset ready for classification."""

    X: np.ndarray
    labels: np.ndarray
    files: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    durations: np.ndarray
    has_overlap: np.ndarray
    feature_dim: int
    extractor_name: str


# ============================================================================== 
# AUDIO LOADER
# ============================================================================== 


class AudioLoader:
    """Load WAV files as mono float32 arrays at the required sample rate."""

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


# ============================================================================== 
# ANNOTATION PARSER
# ============================================================================== 


class AnnotationParser:
    """Parse TXT annotations and normalise labels."""

    def __init__(
        self,
        valid_labels: set = VALID_LABELS,
        label_map: Dict[str, str] = LABEL_MAP,
    ):
        self.valid_labels = valid_labels
        self.label_map = label_map

    @staticmethod
    def _normalise_label(raw_label: str) -> Optional[str]:
        label = raw_label.strip()
        label = re.sub(r"\([^)]*\)", "", label).strip()

        # Compound annotation such as PR,I is handled by splitting in parse_file.
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

                # Preserve the established handling of compound labels.
                raw_parts = [p.strip() for p in raw_label.split(",") if p.strip()]
                if not raw_parts:
                    continue

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

        # Mark temporal overlap with another annotation.
        for i, row in enumerate(rows):
            for j, other in enumerate(rows):
                if i == j or row.file != other.file:
                    continue
                if row.start < other.end and other.start < row.end:
                    row.has_overlap = True
                    break

        return rows

    def parse_dataset(self, dataset_dir: Path) -> List[AnnotationRow]:
        rows: List[AnnotationRow] = []
        for txt_path in sorted(dataset_dir.rglob("*.txt")):
            rows.extend(self.parse_file(txt_path))
        return rows


# ============================================================================== 
# BASE FEATURE EXTRACTOR
# ============================================================================== 


class BaseFeatureExtractor:
    """Shared interface for feature extractors."""

    name = "Base"

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.audio_loader = AudioLoader(sample_rate)
        self._failures: List[dict] = []

    def setup(self) -> None:
        """Initialise resources. Prosodic extraction needs no model weights."""
        pass

    def extract_segment(
        self, audio: np.ndarray, start: float, end: float
    ) -> np.ndarray:
        raise NotImplementedError

    def extract_dataset(
        self,
        rows: List[AnnotationRow],
        dataset_dir: Path,
    ) -> DatasetStore:
        """Extract one fixed-size feature vector for every supplied segment."""

        file_order: List[str] = []
        by_file: Dict[str, List[AnnotationRow]] = {}

        for row in rows:
            if row.file not in by_file:
                file_order.append(row.file)
                by_file[row.file] = []
            by_file[row.file].append(row)

        records: List[FeatureRecord] = []
        failures: List[dict] = []
        total_files = len(file_order)
        t_start = time.time()

        for fi, stem in enumerate(file_order, start=1):
            wav_path = self._find_wav(dataset_dir, stem)
            if wav_path is None:
                for row in by_file[stem]:
                    failures.append(
                        dict(
                            file=stem,
                            start=row.start,
                            end=row.end,
                            label=row.label,
                            reason="WAV file not found",
                        )
                    )
                continue

            try:
                audio = self.audio_loader.load(wav_path)
            except Exception as exc:
                for row in by_file[stem]:
                    failures.append(
                        dict(
                            file=stem,
                            start=row.start,
                            end=row.end,
                            label=row.label,
                            reason=f"audio load failed: {exc}",
                        )
                    )
                continue

            ok = 0
            fail = 0
            for row in by_file[stem]:
                try:
                    vec = self.extract_segment(audio, row.start, row.end)
                    vec = np.asarray(vec, dtype=np.float32)

                    if vec.ndim != 1:
                        raise ValueError("Feature vector is not 1-D")
                    if np.isnan(vec).any() or np.isinf(vec).any():
                        raise ValueError("NaN/Inf in feature vector")

                    records.append(
                        FeatureRecord(
                            file=row.file,
                            start=row.start,
                            end=row.end,
                            label=row.label,
                            duration=row.duration,
                            has_overlap=row.has_overlap,
                            features=vec,
                        )
                    )
                    ok += 1
                except Exception as exc:
                    failures.append(
                        dict(
                            file=stem,
                            start=row.start,
                            end=row.end,
                            label=row.label,
                            reason=str(exc),
                        )
                    )
                    fail += 1

            elapsed = time.time() - t_start
            rate = len(records) / elapsed if elapsed > 0 else 0.0
            print(
                f"[{fi:>3}/{total_files}] {stem} "
                f"ok={ok} fail={fail} total={len(records)} rate={rate:.1f} seg/s"
            )

        if not records:
            raise RuntimeError("No features extracted — check dataset path and annotations.")

        store = DatasetStore(
            X=np.stack([r.features for r in records]).astype(np.float32),
            labels=np.array([r.label for r in records]),
            files=np.array([r.file for r in records]),
            starts=np.array([r.start for r in records], dtype=np.float64),
            ends=np.array([r.end for r in records], dtype=np.float64),
            durations=np.array([r.duration for r in records], dtype=np.float32),
            has_overlap=np.array([r.has_overlap for r in records], dtype=bool),
            feature_dim=records[0].features.shape[0],
            extractor_name=self.name,
        )

        self._failures = failures
        return store

    @staticmethod
    def _find_wav(dataset_dir: Path, stem: str) -> Optional[Path]:
        direct = dataset_dir / f"{stem}.wav"
        if direct.exists():
            return direct

        matches = list(dataset_dir.rglob(f"{stem}.wav"))
        return matches[0] if matches else None

    def save(
        self,
        store: DatasetStore,
        npz_path: Path,
        csv_path: Path,
        failures_csv_path: Optional[Path] = None,
    ) -> None:
        """Save standardized NPZ features, metadata CSV, and optional failures CSV."""

        npz_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            npz_path,
            embeddings=store.X,
            labels=store.labels,
            files=store.files,
            starts=store.starts,
            ends=store.ends,
            durations=store.durations,
            has_overlap=store.has_overlap,
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["idx", "file", "start", "end", "label", "duration", "has_overlap"],
            )
            writer.writeheader()
            for i in range(len(store.files)):
                writer.writerow(
                    {
                        "idx": i,
                        "file": store.files[i],
                        "start": store.starts[i],
                        "end": store.ends[i],
                        "label": store.labels[i],
                        "duration": store.durations[i],
                        "has_overlap": store.has_overlap[i],
                    }
                )

        if failures_csv_path is not None and self._failures:
            with open(failures_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["file", "start", "end", "label", "reason"],
                )
                writer.writeheader()
                writer.writerows(self._failures)

        print(f"Saved features  -> {npz_path}")
        print(f"Saved metadata  -> {csv_path}")
        if failures_csv_path is not None:
            print(f"Saved failures  -> {failures_csv_path}")


# ============================================================================== 
# GAMMATONE / SONORITY / SYLLABLE-LIKE SEGMENTATION
# ============================================================================== 


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
    # Kept identical in spirit to the validated pipeline.
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

    return __import__("pandas").DataFrame(
        rows,
        columns=["syllable_id", "start", "end", "duration"],
    )


# ============================================================================== 
# PROSODIC + ACOUSTIC EXTRACTOR
# ============================================================================== 


class ProsodicAcousticExtractor(BaseFeatureExtractor):
    """Extract validated F0, energy, duration and pause features."""

    name = "ProsodicAcoustic"

    FEATURE_NAMES = [
        "duration",
        "f0_mean", "f0_std", "f0_min", "f0_max", "f0_skew",
        "f0_tilt", "f0_tilt_mean", "f0_tilt_std", "f0_tilt_skew",
        "energy_mean", "energy_std", "energy_skew",
        "energy_tilt_mean", "energy_tilt_std", "energy_tilt_skew",
        "pause_after",
    ]

    def _f0_features(
        self,
        audio: np.ndarray,
        start: float,
        end: float,
    ) -> List[float]:
        snd = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
        pitch = snd.to_pitch(
            time_step=0.005,
            pitch_floor=75,
            pitch_ceiling=500,
        )

        f0_values = pitch.selected_array["frequency"].copy()
        f0_values[f0_values == 0] = np.nan
        f0_times = pitch.xs()

        mask = (f0_times >= start) & (f0_times < end)
        f0 = f0_values[mask]
        f0 = f0[np.isfinite(f0)]

        if len(f0):
            f0_mean = np.mean(f0)
            f0_std = np.std(f0)
            f0_min = np.min(f0)
            f0_max = np.max(f0)
            f0_skew = skew(f0) if len(f0) > 2 else 0.0
        else:
            f0_mean = f0_std = f0_min = f0_max = f0_skew = 0.0

        if len(f0) >= 2:
            x = np.arange(len(f0))
            f0_tilt = np.polyfit(x, f0, 1)[0]
            fd = np.diff(f0)
            f0_tilt_mean = np.mean(fd)
            f0_tilt_std = np.std(fd)
            f0_tilt_skew = skew(fd) if len(fd) > 2 else 0.0
        else:
            f0_tilt = f0_tilt_mean = f0_tilt_std = f0_tilt_skew = 0.0

        return [
            float(f0_mean), float(f0_std), float(f0_min), float(f0_max), float(f0_skew),
            float(f0_tilt), float(f0_tilt_mean), float(f0_tilt_std), float(f0_tilt_skew),
        ]

    def _energy_features(
        self,
        audio: np.ndarray,
        start: float,
        end: float,
    ) -> List[float]:
        n_fft = 320
        hop_length = 160

        rms = librosa.feature.rms(
            y=audio,
            frame_length=n_fft,
            hop_length=hop_length,
        )[0]

        energy_times = np.arange(len(rms)) * hop_length / self.sample_rate
        rms_norm = rms / np.max(rms) if len(rms) and np.max(rms) > 0 else rms

        mask = (energy_times >= start) & (energy_times < end)
        energy = rms_norm[mask]

        if len(energy):
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)
            energy_skew = skew(energy) if len(energy) > 2 else 0.0
        else:
            energy_mean = energy_std = energy_skew = 0.0

        if len(energy) >= 2:
            ed = np.diff(energy)
            energy_tilt_mean = np.mean(ed)
            energy_tilt_std = np.std(ed)
            energy_tilt_skew = skew(ed) if len(ed) > 2 else 0.0
        else:
            energy_tilt_mean = energy_tilt_std = energy_tilt_skew = 0.0

        return [
            float(energy_mean), float(energy_std), float(energy_skew),
            float(energy_tilt_mean), float(energy_tilt_std), float(energy_tilt_skew),
        ]

    @staticmethod
    def _pause_after(audio: np.ndarray, sr: int, end: float) -> float:
        """Estimate the low-energy pause immediately after a segment."""
        start_sample = max(0, int(end * sr))
        if start_sample >= len(audio):
            return 0.0

        # Inspect up to 2 seconds after the detected region.
        max_pause_samples = min(len(audio), start_sample + int(2.0 * sr))
        following = audio[start_sample:max_pause_samples]
        if len(following) < 2:
            return 0.0

        frame_length = 320
        hop_length = 160
        rms = librosa.feature.rms(
            y=following,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]

        if len(rms) == 0:
            return 0.0

        max_rms = np.max(rms)
        if max_rms <= 0:
            return 0.0

        pause_mask = rms < max_rms * 0.10

        frames = 0
        best_frames = 0
        for is_pause in pause_mask:
            if is_pause:
                frames += 1
                best_frames = max(best_frames, frames)
            else:
                frames = 0

        return float(best_frames * hop_length / sr)

    def extract_segment(
        self,
        audio: np.ndarray,
        start: float,
        end: float,
    ) -> np.ndarray:
        duration = max(0.0, float(end - start))

        f0 = self._f0_features(audio, start, end)
        energy = self._energy_features(audio, start, end)
        pause_after = self._pause_after(audio, self.sample_rate, end)

        vector = np.asarray(
            [duration, *f0, *energy, pause_after],
            dtype=np.float32,
        )

        if len(vector) != len(self.FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(self.FEATURE_NAMES)} features, got {len(vector)}"
            )

        return vector


# ============================================================================== 
# REGION LABEL MATCHING
# ============================================================================== 


def assign_annotation_labels(
    regions,
    annotations: List[AnnotationRow],
):
    """Assign the annotation with maximum temporal overlap to each region."""

    output = []

    for _, region in regions.iterrows():
        r_start = float(region["start"])
        r_end = float(region["end"])

        best = None
        best_overlap = 0.0

        for ann in annotations:
            overlap = max(0.0, min(r_end, ann.end) - max(r_start, ann.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best = ann

        if best is None:
            label = "Fluent"
            ann_start = np.nan
            ann_end = np.nan
        else:
            label = best.label
            ann_start = best.start
            ann_end = best.end

        region_duration = max(0.0, r_end - r_start)
        overlap_ratio = best_overlap / region_duration if region_duration > 0 else 0.0

        output.append(
            {
                "syllable_id": int(region["syllable_id"]),
                "label": label,
                "annotation_start": ann_start,
                "annotation_end": ann_end,
                "overlap_duration": best_overlap,
                "overlap_ratio": overlap_ratio,
            }
        )

    return output


# ============================================================================== 
# MAIN PIPELINE
# ============================================================================== 


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  IED PROSODIC + ACOUSTIC FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Dataset : {DATASET_DIR}")
    print(f"Output  : {OUTPUT_DIR}")

    # Step 1 — annotations
    print("\n[Step 1] Parsing annotations ...")
    parser = AnnotationParser()
    annotations = parser.parse_dataset(DATASET_DIR)
    annotations_by_file: Dict[str, List[AnnotationRow]] = {}
    for row in annotations:
        annotations_by_file.setdefault(row.file, []).append(row)
    print(f"Annotation rows: {len(annotations)}")

    # Step 2 — discover WAV/TXT pairs
    wav_paths = {p.stem: p for p in DATASET_DIR.rglob("*.wav")}
    common_ids = sorted(set(wav_paths) & set(annotations_by_file))
    print(f"Matched WAV + TXT files: {len(common_ids)}")

    extractor = ProsodicAcousticExtractor()
    extractor.setup()

    # Step 3 — segmentation + extraction
    print("\n[Step 2] Extracting features ...")
    records: List[FeatureRecord] = []
    failures: List[dict] = []

    for fi, stem in enumerate(common_ids, start=1):
        wav_path = wav_paths[stem]
        try:
            audio = extractor.audio_loader.load(wav_path)
            duration = len(audio) / SAMPLE_RATE
            file_annotations = annotations_by_file[stem]
            file_regions = []

            chunk_start = 0.0
            step = CHUNK_DURATION - CHUNK_OVERLAP

            while chunk_start < duration:
                chunk_end = min(chunk_start + CHUNK_DURATION, duration)
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

                core_start = chunk_start if chunk_start == 0 else chunk_start + CHUNK_OVERLAP
                regions = regions[regions["start"] >= core_start].copy()
                if chunk_end < duration:
                    regions = regions[regions["end"] <= chunk_end].copy()

                for _, region in regions.iterrows():
                    r_start = float(region["start"])
                    r_end = float(region["end"])

                    # Extract from the full audio using global timestamps.
                    vec = extractor.extract_segment(audio, r_start, r_end)

                    label_info = assign_annotation_labels(
                        __import__("pandas").DataFrame([region]),
                        file_annotations,
                    )[0]

                    records.append(
                        FeatureRecord(
                            file=stem,
                            start=r_start,
                            end=r_end,
                            label=label_info["label"],
                            duration=r_end - r_start,
                            has_overlap=False,
                            features=vec,
                        )
                    )

                chunk_start += step

            print(f"[{fi:>3}/{len(common_ids)}] {stem}: {sum(r.file == stem for r in records)} regions")

        except Exception as exc:
            print(f"[{fi:>3}/{len(common_ids)}] FAILED {stem}: {exc}")
            failures.append(
                {"file": stem, "start": 0.0, "end": 0.0, "label": "", "reason": repr(exc)}
            )

    if not records:
        raise RuntimeError("No features extracted.")

    # Step 4 — remove exact duplicate regions and assign sequential IDs.
    unique_records: List[FeatureRecord] = []
    seen = set()
    for record in sorted(records, key=lambda r: (r.file, r.start, r.end)):
        key = (record.file, round(record.start, 6), round(record.end, 6))
        if key in seen:
            continue
        seen.add(key)
        unique_records.append(record)

    records = unique_records

    store = DatasetStore(
        X=np.stack([r.features for r in records]).astype(np.float32),
        labels=np.array([r.label for r in records]),
        files=np.array([r.file for r in records]),
        starts=np.array([r.start for r in records], dtype=np.float64),
        ends=np.array([r.end for r in records], dtype=np.float64),
        durations=np.array([r.duration for r in records], dtype=np.float32),
        has_overlap=np.array([r.has_overlap for r in records], dtype=bool),
        feature_dim=records[0].features.shape[0],
        extractor_name=extractor.name,
    )

    # Step 5 — save standardized outputs.
    print("\n[Step 3] Saving standardized output ...")
    extractor._failures = failures
    extractor.save(
        store,
        npz_path=OUTPUT_DIR / "prosodic_acoustic_features.npz",
        csv_path=OUTPUT_DIR / "prosodic_acoustic_metadata.csv",
        failures_csv_path=OUTPUT_DIR / "prosodic_acoustic_failures.csv",
    )

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"Feature matrix : {store.X.shape}  (N x D)")
    print(f"Feature dim    : {store.feature_dim}")
    print(f"dtype          : {store.X.dtype}")
    print(f"NaN            : {np.isnan(store.X).any()}")
    print(f"Inf            : {np.isinf(store.X).any()}")
    print(f"Files          : {len(set(store.files.tolist()))}")
    print(f"Failures       : {len(failures)}")
    print("Label distribution:")
    unique, counts = np.unique(store.labels, return_counts=True)
    for label, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {label:<10} {count:>7}")
    print("=" * 70)


if __name__ == "__main__":
    main()
