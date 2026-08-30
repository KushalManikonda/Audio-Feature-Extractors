"""
Conformer Feature Extractor — OOP Implementation
=================================================
This file implements the Conformer Feature Extraction pipeline following the
standardized Object-Oriented Architecture described in `oop_structure_proposal.md`.

Components Included:
1. AnnotationRow, FeatureRecord, DatasetStore (Dataclasses)
2. AudioLoader (WAV loading, mono conversion, resampling, slicing)
3. AnnotationParser (Annotation parsing, label normalization, overlap detection)
4. FluencyRegionFinder (Gaps analysis & fluent chunk sampling)
5. BaseFeatureExtractor (Abstract Base Class for feature extractors)
6. ConformerExtractor (Indic-Conformer model embedding extractor)
"""

import os
import glob
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from transformers import AutoModel, AutoFeatureExtractor


# =============================================================================
# 1. Shared Dataclasses (Value Objects)
# =============================================================================

@dataclass
class AnnotationRow:
    """One labelled speech segment from a .txt annotation file."""
    file: str           # Audio file stem (no extension)
    start: float        # Start time in seconds
    end: float          # End time in seconds
    label: str          # Canonical label: I, PR, PhR, WR, PWR, P, Fluent
    duration: float     # Segment duration = end - start
    raw_label: str      # Original label before normalization
    source_line: int    # Line number in the .txt file
    has_overlap: bool = False  # True if this segment overlaps another interval


@dataclass
class FeatureRecord:
    """The result of running a feature extractor on one audio segment."""
    file: str
    start: float
    end: float
    label: str
    duration: float
    has_overlap: bool
    features: np.ndarray  # Fixed-size 1D vector of shape (D,)


@dataclass
class DatasetStore:
    """
    The complete extracted dataset container.
    Produced by calling BaseFeatureExtractor.extract_dataset().
    """
    X: np.ndarray           # Shape: (N, D) — feature matrix
    labels: np.ndarray      # Shape: (N,) — string labels
    files: np.ndarray       # Shape: (N,) — audio file stems
    starts: np.ndarray      # Shape: (N,) — segment start timestamps
    ends: np.ndarray        # Shape: (N,) — segment end timestamps
    durations: np.ndarray   # Shape: (N,) — segment durations
    has_overlap: np.ndarray # Shape: (N,) — overlap boolean flags
    feature_dim: int        # D — dimension of feature vectors
    extractor_name: str     # Name of extractor (e.g., "Conformer")


# =============================================================================
# 2. AudioLoader — Audio Processing Utility
# =============================================================================

class AudioLoader:
    """
    Loads WAV files, handles mono conversion, sample rate verification, and slicing.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def load(self, wav_path: str) -> Tuple[np.ndarray, int]:
        """
        Load WAV file -> mono float32 numpy array. Resamples if sr != self.sample_rate.
        Returns: (np.ndarray of shape (n_samples,), sample_rate)
        """
        waveform_np, sr = sf.read(wav_path)
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.mean(axis=1)
        waveform_np = waveform_np.astype(np.float32)

        if sr != self.sample_rate:
            tensor_wav = torch.from_numpy(waveform_np).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform_np = resampler(tensor_wav).squeeze(0).numpy()

        return waveform_np, self.sample_rate

    def slice_segment(self, audio: np.ndarray, start: float, end: float) -> np.ndarray:
        """
        Slice audio segment audio[start_sec : end_sec].
        Returns: np.ndarray of shape (n_segment_samples,)
        """
        start_idx = int(start * self.sample_rate)
        end_idx = int(end * self.sample_rate)
        return audio[start_idx:end_idx]


# =============================================================================
# 3. AnnotationParser — Parse & Normalize Labels
# =============================================================================

class AnnotationParser:
    """
    Reads .txt annotation files, normalizes raw disfluency labels, and marks overlaps.
    """

    LABEL_MAP = {
        "WR": "WR", "wr": "WR", "W.R": "WR", "Word Repetition": "WR",
        "PWR": "PWR", "pwr": "PWR", "P.W.R": "PWR", "Part-word Repetition": "PWR",
        "PR": "PR", "pr": "PR", "P.R": "PR", "Prolongation": "PR",
        "PhR": "PhR", "phr": "PhR", "Ph.R": "PhR", "Phrase Repetition": "PhR",
        "I": "I", "i": "I", "Interjection": "I", "Filled Pause": "I",
        "P": "P", "p": "P", "Pause": "P", "Silent Pause": "P",
        "Fluent": "Fluent", "F": "Fluent", "fluent": "Fluent"
    }
    VALID_LABELS = {"I", "PR", "PhR", "WR", "PWR", "P", "Fluent"}

    def normalise_label(self, raw: str) -> Tuple[Optional[str], str]:
        """
        Map a raw label string to canonical form.
        Handles compound labels, qualifiers, and casing variations.
        """
        raw_clean = raw.strip()
        if "," in raw_clean:
            raw_clean = raw_clean.split(",")[0].strip()
        if " " in raw_clean:
            parts = raw_clean.split()
            raw_clean = parts[0].strip()

        canonical = self.LABEL_MAP.get(raw_clean, raw_clean)
        if canonical in self.VALID_LABELS:
            return canonical, f"Mapped '{raw}' -> '{canonical}'"
        return None, f"Unrecognized label '{raw}'"

    def mark_overlaps(self, rows: List[AnnotationRow]) -> None:
        """Set has_overlap=True for any AnnotationRow that overlaps another row."""
        n = len(rows)
        for i in range(n):
            for j in range(i + 1, n):
                if max(rows[i].start, rows[j].start) < min(rows[i].end, rows[j].end):
                    rows[i].has_overlap = True
                    rows[j].has_overlap = True

    def parse_file(self, txt_path: str) -> Tuple[List[AnnotationRow], List[str]]:
        """Parse one .txt annotation file."""
        rows = []
        warnings = []
        file_stem = os.path.splitext(os.path.basename(txt_path))[0]

        if not os.path.exists(txt_path):
            return rows, [f"File not found: {txt_path}"]

        with open(txt_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue
                parts = line_str.split('\t')
                if len(parts) < 3:
                    parts = line_str.split()
                if len(parts) >= 3:
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        raw_label = parts[2]
                        label, note = self.normalise_label(raw_label)
                        if label:
                            rows.append(AnnotationRow(
                                file=file_stem,
                                start=start,
                                end=end,
                                label=label,
                                duration=max(0.0, end - start),
                                raw_label=raw_label,
                                source_line=line_num
                            ))
                        else:
                            warnings.append(f"Line {line_num} in {file_stem}: {note}")
                    except ValueError:
                        warnings.append(f"Line {line_num} in {file_stem}: Parse error.")
        self.mark_overlaps(rows)
        return rows, warnings

    def parse_dataset(self, dataset_dir: str) -> List[AnnotationRow]:
        """Parse all .txt annotation files in dataset_dir."""
        txt_files = glob.glob(os.path.join(dataset_dir, "*.txt"))
        all_rows = []
        for txt_path in sorted(txt_files):
            rows, warnings = self.parse_file(txt_path)
            all_rows.extend(rows)
        return all_rows


# =============================================================================
# 4. FluencyRegionFinder — Sample Fluent Speech Gaps
# =============================================================================

class FluencyRegionFinder:
    """
    Identifies non-disfluent gap regions in speech to sample 'Fluent' speech chunks.
    """

    def __init__(self, chunk_duration: float = 1.0, random_seed: int = 42):
        self.chunk_duration = chunk_duration
        self.random_seed = random_seed

    def find_fluent_chunks(
        self, file_stem: str, total_duration: float, disfluent_intervals: List[Tuple[float, float]]
    ) -> List[AnnotationRow]:
        """Identify fluent chunk AnnotationRows outside disfluent intervals."""
        if not disfluent_intervals:
            intervals = []
        else:
            intervals = sorted(disfluent_intervals, key=lambda x: x[0])

        merged = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        gaps = []
        current_t = 0.0
        for start, end in merged:
            if start > current_t:
                gaps.append((current_t, start))
            current_t = max(current_t, end)
        if current_t < total_duration:
            gaps.append((current_t, total_duration))

        fluent_rows = []
        line_idx = 1000
        for g_start, g_end in gaps:
            t = g_start
            while t + self.chunk_duration <= g_end:
                fluent_rows.append(AnnotationRow(
                    file=file_stem,
                    start=round(t, 4),
                    end=round(t + self.chunk_duration, 4),
                    label="Fluent",
                    duration=self.chunk_duration,
                    raw_label="Fluent_Gap",
                    source_line=line_idx,
                    has_overlap=False
                ))
                t += self.chunk_duration
                line_idx += 1
        return fluent_rows

    def sample(self, candidates: List[AnnotationRow], target_count: int) -> List[AnnotationRow]:
        """Randomly sample target_count candidate fluent rows."""
        if len(candidates) <= target_count:
            return candidates
        rng = np.random.RandomState(self.random_seed)
        indices = rng.choice(len(candidates), size=target_count, replace=False)
        return [candidates[i] for i in sorted(indices)]


# =============================================================================
# 5. BaseFeatureExtractor — Abstract Base Class
# =============================================================================

class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.
    Subclass this and implement extract_segment() or override extract_dataset()
    for custom batching behavior.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_loader = AudioLoader(sample_rate)

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the feature extractor (e.g., 'Conformer')."""
        pass

    @abstractmethod
    def setup(self) -> None:
        """Load model weights, device setup, etc."""
        pass

    @abstractmethod
    def extract_segment(self, audio: np.ndarray, start: float, end: float) -> np.ndarray:
        """Extract a single (D,) feature vector for segment audio[start:end]."""
        pass

    def extract_dataset(self, rows: List[AnnotationRow], dataset_dir: str) -> DatasetStore:
        """Run extraction across all AnnotationRows and construct a DatasetStore."""
        file_to_rows: Dict[str, List[AnnotationRow]] = {}
        for r in rows:
            file_to_rows.setdefault(r.file, []).append(r)

        records: List[FeatureRecord] = []

        for file_stem, f_rows in file_to_rows.items():
            wav_path = os.path.join(dataset_dir, f"{file_stem}.wav")
            if not os.path.exists(wav_path):
                print(f"Warning: Audio file '{wav_path}' not found. Skipping.")
                continue

            try:
                audio, _ = self.audio_loader.load(wav_path)
            except Exception as e:
                print(f"Error loading audio '{wav_path}': {e}")
                continue

            for row in f_rows:
                try:
                    feat = self.extract_segment(audio, row.start, row.end)
                    records.append(FeatureRecord(
                        file=row.file,
                        start=row.start,
                        end=row.end,
                        label=row.label,
                        duration=row.duration,
                        has_overlap=row.has_overlap,
                        features=feat
                    ))
                except Exception as e:
                    print(f"Error extracting segment {row.file} [{row.start}-{row.end}]: {e}")

        if not records:
            raise RuntimeError("No feature records were successfully extracted!")

        X = np.stack([rec.features for rec in records], axis=0)
        labels = np.array([rec.label for rec in records])
        files = np.array([rec.file for rec in records])
        starts = np.array([rec.start for rec in records])
        ends = np.array([rec.end for rec in records])
        durations = np.array([rec.duration for rec in records])
        has_overlap = np.array([rec.has_overlap for rec in records])

        return DatasetStore(
            X=X,
            labels=labels,
            files=files,
            starts=starts,
            ends=ends,
            durations=durations,
            has_overlap=has_overlap,
            feature_dim=X.shape[1],
            extractor_name=self.name
        )

    def save(self, store: DatasetStore, npz_path: str, csv_path: str) -> None:
        """Save DatasetStore to .npz feature matrix + .csv metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

        np.savez_compressed(
            npz_path,
            X=store.X,
            labels=store.labels,
            files=store.files,
            starts=store.starts,
            ends=store.ends,
            durations=store.durations,
            has_overlap=store.has_overlap,
            feature_dim=store.feature_dim,
            extractor_name=store.extractor_name
        )

        df = pd.DataFrame({
            'file': store.files,
            'start': store.starts,
            'end': store.ends,
            'label': store.labels,
            'duration': store.durations,
            'has_overlap': store.has_overlap
        })
        df.to_csv(csv_path, index=False)
        print(f"Successfully saved features to '{npz_path}' and metadata to '{csv_path}'.")

    def load(self, npz_path: str, csv_path: str) -> DatasetStore:
        """Load DatasetStore from .npz + .csv."""
        data = np.load(npz_path)
        return DatasetStore(
            X=data['X'],
            labels=data['labels'],
            files=data['files'],
            starts=data['starts'],
            ends=data['ends'],
            durations=data['durations'],
            has_overlap=data['has_overlap'],
            feature_dim=int(data['feature_dim']),
            extractor_name=str(data['extractor_name'])
        )


# =============================================================================
# 6. ConformerExtractor — Concrete Implementation
# =============================================================================

class ConformerExtractor(BaseFeatureExtractor):
    """
    Feature Extractor subclass using Conformer model embeddings
    (e.g., ai4bharat/indic-conformer-600m-multilingual).
    """

    def __init__(
        self,
        model_id: str = "ai4bharat/indic-conformer-600m-multilingual",
        sample_rate: int = 16000,
        device: Optional[str] = None
    ):
        super().__init__(sample_rate=sample_rate)
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    @property
    def name(self) -> str:
        return "Conformer"

    def setup(self) -> None:
        """Load Indic Conformer model weights to designated compute device."""
        print(f"Loading Conformer Model '{self.model_id}' on device '{self.device}'...")
        try:
            self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
            self.model.eval()
            self.model.to(self.device)
            print("Conformer model loaded successfully.")
        except Exception as e:
            print(f"Error loading model '{self.model_id}' via AutoModel: {e}")
            raise RuntimeError(f"Failed to setup ConformerExtractor model: {e}")

    def _run_forward(self, waveform_tensor: torch.Tensor) -> np.ndarray:
        """
        Process audio tensor through Conformer encoder in 30s chunks.
        Returns: concatenated frame embeddings np.ndarray of shape (T, D)
        """
        chunk_size = int(30.0 * self.sample_rate)
        embeddings_list = []

        with torch.no_grad():
            for i in range(0, waveform_tensor.shape[1], chunk_size):
                chunk = waveform_tensor[:, i:i+chunk_size].to(self.device)
                if chunk.shape[1] < int(0.1 * self.sample_rate): # Skip tiny chunks <100ms
                    continue

                if hasattr(self.model, 'encode'):
                    encoder_outputs, encoded_lengths = self.model.encode(chunk)
                    if encoder_outputs.ndim == 3:
                        if encoder_outputs.shape[1] > encoder_outputs.shape[2]: # (1, T, D)
                            chunk_emb = encoder_outputs[0, :encoded_lengths[0], :].cpu().numpy()
                        else: # (1, D, T)
                            chunk_emb = encoder_outputs[0, :, :encoded_lengths[0]].T.cpu().numpy()
                    else:
                        chunk_emb = encoder_outputs.squeeze(0).cpu().numpy()
                    embeddings_list.append(chunk_emb)
                else:
                    try:
                        outputs = self.model(chunk)
                    except Exception:
                        outputs = self.model(chunk, output_hidden_states=True)

                    if hasattr(outputs, 'last_hidden_state'):
                        chunk_emb = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                    elif hasattr(outputs, 'hidden_states'):
                        chunk_emb = outputs.hidden_states[-1].squeeze(0).cpu().numpy()
                    elif isinstance(outputs, tuple):
                        chunk_emb = outputs[0].squeeze(0).cpu().numpy()
                    else:
                        chunk_emb = outputs.squeeze(0).cpu().numpy()
                    embeddings_list.append(chunk_emb)

        if not embeddings_list:
            raise ValueError("No embeddings were generated during forward pass.")
        return np.concatenate(embeddings_list, axis=0)

    def extract_segment(self, audio: np.ndarray, start: float, end: float) -> np.ndarray:
        """Extract mean-pooled fixed-size feature vector (D,) for a single audio segment."""
        segment_audio = self.audio_loader.slice_segment(audio, start, end)
        min_samples = int(0.05 * self.sample_rate)
        if len(segment_audio) < min_samples:
            segment_audio = np.pad(segment_audio, (0, min_samples - len(segment_audio)))

        wav_tensor = torch.from_numpy(segment_audio).unsqueeze(0).float()
        frame_embeddings = self._run_forward(wav_tensor)  # (T, D)
        return frame_embeddings.mean(axis=0)  # Mean-pooled 1D embedding (D,)

    def extract_dataset(self, rows: List[AnnotationRow], dataset_dir: str) -> DatasetStore:
        """
        Optimized Dataset Extraction:
        Runs Conformer forward pass ONCE per full WAV file, then slices and mean-pools
        frame embeddings corresponding to each annotation segment interval.
        """
        file_to_rows: Dict[str, List[AnnotationRow]] = {}
        for r in rows:
            file_to_rows.setdefault(r.file, []).append(r)

        records: List[FeatureRecord] = []

        for file_stem, f_rows in file_to_rows.items():
            wav_path = os.path.join(dataset_dir, f"{file_stem}.wav")
            if not os.path.exists(wav_path):
                print(f"Warning: WAV file '{wav_path}' not found. Skipping.")
                continue

            try:
                audio, sr = self.audio_loader.load(wav_path)
                wav_tensor = torch.from_numpy(audio).unsqueeze(0).float()
                frame_embeddings = self._run_forward(wav_tensor)  # (T, D)

                total_sec = len(audio) / sr
                num_frames = frame_embeddings.shape[0]
                sec_per_frame = total_sec / num_frames if num_frames > 0 else 0.02

                for row in f_rows:
                    start_frame = int(row.start / sec_per_frame)
                    end_frame = max(start_frame + 1, int(row.end / sec_per_frame))

                    seg_frames = frame_embeddings[start_frame:min(end_frame, num_frames)]
                    if len(seg_frames) == 0:
                        seg_vector = frame_embeddings[min(start_frame, num_frames - 1)]
                    else:
                        seg_vector = seg_frames.mean(axis=0)

                    records.append(FeatureRecord(
                        file=row.file,
                        start=row.start,
                        end=row.end,
                        label=row.label,
                        duration=row.duration,
                        has_overlap=row.has_overlap,
                        features=seg_vector
                    ))
            except Exception as e:
                print(f"Error processing audio file '{file_stem}': {e}")

        if not records:
            raise RuntimeError("No feature records were successfully extracted!")

        X = np.stack([rec.features for rec in records], axis=0)
        labels = np.array([rec.label for rec in records])
        files = np.array([rec.file for rec in records])
        starts = np.array([rec.start for rec in records])
        ends = np.array([rec.end for rec in records])
        durations = np.array([rec.duration for rec in records])
        has_overlap = np.array([rec.has_overlap for rec in records])

        return DatasetStore(
            X=X,
            labels=labels,
            files=files,
            starts=starts,
            ends=ends,
            durations=durations,
            has_overlap=has_overlap,
            feature_dim=X.shape[1],
            extractor_name=self.name
        )


# =============================================================================
# 7. Main Execution Pipeline (Demonstration)
# =============================================================================

if __name__ == "__main__":
    DATASET_DIR = "./IED Dataset"
    OUTPUT_DIR = "./extracted_features"

    print("Step 1: Parsing Disfluency Annotations...")
    parser = AnnotationParser()
    rows = parser.parse_dataset(DATASET_DIR)
    print(f"Parsed {len(rows)} disfluency annotation segments.")

    print("\nStep 2: Identifying Fluent Regions...")
    finder = FluencyRegionFinder(chunk_duration=1.0)
    loader = AudioLoader()

    file_stems = sorted(list(set(r.file for r in rows)))
    fluent_candidates = []
    for file_stem in file_stems:
        wav_path = os.path.join(DATASET_DIR, f"{file_stem}.wav")
        if os.path.exists(wav_path):
            audio, sr = loader.load(wav_path)
            total_dur = len(audio) / sr
            disf_intervals = [(r.start, r.end) for r in rows if r.file == file_stem]
            f_rows = finder.find_fluent_chunks(file_stem, total_dur, disf_intervals)
            fluent_candidates.extend(f_rows)

    sampled_fluent = finder.sample(fluent_candidates, target_count=len(rows))
    all_rows = rows + sampled_fluent
    print(f"Sampled {len(sampled_fluent)} fluent segments. Total segments: {len(all_rows)}.")

    print("\nStep 3: Initializing Conformer Extractor...")
    extractor = ConformerExtractor(model_id="ai4bharat/indic-conformer-600m-multilingual")
    extractor.setup()

    print("\nStep 4: Extracting Features...")
    store = extractor.extract_dataset(all_rows, DATASET_DIR)
    print(f"Extraction Complete. Matrix shape: {store.X.shape}, Extractor: {store.extractor_name}")

    print("\nStep 5: Saving Dataset Store...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    npz_out = os.path.join(OUTPUT_DIR, "conformer_features.npz")
    csv_out = os.path.join(OUTPUT_DIR, "conformer_metadata.csv")
    extractor.save(store, npz_out, csv_out)
    print("Done!")
