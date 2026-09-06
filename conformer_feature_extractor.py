"""
Conformer Feature Extractor & Multi-Model Classifier Pipeline
============================================================
This file implements the complete unified pipeline containing both:
1. Conformer Feature Extractor (Indic-Conformer acoustic embeddings)
2. Classifiers (Random Forest, DNN, BiLSTM) & Evaluation Pipeline

Following the standardized Object-Oriented Architecture (`oop_structure_proposal.md`).

Components Included:
1. AnnotationRow, FeatureRecord, DatasetStore, ClassificationTask, EvaluationResult (Dataclasses)
2. AudioLoader (WAV loading, mono conversion, resampling, slicing)
3. AnnotationParser (Annotation parsing, label normalization, overlap detection)
4. FluencyRegionFinder (Gaps analysis & fluent chunk sampling)
5. BaseFeatureExtractor (Abstract Base Class for feature extractors)
6. ConformerExtractor (Indic-Conformer model embedding extractor)
7. SplitManager (Speaker-independent dataset partitioning & sequence creation)
8. BaseClassifier (Abstract Base Class for models: RandomForestWrapper, DNNWrapper, BiLSTMWrapper)
9. Evaluator & ClassifierRunner (Multi-experiment evaluation, table formatting, CSV/TXT export)
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import joblib
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


@dataclass
class ClassificationTask:
    """Defines one of the standard classification experiments."""
    name: str                       # e.g. "Exp1_FluDis", "Exp2_Flu_I", "Exp3_6class"
    filter_classes: List[str]       # labels to include in task
    is_binary: bool = True          # True for binary classification


@dataclass
class EvaluationResult:
    """Metrics from one classifier on one task/experiment."""
    experiment: str
    model: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    precision: float
    recall: float
    roc_auc: float
    confusion_matrix: np.ndarray


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
        "PWR": "PWR", "pwr": "PWR", "P.W.R": "PWR", "Part-word Repetition": "PWR", "PWR OR BLOCK": "PWR",
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
        if " " in raw_clean and raw_clean not in self.LABEL_MAP:
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
# 7. SplitManager — Data Loading & Speaker Independent Splitting
# =============================================================================

class SplitManager:
    """
    Handles speaker-independent train/test splitting using GroupShuffleSplit.
    Can load existing frame-level feature files from `extracted_features` or process a DatasetStore.
    """

    LABEL_MAP = {
        'PWR OR BLOCK': 'PWR',
        'P.W.R': 'PWR',
        'Ph.R': 'PhR',
        'P.R': 'PR',
        'W.R': 'WR'
    }

    def __init__(self, data_dir: str = "./extracted_features", test_size: float = 0.2, random_state: int = 42):
        self.data_dir = data_dir
        self.test_size = test_size
        self.random_state = random_state

    def load_extracted_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Load all frame-level features (*_features.npy) and labels (*_labels.csv).
        Returns: (X, y_binary, y_multiclass, groups, file_records)
        """
        npy_files = glob.glob(os.path.join(self.data_dir, "*_features.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No *_features.npy files found in '{self.data_dir}'.")

        features_list = []
        binary_labels_list = []
        multiclass_labels_list = []
        speaker_ids_list = []
        file_records = []

        for npy_path in sorted(npy_files):
            base_name = os.path.basename(npy_path).replace("_features.npy", "")
            csv_path = os.path.join(self.data_dir, f"{base_name}_labels.csv")
            if not os.path.exists(csv_path):
                continue

            feats = np.load(npy_path)
            df_labels = pd.read_csv(csv_path)

            if len(feats) != len(df_labels):
                min_len = min(len(feats), len(df_labels))
                feats = feats[:min_len]
                df_labels = df_labels.iloc[:min_len]

            df_labels['multiclass_label'] = df_labels['multiclass_label'].replace(self.LABEL_MAP)
            speaker_id = base_name.split("__")[1] if "__" in base_name else base_name

            features_list.append(feats)
            binary_labels_list.append(df_labels['binary_label'].values)
            multiclass_labels_list.append(df_labels['multiclass_label'].values)
            speaker_ids_list.append(np.full(len(feats), speaker_id))

            file_records.append({
                'base_name': base_name,
                'feats': feats,
                'binary_labels': df_labels['binary_label'].values,
                'multiclass_labels': df_labels['multiclass_label'].values,
                'speaker_id': speaker_id
            })

        X = np.vstack(features_list)
        y_binary = np.concatenate(binary_labels_list)
        y_multi = np.concatenate(multiclass_labels_list)
        groups = np.concatenate(speaker_ids_list)

        return X, y_binary, y_multi, groups, file_records

    def get_speaker_split(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform speaker-independent GroupShuffleSplit on X and y."""
        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        return train_idx, test_idx


# =============================================================================
# 8. BaseClassifier — Abstract Interface & Model Wrappers
# =============================================================================

class BaseClassifier(ABC):
    """
    Abstract interface for all model wrappers (RF, DNN, BiLSTM).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Classifier abbreviation (e.g. 'RF', 'DNN', 'BiLSTM')."""
        pass

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit model on training set."""
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict labels on test set."""
        pass

    def predict_proba(self, X_test: np.ndarray) -> Optional[np.ndarray]:
        """Predict probabilities on test set (optional)."""
        return None


class RandomForestWrapper(BaseClassifier):
    """Random Forest Classifier Wrapper."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 20, random_state: int = 42):
        self._name = "RF"
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
            class_weight='balanced'
        )

    @property
    def name(self) -> str:
        return self._name

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        max_samples = 50000
        if len(X_train) > max_samples:
            indices = np.random.choice(len(X_train), max_samples, replace=False)
            X_tr_sub, y_tr_sub = X_train[indices], y_train[indices]
        else:
            X_tr_sub, y_tr_sub = X_train, y_train
        self.model.fit(X_tr_sub, y_tr_sub)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)
        return None


class DisfluencyDNN(nn.Module):
    """PyTorch Deep Neural Network (Dense MLP)."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DNNWrapper(BaseClassifier):
    """DNN Classifier Wrapper."""

    def __init__(self, is_binary: bool = True, num_classes: int = 1, epochs: int = 8, batch_size: int = 2048, lr: float = 1e-3):
        self._name = "DNN"
        self.is_binary = is_binary
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    @property
    def name(self) -> str:
        return self._name

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        input_dim = X_train.shape[1]
        out_dim = 1 if self.is_binary else self.num_classes
        self.model = DisfluencyDNN(input_dim, out_dim).to(self.device)

        X_tr_t = torch.tensor(X_train, dtype=torch.float32)

        if self.is_binary:
            y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            num_pos = torch.sum(y_tr_t == 1)
            num_neg = torch.sum(y_tr_t == 0)
            pos_weight = (num_neg / (num_pos + 1e-5)).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            y_tr_t = torch.tensor(y_train, dtype=torch.long)
            class_counts = np.bincount(y_train, minlength=self.num_classes)
            class_weights = 1.0 / (class_counts + 1e-5)
            class_weights = class_weights / class_weights.sum()
            weights_t = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights_t)

        train_dataset = TensorDataset(X_tr_t, y_tr_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_te_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_te_t)
            if self.is_binary:
                probs = torch.sigmoid(outputs).cpu().numpy()
                return (probs >= 0.5).astype(int).flatten()
            else:
                return torch.argmax(outputs, dim=1).cpu().numpy().astype(int)

    def predict_proba(self, X_test: np.ndarray) -> Optional[np.ndarray]:
        self.model.eval()
        X_te_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_te_t)
            if self.is_binary:
                probs = torch.sigmoid(outputs).cpu().numpy()
                return np.hstack([1.0 - probs, probs])
            else:
                return torch.softmax(outputs, dim=1).cpu().numpy()


class DisfluencyBiLSTM(nn.Module):
    """PyTorch Sequence Model (Bidirectional LSTM)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        B, T, H = lstm_out.shape
        out_flat = lstm_out.reshape(-1, H)
        logits_flat = self.fc(out_flat)
        return logits_flat.reshape(B, T, -1)


class BiLSTMWrapper(BaseClassifier):
    """BiLSTM Classifier Wrapper operating on sequence windows."""

    def __init__(self, is_binary: bool = True, num_classes: int = 1, epochs: int = 4, batch_size: int = 512, seq_len: int = 50, stride: int = 50):
        self._name = "BiLSTM"
        self.is_binary = is_binary
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stride = stride
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    @property
    def name(self) -> str:
        return self._name

    def _create_sequences(self, X: np.ndarray, y: np.ndarray, stride: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Convert flat frame features into sequence chunks of shape (B, T, D)."""
        stride = stride or self.stride
        X_seqs, y_seqs = [], []
        n_frames = len(X)
        if n_frames < self.seq_len:
            return np.expand_dims(X, 0), np.expand_dims(y, 0)

        for i in range(0, n_frames - self.seq_len + 1, stride):
            X_seqs.append(X[i:i+self.seq_len])
            y_seqs.append(y[i:i+self.seq_len])

        return np.array(X_seqs), np.array(y_seqs)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_seq, y_seq = self._create_sequences(X_train, y_train, stride=self.stride)
        max_seqs = 3500
        if len(X_seq) > max_seqs:
            indices = np.random.choice(len(X_seq), max_seqs, replace=False)
            X_seq, y_seq = X_seq[indices], y_seq[indices]

        input_dim = X_seq.shape[2]
        out_dim = 1 if self.is_binary else self.num_classes
        self.model = DisfluencyBiLSTM(input_dim=input_dim, hidden_dim=128, num_classes=out_dim).to(self.device)

        X_tr_t = torch.tensor(X_seq, dtype=torch.float32)

        if self.is_binary:
            y_tr_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1)
            num_pos = torch.sum(y_tr_t == 1)
            num_neg = torch.sum(y_tr_t == 0)
            pos_weight = (num_neg / (num_pos + 1e-5)).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            y_tr_t = torch.tensor(y_seq, dtype=torch.long)
            class_counts = np.bincount(y_seq.flatten(), minlength=self.num_classes)
            class_weights = 1.0 / (class_counts + 1e-5)
            class_weights = class_weights / class_weights.sum()
            weights_t = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights_t)

        train_dataset = TensorDataset(X_tr_t, y_tr_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                if self.is_binary:
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.reshape(-1, self.num_classes), batch_y.reshape(-1))
                loss.backward()
                optimizer.step()
            scheduler.step()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        dummy_y = np.zeros(len(X_test))
        X_seq, _ = self._create_sequences(X_test, dummy_y)

        self.model.eval()
        X_te_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_te_t)
            if self.is_binary:
                probs = torch.sigmoid(outputs).cpu().numpy().reshape(-1)
                preds = (probs >= 0.5).astype(int)
            else:
                preds = torch.argmax(outputs, dim=-1).cpu().numpy().flatten().astype(int)

        if len(preds) < len(X_test):
            preds = np.pad(preds, (0, len(X_test) - len(preds)), mode='edge')
        else:
            preds = preds[:len(X_test)]
        return preds

    def predict_proba(self, X_test: np.ndarray) -> Optional[np.ndarray]:
        dummy_y = np.zeros(len(X_test))
        X_seq, _ = self._create_sequences(X_test, dummy_y)

        self.model.eval()
        X_te_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_te_t)
            if self.is_binary:
                probs = torch.sigmoid(outputs).cpu().numpy().reshape(-1)
                if len(probs) < len(X_test):
                    probs = np.pad(probs, (0, len(X_test) - len(probs)), mode='edge')
                else:
                    probs = probs[:len(X_test)]
                return np.vstack([1.0 - probs, probs]).T
            else:
                probs = torch.softmax(outputs, dim=-1).cpu().numpy().reshape(-1, self.num_classes)
                if len(probs) < len(X_test):
                    pad_len = len(X_test) - len(probs)
                    probs = np.pad(probs, ((0, pad_len), (0, 0)), mode='edge')
                else:
                    probs = probs[:len(X_test)]
                return probs


# =============================================================================
# 9. Evaluator & ClassifierRunner — Experiments & Results Formatting
# =============================================================================

class Evaluator:
    """
    Computes classification evaluation metrics:
    Accuracy, Macro-F1, Weighted-F1, Precision, Recall, and ROC-AUC.
    """

    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray], experiment: str, model_name: str, is_binary: bool
    ) -> EvaluationResult:
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        precision = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        recall = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

        roc_auc = float('nan')
        if is_binary and y_prob is not None:
            try:
                if y_prob.ndim == 2:
                    prob_positive = y_prob[:, 1]
                else:
                    prob_positive = y_prob
                roc_auc = float(roc_auc_score(y_true, prob_positive))
            except Exception:
                roc_auc = float('nan')

        cm = confusion_matrix(y_true, y_pred)

        return EvaluationResult(
            experiment=experiment,
            model=model_name,
            accuracy=acc,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            precision=precision,
            recall=recall,
            roc_auc=roc_auc,
            confusion_matrix=cm
        )


class ClassifierRunner:
    """
    Runs multi-model classification tasks across all standard experiments:
    1. Exp1_FluDis (Fluent vs Disfluent)
    2. Exp2_Flu_<class> (Fluent vs specific disfluency class: I, PhR, WR, PWR, P)
    3. Exp3_6class (6-class multi-class disfluency classification)
    """

    DISFLUENCY_CLASSES = ['I', 'PhR', 'WR', 'PWR', 'P']

    def __init__(self, data_dir: str = "./extracted_features", output_dir: str = "./"):
        self.splitter = SplitManager(data_dir=data_dir)
        self.evaluator = Evaluator()
        self.output_dir = output_dir

    def run_all_experiments(self) -> List[EvaluationResult]:
        print("\nLoading dataset for classification pipeline...")
        X, y_binary, y_multi, groups, _ = self.splitter.load_extracted_dataset()
        print(f"Loaded feature matrix shape: X={X.shape}, y_binary={y_binary.shape}, y_multi={y_multi.shape}")

        train_idx, test_idx = self.splitter.get_speaker_split(X, y_binary, groups)

        X_tr, X_te = X[train_idx], X[test_idx]
        y_bin_tr, y_bin_te = y_binary[train_idx], y_binary[test_idx]
        y_multi_tr, y_multi_te = y_multi[train_idx], y_multi[test_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        results: List[EvaluationResult] = []

        # ---------------------------------------------------------------------
        # Task 1: Exp1_FluDis (Fluent vs Disfluent)
        # ---------------------------------------------------------------------
        exp_name = "Exp1_FluDis"
        print(f"\n--- Running Experiment: {exp_name} ---")

        models_task1 = [
            RandomForestWrapper(),
            DNNWrapper(is_binary=True, num_classes=1, epochs=8, batch_size=2048),
            BiLSTMWrapper(is_binary=True, num_classes=1, epochs=5, batch_size=256)
        ]

        for model in models_task1:
            print(f"Training {model.name} for {exp_name}...")
            model.fit(X_tr_scaled, y_bin_tr)
            preds = model.predict(X_te_scaled)
            probs = model.predict_proba(X_te_scaled)
            res = self.evaluator.evaluate(y_bin_te, preds, probs, exp_name, model.name, is_binary=True)
            results.append(res)

        # ---------------------------------------------------------------------
        # Task 2: Exp2_Flu_<Class> (Per Class Binary Classifications)
        # ---------------------------------------------------------------------
        for dis_class in self.DISFLUENCY_CLASSES:
            exp_name = f"Exp2_Flu_{dis_class}"
            print(f"\n--- Running Experiment: {exp_name} ---")

            mask_tr = (y_multi_tr == 'F') | (y_multi_tr == dis_class)
            mask_te = (y_multi_te == 'F') | (y_multi_te == dis_class)

            if np.sum(y_multi_tr == dis_class) == 0 or np.sum(y_multi_te == dis_class) == 0:
                print(f"Skipping {exp_name} (insufficient samples).")
                continue

            X_tr_sub, y_tr_sub = X_tr_scaled[mask_tr], y_multi_tr[mask_tr]
            X_te_sub, y_te_sub = X_te_scaled[mask_te], y_multi_te[mask_te]

            y_tr_bin_sub = np.where(y_tr_sub == 'F', 0, 1)
            y_te_bin_sub = np.where(y_te_sub == 'F', 0, 1)

            models_task2 = [
                RandomForestWrapper(),
                DNNWrapper(is_binary=True, num_classes=1, epochs=8, batch_size=2048),
                BiLSTMWrapper(is_binary=True, num_classes=1, epochs=5, batch_size=256)
            ]

            for model in models_task2:
                print(f"Training {model.name} for {exp_name}...")
                model.fit(X_tr_sub, y_tr_bin_sub)
                preds = model.predict(X_te_sub)
                probs = model.predict_proba(X_te_sub)
                res = self.evaluator.evaluate(y_te_bin_sub, preds, probs, exp_name, model.name, is_binary=True)
                results.append(res)

        # ---------------------------------------------------------------------
        # Task 3: Exp3_6class (Multi-Class Disfluency Classification)
        # ---------------------------------------------------------------------
        exp_name = "Exp3_6class"
        print(f"\n--- Running Experiment: {exp_name} ---")

        le = LabelEncoder()
        le.fit(y_multi)
        y_multi_tr_enc = le.transform(y_multi_tr)
        y_multi_te_enc = le.transform(y_multi_te)
        num_classes = len(le.classes_)

        models_task3 = [
            RandomForestWrapper(),
            DNNWrapper(is_binary=False, num_classes=num_classes, epochs=8, batch_size=2048),
            BiLSTMWrapper(is_binary=False, num_classes=num_classes, epochs=5, batch_size=256)
        ]

        for model in models_task3:
            print(f"Training {model.name} for {exp_name}...")
            model.fit(X_tr_scaled, y_multi_tr_enc)
            preds = model.predict(X_te_scaled)
            probs = model.predict_proba(X_te_scaled)
            res = self.evaluator.evaluate(y_multi_te_enc, preds, probs, exp_name, model.name, is_binary=False)
            results.append(res)

        return results

    def format_results_table(self, results: List[EvaluationResult]) -> str:
        """Format results as the exact text table shown in uploaded image."""
        lines = []
        lines.append("================================================================================")
        lines.append("FINAL RESULTS TABLE (FRAME-LEVEL EVALUATION - Primary metric: Macro-F1)")
        lines.append("================================================================================")
        lines.append(f"{'experiment':>12} {'model':>6} {'accuracy':>9} {'macro_f1':>9} {'weighted_f1':>12} {'precision':>10} {'recall':>7} {'roc_auc':>8}")

        exp_best: Dict[str, Tuple[str, float]] = {}

        for res in results:
            roc_str = "     NaN" if np.isnan(res.roc_auc) else f"{res.roc_auc:8.4f}"
            lines.append(
                f"{res.experiment:>12} {res.model:>6} {res.accuracy:9.4f} {res.macro_f1:9.4f} {res.weighted_f1:12.4f} {res.precision:10.4f} {res.recall:7.4f} {roc_str}"
            )

            if res.experiment not in exp_best or res.macro_f1 > exp_best[res.experiment][1]:
                exp_best[res.experiment] = (res.model, res.macro_f1)

        lines.append("\n--- Best Model per Experiment ---")
        for exp, (b_model, b_f1) in exp_best.items():
            lines.append(f"{exp:<22} -> {b_model:<7} Macro-F1={b_f1:.4f}")

        return "\n".join(lines)

    def export_results(self, results: List[EvaluationResult], csv_path: str, txt_path: str) -> None:
        """Export results table to CSV and TXT files."""
        formatted_table = self.format_results_table(results)

        # 1. Save TXT file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(formatted_table + "\n")
        print(f"\nSaved formatted results TXT file to: '{txt_path}'")

        # 2. Save CSV file
        df = pd.DataFrame([{
            'experiment': r.experiment,
            'model': r.model,
            'accuracy': round(r.accuracy, 4),
            'macro_f1': round(r.macro_f1, 4),
            'weighted_f1': round(r.weighted_f1, 4),
            'precision': round(r.precision, 4),
            'recall': round(r.recall, 4),
            'roc_auc': "NaN" if np.isnan(r.roc_auc) else round(r.roc_auc, 4)
        } for r in results])

        df.to_csv(csv_path, index=False)
        print(f"Saved formatted results CSV file to: '{csv_path}'")


# =============================================================================
# 10. Main Execution Pipeline (Unified Extractor & Classifier)
# =============================================================================

if __name__ == "__main__":
    DATASET_DIR = "./IED Dataset"
    OUTPUT_DIR = "./extracted_features"
    RESULTS_CSV = "./final_results_table.csv"
    RESULTS_TXT = "./final_results_table.txt"

    print("================================================================================")
    print(" UNIFIED CONFORMER FEATURE EXTRACTOR & MULTI-MODEL CLASSIFIER PIPELINE")
    print("================================================================================")

    # Step 1: Feature Extraction Pipeline (Run if feature directory empty or requested)
    npy_files = glob.glob(os.path.join(OUTPUT_DIR, "*_features.npy"))
    if not npy_files:
        print("\nStep 1: Parsing Disfluency Annotations...")
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
    else:
        print(f"\nStep 1: Pre-extracted feature dataset found in '{OUTPUT_DIR}' ({len(npy_files)} files).")

    # Step 2: Model Training & Evaluation Pipeline
    print("\nStep 2: Starting Multi-Model Classifier Evaluation Engine...")
    runner = ClassifierRunner(data_dir=OUTPUT_DIR)
    results = runner.run_all_experiments()

    # Step 3: Print & Save Final Results Table
    print("\nStep 3: Displaying & Exporting Final Results Table...")
    table_str = runner.format_results_table(results)
    print("\n" + table_str + "\n")

    runner.export_results(results, csv_path=RESULTS_CSV, txt_path=RESULTS_TXT)
    print("\nPipeline Execution Complete!")
