"""
mfcc_feature_extractor.py
IED Disfluency Dataset — MFCC Feature Extraction
=================================================
Converted from Untitled1.ipynb into a standalone script, following the
same structural conventions as wav2vec2_feature_extractor.py and
whisper_feature_extractor.py. All logic/behavior is identical to the
original notebook — only the organization (docstring, config/imports
sections, function + main() wrapping) has changed.

Extractor   : Librosa MFCC
Feature dim : N_MFCC (13 by default)

Pipeline
--------
1.  Discover .wav files under DATASET_DIR (and check for matching .txt
    annotation files).
2.  Sanity-test MFCC extraction + visualization on the first file.
3.  extract_and_save_mfcc() — extract MFCC features per file and save
    them (+ frame timestamps + metadata) as a compressed .npz.
4.  Save an extraction log (.csv) and a metadata summary (.csv).
5.  Spot-check one of the saved .npz files.

Output
------
  <OUTPUT_DIR>/features/**/*.npz
  <OUTPUT_DIR>/logs/mfcc_extraction_log.csv
  <OUTPUT_DIR>/mfcc_metadata.csv

Usage
-----
  python mfcc_feature_extractor.py
"""

# ============================================================
# SECTION 0 — CONFIGURATION
# All tunable parameters in one place.
# ============================================================

from pathlib import Path

# ── Data paths ────────────────────────────────────────────────
DATASET_DIR = Path(r"D:\IED_Dataset")

# MFCC features will be stored here
OUTPUT_DIR = Path(r"D:\IED_MFCC")

# Individual MFCC files
FEATURE_DIR = OUTPUT_DIR / "features"

# Logs and metadata
LOG_DIR = OUTPUT_DIR / "logs"

# ── MFCC parameters ───────────────────────────────────────────
TARGET_SR = 16000

N_MFCC = 13
N_MELS = 40
N_FFT = 512

# 25 ms window at 16 kHz
WIN_LENGTH = int(0.025 * TARGET_SR)

# 10 ms hop at 16 kHz
HOP_LENGTH = int(0.010 * TARGET_SR)

# ============================================================
# SECTION 1 — IMPORTS
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# SECTION 2 — FEATURE EXTRACTION
# ============================================================

def extract_and_save_mfcc(wav_path):
    """
    Extract MFCC features from one WAV file and save them as NPZ.

    Returns:
        status, output_path, number_of_frames, duration, error
    """

    try:
        # Create a unique output filename
        relative_path = wav_path.relative_to(DATASET_DIR)

        # Preserve folder structure
        output_path = FEATURE_DIR / relative_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already processed
        if output_path.exists():
            try:
                existing = np.load(output_path, allow_pickle=True)
                n_frames = existing["mfcc"].shape[1]
                duration = float(existing["duration"])
                existing.close()

                return (
                    "skipped",
                    str(output_path),
                    n_frames,
                    duration,
                    ""
                )
            except Exception:
                # If existing file is corrupted, recompute it
                pass

        # Load audio
        y, sr = librosa.load(
            wav_path,
            sr=TARGET_SR,
            mono=True
        )

        duration = len(y) / sr

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS
        )

        # Frame timestamps
        frame_times = librosa.frames_to_time(
            np.arange(mfcc.shape[1]),
            sr=sr,
            hop_length=HOP_LENGTH
        )

        # Corresponding TXT path
        txt_path = wav_path.with_suffix(".txt")

        # Save
        np.savez_compressed(
            output_path,
            mfcc=mfcc.astype(np.float32),
            frame_times=frame_times.astype(np.float32),
            sample_rate=np.array(sr),
            duration=np.array(duration),
            n_mfcc=np.array(N_MFCC),
            n_mels=np.array(N_MELS),
            n_fft=np.array(N_FFT),
            hop_length=np.array(HOP_LENGTH),
            win_length=np.array(WIN_LENGTH),
            audio_file=np.array(str(wav_path)),
            annotation_file=np.array(
                str(txt_path) if txt_path.exists() else ""
            )
        )

        return (
            "processed",
            str(output_path),
            mfcc.shape[1],
            duration,
            ""
        )

    except Exception as e:

        return (
            "failed",
            "",
            0,
            0,
            repr(e)
        )


# ============================================================
# SECTION 3 — MAIN
# ============================================================

def main() -> None:
    print("Libraries loaded successfully.")

    # ── Paths ──────────────────────────────────────────────────
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset :", DATASET_DIR)
    print("Output  :", OUTPUT_DIR)
    print("Features:", FEATURE_DIR)

    # ── Dataset discovery ──────────────────────────────────────
    wav_files = sorted(DATASET_DIR.rglob("*.wav"))

    print("Total WAV files found:", len(wav_files))

    if len(wav_files) > 0:
        print("\nFirst 10 files:")
        for f in wav_files[:10]:
            print(f)
    else:
        print("ERROR: No WAV files found.")

    # ── WAV / TXT pairing check ────────────────────────────────
    pairs = []
    missing_txt = []

    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")

        if txt_path.exists():
            pairs.append((wav_path, txt_path))
        else:
            missing_txt.append(wav_path)

    print("WAV files              :", len(wav_files))
    print("WAV + TXT pairs        :", len(pairs))
    print("WAV files without TXT  :", len(missing_txt))

    if missing_txt:
        print("\nFirst missing TXT files:")
        for f in missing_txt[:20]:
            print(f)

    print("MFCC parameters:")
    print("Sample rate :", TARGET_SR)
    print("N_MFCC      :", N_MFCC)
    print("N_MELS      :", N_MELS)
    print("N_FFT       :", N_FFT)
    print("Window      :", WIN_LENGTH, "samples")
    print("Hop         :", HOP_LENGTH, "samples")

    # ── Sanity test on first file ──────────────────────────────
    test_wav = wav_files[0]

    print("Testing:", test_wav)

    y, sr = librosa.load(
        test_wav,
        sr=TARGET_SR,
        mono=True
    )

    print("Loaded successfully")
    print("Original path:", test_wav)
    print("Samples      :", len(y))
    print("Sample rate  :", sr)
    print("Duration     :", len(y) / sr, "seconds")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS
    )

    print("MFCC shape:", mfcc.shape)

    print("\nInterpretation:")
    print("Number of MFCC coefficients:", mfcc.shape[0])
    print("Number of time frames      :", mfcc.shape[1])

    # ── Visualize sample MFCC ──────────────────────────────────
    plt.figure(figsize=(15, 6))

    librosa.display.specshow(
        mfcc,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mfcc"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title(f"MFCC - {test_wav.name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("MFCC Coefficient")

    plt.tight_layout()
    plt.show()

    # ── Batch extraction ────────────────────────────────────────
    results = []

    print("=" * 70)
    print("STARTING MFCC EXTRACTION")
    print("=" * 70)

    print("Total WAV files:", len(wav_files))
    print("Output folder :", FEATURE_DIR)
    print()

    for wav_path in tqdm(
        wav_files,
        desc="Extracting MFCC",
        unit="file"
    ):

        status, output_path, n_frames, duration, error = \
            extract_and_save_mfcc(wav_path)

        results.append({
            "audio_file": str(wav_path),
            "output_file": output_path,
            "status": status,
            "frames": n_frames,
            "duration_seconds": duration,
            "error": error
        })

    print("\nExtraction completed.")

    # ── Extraction log ───────────────────────────────────────────
    results_df = pd.DataFrame(results)

    log_path = LOG_DIR / "mfcc_extraction_log.csv"

    results_df.to_csv(
        log_path,
        index=False
    )

    print("Log saved to:")
    print(log_path)

    print("\nSummary:")
    print(results_df["status"].value_counts())

    # ── Metadata summary ─────────────────────────────────────────
    metadata = []

    for row in results:
        metadata.append({
            "audio_file": row["audio_file"],
            "mfcc_file": row["output_file"],
            "status": row["status"],
            "frames": row["frames"],
            "duration_seconds": row["duration_seconds"]
        })

    metadata_df = pd.DataFrame(metadata)

    metadata_path = OUTPUT_DIR / "mfcc_metadata.csv"

    metadata_df.to_csv(
        metadata_path,
        index=False
    )

    print("Metadata saved:")
    print(metadata_path)

    print("Total duration:",
          metadata_df["duration_seconds"].sum() / 3600,
          "hours")

    print("Total MFCC frames:",
          metadata_df["frames"].sum())

    # ── Spot-check a saved file ──────────────────────────────────
    saved_files = list(FEATURE_DIR.rglob("*.npz"))

    print("Saved MFCC files:", len(saved_files))

    sample_feature = saved_files[0]

    data = np.load(
        sample_feature,
        allow_pickle=True
    )

    mfcc = data["mfcc"]
    frame_times = data["frame_times"]

    print("\nFile:", sample_feature)
    print("MFCC shape:", mfcc.shape)
    print("Frame times:", frame_times.shape)
    print("Duration:", float(data["duration"]))
    print("Sample rate:", int(data["sample_rate"]))

    data.close()


if __name__ == "__main__":
    main()
