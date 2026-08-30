"""
MFCC + SDC Feature Extractor
IED Disfluency Dataset

Feature configuration
---------------------
Sample rate : 16000 Hz
Window      : 25 ms
Hop         : 10 ms
MFCC        : 13
Energy      : 1
Base dim    : 14

SDC configuration : 14-1-2-7
SDC dimension     : 210

Final dimension   : 224
"""

import numpy as np
import librosa

from base_feature_extractor import BaseFeatureExtractor


class MFCCSDCExtractor(BaseFeatureExtractor):
    """
    MFCC + Shifted Delta Cepstral (SDC) feature extractor.

    Inherits the common dataset-processing pipeline from
    BaseFeatureExtractor.

    The extractor-specific work is:
        1. Slice the requested audio segment.
        2. Compute 13 MFCC coefficients.
        3. Compute frame energy.
        4. Combine them into 14-dimensional base features.
        5. Compute SDC features.
        6. Return a fixed 224-dimensional feature vector.
    """

    name = "MFCC_SDC"

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        win_ms: float = 25.0,
        hop_ms: float = 10.0,
        sdc_N: int = 14,
        sdc_d: int = 1,
        sdc_p: int = 2,
        sdc_k: int = 7,
    ):
        """
        Parameters
        ----------
        sample_rate : int
            Audio sampling rate.

        n_mfcc : int
            Number of MFCC coefficients.

        win_ms : float
            Analysis window size in milliseconds.

        hop_ms : float
            Frame hop size in milliseconds.

        sdc_N : int
            Number of dimensions in the base feature vector.

        sdc_d : int
            Delta step.

        sdc_p : int
            Shift between successive SDC blocks.

        sdc_k : int
            Number of shifts on either side of the current frame.
        """

        super().__init__(sample_rate=sample_rate)

        self.n_mfcc = n_mfcc

        self.win_length = int(self.sample_rate * win_ms / 1000)

        self.hop_length = int(self.sample_rate * hop_ms / 1000)

        self.sdc_N = sdc_N
        self.sdc_d = sdc_d
        self.sdc_p = sdc_p
        self.sdc_k = sdc_k

        # 13 MFCC + 1 energy
        self.base_dim = self.n_mfcc + 1

        # With k=7, the implementation uses
        # 2*k + 1 shifted blocks.
        self.sdc_blocks = 2 * self.sdc_k + 1

        self.sdc_dim = self.base_dim * self.sdc_blocks

        self.feature_dim = self.base_dim + self.sdc_dim

    def setup(self) -> None:
        """
        No model needs to be loaded for MFCC + SDC.

        librosa performs the signal-processing operations
        during extract_segment().
        """
        pass

    def _compute_base_features(self, segment: np.ndarray) -> np.ndarray:
        """
        Compute the 14-dimensional base feature sequence.

        Returns
        -------
        np.ndarray
            Shape: (14, T)

            13 MFCC coefficients
            + 1 energy feature
        """

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )

        # Frame-level energy
        energy = librosa.feature.rms(
            y=segment,
            frame_length=self.win_length,
            hop_length=self.hop_length,
        )

        # Make sure both have the same number of frames
        min_frames = min(mfcc.shape[1], energy.shape[1])

        mfcc = mfcc[:, :min_frames]
        energy = energy[:, :min_frames]

        # 13 MFCC + 1 energy
        base_features = np.vstack([mfcc, energy])

        return base_features.astype(np.float32)

    def _compute_sdc(self, base_features: np.ndarray) -> np.ndarray:
        """
        Compute Shifted Delta Cepstral features.

        Parameters
        ----------
        base_features : np.ndarray
            Shape: (14, T)

        Returns
        -------
        np.ndarray
            Shape: (210, T)

        Configuration:
            N = 14
            d = 1
            p = 2
            k = 7

        This implementation concatenates 15 shifted delta blocks:
            -7 ... 0 ... +7

        Therefore:
            14 × 15 = 210 dimensions.
        """

        N, T = base_features.shape

        # Calculate first-order delta features.
        delta = librosa.feature.delta(
            base_features,
            width=3,
            order=1,
            axis=1,
        )

        shifted_blocks = []

        # Shifts from -7 to +7
        for shift_index in range(-self.sdc_k, self.sdc_k + 1):
            shift = shift_index * self.sdc_p

            shifted = np.zeros_like(delta)

            if shift == 0:
                shifted = delta

            elif shift > 0:
                # Future shift
                shifted[:, :-shift] = delta[:, shift:]

                # Replicate last frame at the boundary
                shifted[:, -shift:] = delta[:, -1:]

            else:
                # Past shift
                amount = abs(shift)

                shifted[:, amount:] = delta[:, :-amount]

                # Replicate first frame at the boundary
                shifted[:, :amount] = delta[:, :1]

            shifted_blocks.append(shifted)

        # Concatenate all 15 blocks
        sdc = np.concatenate(shifted_blocks, axis=0)

        # Expected:
        # 14 × 15 = 210
        assert sdc.shape[0] == self.sdc_dim, (
            f"Unexpected SDC dimension: {sdc.shape[0]}, expected {self.sdc_dim}"
        )

        return sdc.astype(np.float32)

    def extract_segment(
        self, audio: np.ndarray, start: float, end: float
    ) -> np.ndarray:
        """
        Extract MFCC + SDC features from one annotated segment.

        Parameters
        ----------
        audio : np.ndarray
            Full audio signal.
            Shape: (n_samples,)

        start : float
            Segment start time in seconds.

        end : float
            Segment end time in seconds.

        Returns
        -------
        np.ndarray
            Shape: (224,)

            14 base MFCC+energy features
            +
            210 SDC features
            =
            224 dimensions
        """

        # ---------------------------------------------------------
        # 1. Slice the audio segment
        # ---------------------------------------------------------

        start_sample = int(start * self.sample_rate)

        end_sample = int(end * self.sample_rate)

        segment = audio[start_sample:end_sample]

        if len(segment) == 0:
            raise ValueError(f"Empty audio segment: {start:.3f}s - {end:.3f}s")

        # ---------------------------------------------------------
        # 2. Compute 14-dimensional base features
        # ---------------------------------------------------------

        base_features = self._compute_base_features(segment)

        # ---------------------------------------------------------
        # 3. Compute 210-dimensional SDC
        # ---------------------------------------------------------

        sdc_features = self._compute_sdc(base_features)

        # ---------------------------------------------------------
        # 4. Combine base + SDC
        # ---------------------------------------------------------

        combined = np.concatenate([base_features, sdc_features], axis=0)

        # combined shape:
        # (224, T)

        # ---------------------------------------------------------
        # 5. Mean-pool over time
        # ---------------------------------------------------------
        #
        # BaseFeatureExtractor requires ONE fixed-size
        # vector per AnnotationRow:
        #
        #     shape = (D,)
        #
        # Therefore we average across frames.
        # ---------------------------------------------------------

        feature_vector = combined.mean(axis=1)

        feature_vector = feature_vector.astype(np.float32)

        # ---------------------------------------------------------
        # 6. Final dimension check
        # ---------------------------------------------------------

        assert feature_vector.shape == (self.feature_dim,), (
            f"Unexpected feature shape: "
            f"{feature_vector.shape}, "
            f"expected ({self.feature_dim},)"
        )

        return feature_vector
