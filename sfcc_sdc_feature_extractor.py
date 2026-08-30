from typing import Union, Optional
import numpy as np
from scipy.fftpack import dct

from base_feature_extractor import BaseFeatureExtractor

EPS: float = float(np.finfo(np.float32).eps)


class SFCCSDCExtractor(BaseFeatureExtractor):
    """
    Sub-band Frequency Cepstral Coefficients (SFCC) and Shifted Delta Cepstral (SDC)
    feature extractor for speech disfluency and interruption event detection.

    This class implements a complete manual SFCC feature extraction pipeline coupled
    with Shifted Delta Cepstral (SDC) computation and temporal mean pooling to produce
    a single 1-D feature vector per audio segment.

    Parameters
    ----------
    sample_rate : int, default=16000
        Sampling rate of input audio signals in Hz.
    n_sfcc : int, default=13
        Number of static SFCC cepstral coefficients to keep per frame (N).
    n_bands : int, default=24
        Number of sub-band frequency filters in the filterbank.
    n_fft : int, default=512
        FFT size for Short-Time Fourier Transform computation.
    hop_length : int, default=160
        Frame step (hop length) in samples.
    win_length : int, default=400
        Frame window length in samples.
    pre_emphasis_coeff : float, default=0.97
        Pre-emphasis filter coefficient.
    sdc_d : int, default=1
        Advance/delay parameter d for delta computation in SDC.
    sdc_p : int, default=3
        Shift step parameter P between consecutive delta evaluations in SDC.
    sdc_k : int, default=7
        Number of delta blocks k stacked in SDC representation.
    include_static : bool, default=True
        Whether to concatenate static SFCC coefficients with SDC features.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_sfcc: int = 13,
        n_bands: int = 24,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        pre_emphasis_coeff: float = 0.97,
        sdc_d: int = 1,
        sdc_p: int = 3,
        sdc_k: int = 7,
        include_static: bool = True,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        self.n_sfcc: int = n_sfcc
        self.n_bands: int = n_bands
        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.win_length: int = win_length
        self.pre_emphasis_coeff: float = pre_emphasis_coeff
        self.sdc_d: int = sdc_d
        self.sdc_p: int = sdc_p
        self.sdc_k: int = sdc_k
        self.include_static: bool = include_static

        self._window: Optional[np.ndarray] = None
        self._filterbank: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        """
        Unique identifier name for this feature extractor module.

        Returns
        -------
        str
            Extractor name string ("SFCC_SDC").
        """
        return "SFCC_SDC"

    def setup(self) -> None:
        """
        Pre-compute and initialize reusable resources for feature extraction.

        Pre-calculates the Hamming window function and the linear/logarithmic sub-band
        frequency filterbank spanning from 0 Hz to the Nyquist frequency.
        """
        self._window = np.hamming(self.win_length).astype(np.float32)
        n_freqs = self.n_fft // 2 + 1
        self._filterbank = self._create_subband_filterbank(
            n_freqs=n_freqs,
            n_bands=self.n_bands,
            sample_rate=self.sample_rate,
        )

    def extract_segment(
        self,
        audio: np.ndarray,
        start: Union[float, int],
        end: Union[float, int],
    ) -> np.ndarray:
        """
        Extract fixed-length SFCC + SDC feature vector for a segment of audio.

        Parameters
        ----------
        audio : np.ndarray
            1-D mono float32 audio waveform array sampled at `self.sample_rate`.
        start : Union[float, int]
            Segment start marker in seconds (float) or sample index (int).
        end : Union[float, int]
            Segment end marker in seconds (float) or sample index (int).

        Returns
        -------
        np.ndarray
            1-D float64 numpy array of shape (D,) representing the mean-pooled feature vector.
        """
        if isinstance(start, float) or isinstance(end, float):
            start_sample = int(round(start * self.sample_rate))
            end_sample = int(round(end * self.sample_rate))
        else:
            start_sample = int(start)
            end_sample = int(end)

        start_sample = max(0, min(start_sample, len(audio)))
        end_sample = max(start_sample, min(end_sample, len(audio)))

        segment = audio[start_sample:end_sample]

        feature_dim = (
            self.n_sfcc * (1 + self.sdc_k)
            if self.include_static
            else self.n_sfcc * self.sdc_k
        )

        if len(segment) == 0:
            return np.zeros(feature_dim, dtype=np.float64)

        # 1. Pre-emphasis
        emphasized = self._pre_emphasis(segment)

        # 2. Framing
        frames = self._frame_signal(emphasized)
        if frames.shape[0] == 0:
            return np.zeros(feature_dim, dtype=np.float64)

        # 3. Windowing (Hamming)
        windowed_frames = self._apply_window(frames)

        # 4. STFT / FFT & Power spectrum
        power_spec = self._compute_power_spectrum(windowed_frames)

        # 5. Sub-band energy computation
        subband_energies = self._compute_subband_energies(power_spec)

        # 6. Log compression & DCT-II (SFCC extraction)
        sfcc_raw = self._compute_sfcc(subband_energies)

        # 7. Cepstral Mean Normalization (CMN)
        sfcc_cmn = self._apply_cmn(sfcc_raw)

        # 8. Shifted Delta Cepstral (SDC) computation
        sdc_features = self._compute_sdc(sfcc_cmn)

        # 9. Feature concatenation
        if self.include_static:
            combined_frames = np.hstack((sfcc_cmn, sdc_features))
        else:
            combined_frames = sdc_features

        # 10. Temporal mean pooling
        feature_vector = self._mean_pool(combined_frames)

        return feature_vector

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

    def _pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply a first-order high-pass pre-emphasis filter to boost high frequencies.

        Parameters
        ----------
        signal : np.ndarray
            1-D input audio signal.

        Returns
        -------
        np.ndarray
            1-D pre-emphasized signal.
        """
        if len(signal) == 0:
            return signal
        return np.append(signal[0], signal[1:] - self.pre_emphasis_coeff * signal[:-1])

    def _frame_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Split a 1-D signal into overlapping frames.

        Parameters
        ----------
        signal : np.ndarray
            1-D pre-emphasized signal array.

        Returns
        -------
        np.ndarray
            2-D array of shape (n_frames, win_length).
        """
        signal_length = len(signal)
        if signal_length < self.win_length:
            pad_amount = self.win_length - signal_length
            signal = np.pad(signal, (0, pad_amount), mode="constant")
            signal_length = len(signal)

        n_frames = 1 + int(np.floor((signal_length - self.win_length) / self.hop_length))
        if n_frames <= 0:
            return np.empty((0, self.win_length), dtype=np.float32)

        indices = (
            np.tile(np.arange(0, self.win_length), (n_frames, 1))
            + np.tile(np.arange(0, n_frames * self.hop_length, self.hop_length), (self.win_length, 1)).T
        )
        frames = signal[indices].astype(np.float32)
        return frames

    def _apply_window(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply pre-computed Hamming window to each frame.

        Parameters
        ----------
        frames : np.ndarray
            2-D framed signal matrix of shape (n_frames, win_length).

        Returns
        -------
        np.ndarray
            2-D windowed frames of shape (n_frames, win_length).
        """
        if self._window is None:
            self._window = np.hamming(self.win_length).astype(np.float32)
        return frames * self._window

    def _compute_power_spectrum(self, windowed_frames: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform (STFT) power spectrum via FFT.

        Parameters
        ----------
        windowed_frames : np.ndarray
            2-D array of windowed frames of shape (n_frames, win_length).

        Returns
        -------
        np.ndarray
            Power spectrum matrix of shape (n_frames, n_fft // 2 + 1).
        """
        mag_spectrum = np.abs(np.fft.rfft(windowed_frames, n=self.n_fft))
        power_spectrum = (1.0 / self.n_fft) * (mag_spectrum ** 2.0)
        return power_spectrum

    def _create_subband_filterbank(
        self,
        n_freqs: int,
        n_bands: int,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Construct sub-band frequency triangular filterbank across uniform spectral sub-bands.

        Parameters
        ----------
        n_freqs : int
            Number of FFT frequency bins (n_fft // 2 + 1).
        n_bands : int
            Number of sub-band filters to construct.
        sample_rate : int
            Sampling rate of the audio in Hz.

        Returns
        -------
        np.ndarray
            Sub-band filterbank matrix of shape (n_bands, n_freqs).
        """
        # Linearly / sub-band spaced frequency band boundaries
        low_freq = 0.0
        high_freq = sample_rate / 2.0
        band_points = np.linspace(low_freq, high_freq, n_bands + 2)

        fft_freqs = np.linspace(0.0, high_freq, n_freqs)
        filterbank = np.zeros((n_bands, n_freqs), dtype=np.float64)

        for b in range(n_bands):
            f_left = band_points[b]
            f_center = band_points[b + 1]
            f_right = band_points[b + 2]

            up_slope = (fft_freqs - f_left) / (f_center - f_left + EPS)
            down_slope = (f_right - fft_freqs) / (f_right - f_center + EPS)

            filterbank[b] = np.maximum(0.0, np.minimum(up_slope, down_slope))

        # Equal-energy sub-band normalization
        band_widths = band_points[2:] - band_points[:-2]
        enorm = 2.0 / (band_widths + EPS)
        filterbank *= enorm[:, np.newaxis]

        return filterbank

    def _compute_subband_energies(self, power_spec: np.ndarray) -> np.ndarray:
        """
        Compute energy for each frequency sub-band frame-by-frame.

        Parameters
        ----------
        power_spec : np.ndarray
            Power spectrum matrix of shape (n_frames, n_freqs).

        Returns
        -------
        np.ndarray
            Sub-band energy matrix of shape (n_frames, n_bands).
        """
        if self._filterbank is None:
            self.setup()
        assert self._filterbank is not None

        if self._filterbank.shape[1] != power_spec.shape[1]:
            fb = self._create_subband_filterbank(
                n_freqs=power_spec.shape[1],
                n_bands=self.n_bands,
                sample_rate=self.sample_rate,
            )
        else:
            fb = self._filterbank

        subband_energies = np.dot(power_spec, fb.T)
        return subband_energies

    def _compute_sfcc(self, subband_energies: np.ndarray) -> np.ndarray:
        """
        Compute raw SFCC features via log energy compression and DCT-II.

        Parameters
        ----------
        subband_energies : np.ndarray
            Sub-band energy matrix of shape (n_frames, n_bands).

        Returns
        -------
        np.ndarray
            Static SFCC matrix of shape (n_frames, n_sfcc).
        """
        log_energies = np.log(subband_energies + EPS)
        # DCT-II across sub-band dimension (axis=1)
        sfcc_all = dct(log_energies, type=2, axis=1, norm="ortho")
        return sfcc_all[:, : self.n_sfcc]

    def _apply_cmn(self, sfcc_frames: np.ndarray) -> np.ndarray:
        """
        Perform Cepstral Mean Normalization (CMN) across frame sequence.

        Parameters
        ----------
        sfcc_frames : np.ndarray
            Raw SFCC matrix of shape (n_frames, n_sfcc).

        Returns
        -------
        np.ndarray
            Mean-normalized SFCC matrix of shape (n_frames, n_sfcc).
        """
        if sfcc_frames.shape[0] == 0:
            return sfcc_frames
        mean_vector = np.mean(sfcc_frames, axis=0, keepdims=True)
        return sfcc_frames - mean_vector

    def _compute_sdc(self, sfcc_frames: np.ndarray) -> np.ndarray:
        """
        Compute Shifted Delta Cepstral (SDC) features.

        SDC vector at frame t concatenates k delta vectors evaluated at shifts i * P:
        Delta_c(t + i*P) = c(t + i*P + d) - c(t + i*P - d) for i = 0 ... k-1.

        Parameters
        ----------
        sfcc_frames : np.ndarray
            Cepstral feature matrix of shape (n_frames, N).

        Returns
        -------
        np.ndarray
            Stacked SDC matrix of shape (n_frames, N * k).
        """
        n_frames, N = sfcc_frames.shape
        d = self.sdc_d
        P = self.sdc_p
        k = self.sdc_k

        max_lookahead = (k - 1) * P + d
        padded_sfcc = np.pad(sfcc_frames, ((d, max_lookahead), (0, 0)), mode="edge")

        sdc_list = []
        for i in range(k):
            shift = i * P
            t_plus = d + shift + d
            t_minus = d + shift - d
            delta_i = (
                padded_sfcc[t_plus : t_plus + n_frames, :]
                - padded_sfcc[t_minus : t_minus + n_frames, :]
            )
            sdc_list.append(delta_i)

        sdc_matrix = np.hstack(sdc_list)
        return sdc_matrix

    def _mean_pool(self, frames: np.ndarray) -> np.ndarray:
        """
        Perform temporal mean pooling over frame-level feature matrix.

        Parameters
        ----------
        frames : np.ndarray
            Combined feature matrix of shape (n_frames, total_features).

        Returns
        -------
        np.ndarray
            1-D aggregated feature vector of shape (total_features,).
        """
        if frames.shape[0] == 0:
            return np.zeros(frames.shape[1], dtype=np.float64)
        return np.mean(frames, axis=0, dtype=np.float64)
