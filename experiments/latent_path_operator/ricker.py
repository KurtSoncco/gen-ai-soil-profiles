"""Ricker outcrop velocity and the time-domain traces it drives."""

from __future__ import annotations

import numpy as np

N_TIME = 256
DT = 4.0 / N_TIME


def time_axis(n_time: int = N_TIME, dt: float = DT) -> np.ndarray:
    return np.arange(n_time) * dt


def positive_frequencies(n_time: int = N_TIME, dt: float = DT) -> np.ndarray:
    """rFFT bins excluding DC. The operator is singular at ω = 0."""
    return np.fft.rfftfreq(n_time, dt)[1:]


def ricker(time: np.ndarray, f_peak: float, t0: float | None = None) -> np.ndarray:
    """Velocity wavelet (1 - 2 π² f_p² τ²) exp(-π² f_p² τ²), peak at ``t0``."""
    if t0 is None:
        t0 = 1.0 / f_peak
    tau = np.asarray(time, dtype=np.float64) - t0
    phase = (np.pi * f_peak * tau) ** 2
    return (1.0 - 2.0 * phase) * np.exp(-phase)


def ricker_spectrum(
    freq: np.ndarray, f_peak: float, t0: float | None = None
) -> np.ndarray:
    """Integral Fourier transform of :func:`ricker`, convention e^{-2π i f t}."""
    if t0 is None:
        t0 = 1.0 / f_peak
    freq = np.asarray(freq, dtype=np.float64)
    amp = np.zeros(freq.shape, dtype=np.complex128)
    live = freq > 0
    amp[live] = (
        2.0
        * freq[live] ** 2
        / (f_peak**3 * np.sqrt(np.pi))
        * np.exp(-((freq[live] / f_peak) ** 2))
    )
    amp *= np.exp(-2j * np.pi * freq * t0)
    return amp


def rfft_coefficients(
    freq: np.ndarray, f_peak: float, dt: float, t0: float | None = None
) -> np.ndarray:
    """Spectrum divided by ``dt``, which is what ``irfft`` expects for this grid."""
    return ricker_spectrum(freq, f_peak, t0) / dt


def velocity_traces(
    displacement: np.ndarray,
    wavelet: np.ndarray,
    n_time: int = N_TIME,
) -> np.ndarray:
    """Inverse FFT of transfer values times the Ricker spectrum.

    ``displacement`` is (F, M) complex on the positive rFFT bins.
    ``wavelet`` is (F,) complex rFFT coefficients. Returns (M, n_time).
    """
    n_bins = displacement.shape[0] + 1
    spectrum = np.zeros((n_bins, displacement.shape[1]), dtype=np.complex128)
    spectrum[1:] = displacement * wavelet[:, None]
    return np.fft.irfft(spectrum, n=n_time, axis=0).T


def peak_ratio(predicted: np.ndarray, exact: np.ndarray) -> float:
    return float(np.max(predicted) / np.max(exact))


def peak_lag_ms(predicted: np.ndarray, exact: np.ndarray, dt: float = DT) -> float:
    return float((np.argmax(predicted) - np.argmax(exact)) * dt * 1000.0)
