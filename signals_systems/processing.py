"""Signals & Systems: filtering, spectral gating, and plot generation."""

from __future__ import annotations

import base64
from io import BytesIO
import threading

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

LOWCUT_HZ = 1500.0
HIGHCUT_HZ = 9000.0
FILTER_ORDER = 5
STFT_FRAME_SIZE = 2048
STFT_OVERLAP = 1536
NOISE_PERCENTILE = 20
NOISE_ATTENUATION = 1.4
GAIN_FLOOR = 0.08

_PLOT_LOCK = threading.Lock()


def spectral_gate(
    x: np.ndarray,
    sample_rate: int,
    frame_size: int = STFT_FRAME_SIZE,
    overlap: int = STFT_OVERLAP,
    noise_percentile: float = NOISE_PERCENTILE,
    attenuation: float = NOISE_ATTENUATION,
    gain_floor: float = GAIN_FLOOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce persistent background energy using an STFT spectral gate."""
    reduced, frame_times, gain, _ = _spectral_gate_with_details(
        x,
        sample_rate,
        frame_size,
        overlap,
        noise_percentile,
        attenuation,
        gain_floor,
    )
    return reduced, frame_times, gain


def _spectral_gate_with_details(
    x: np.ndarray,
    sample_rate: int,
    frame_size: int = STFT_FRAME_SIZE,
    overlap: int = STFT_OVERLAP,
    noise_percentile: float = NOISE_PERCENTILE,
    attenuation: float = NOISE_ATTENUATION,
    gain_floor: float = GAIN_FLOOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Apply the spectral gate and retain statistics from its actual STFT."""
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Audio must be a non-empty mono signal.")
    if not 0 <= noise_percentile < 100:
        raise ValueError("noise_percentile must be in the range [0, 100).")
    if attenuation < 0:
        raise ValueError("attenuation must be non-negative.")
    if not 0 < gain_floor <= 1:
        raise ValueError("gain_floor must be in the range (0, 1].")
    if frame_size < 2 or not 0 <= overlap < frame_size:
        raise ValueError("overlap must be non-negative and smaller than frame_size.")

    actual_frame_size = min(frame_size, x.size)
    actual_overlap = min(overlap, actual_frame_size - 1)
    _, frame_times, spectrum = signal.stft(
        x,
        fs=sample_rate,
        nperseg=actual_frame_size,
        noverlap=actual_overlap,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(spectrum)
    noise_profile = np.percentile(
        magnitude, noise_percentile, axis=1, keepdims=True
    )
    estimated_noise = attenuation * noise_profile
    gain = np.maximum(
        1.0 - estimated_noise / np.maximum(magnitude, 1e-12),
        gain_floor,
    )
    gain = median_filter(gain, size=(1, 5), mode="nearest")
    _, reduced = signal.istft(
        spectrum * gain,
        fs=sample_rate,
        nperseg=actual_frame_size,
        noverlap=actual_overlap,
        input_onesided=True,
        boundary=True,
    )
    noise_min, noise_median, noise_max = np.percentile(
        noise_profile, (0, 50, 100)
    )
    example_bin, example_frame = np.unravel_index(
        np.argmax(magnitude), magnitude.shape
    )
    example_magnitude = float(magnitude[example_bin, example_frame])
    example_noise_profile = float(noise_profile[example_bin, 0])
    example_estimated_noise = attenuation * example_noise_profile
    example_raw_gain = max(
        1.0
        - example_estimated_noise / max(example_magnitude, 1e-12),
        gain_floor,
    )
    details: dict[str, float | int] = {
        "frame_size": actual_frame_size,
        "overlap": actual_overlap,
        "hop_size": actual_frame_size - actual_overlap,
        "frequency_bins": int(magnitude.shape[0]),
        "time_frames": int(magnitude.shape[1]),
        "noise_percentile": noise_percentile,
        "noise_attenuation": attenuation,
        "gain_floor": gain_floor,
        "noise_magnitude_min": float(noise_min),
        "noise_magnitude_median": float(noise_median),
        "noise_magnitude_max": float(noise_max),
        "gain_min": float(np.min(gain)),
        "gain_median": float(np.median(gain)),
        "gain_max": float(np.max(gain)),
        "attenuated_percent": float(np.mean(gain < 0.999) * 100),
        "gain_floor_percent": float(
            np.mean(gain <= gain_floor + 1e-12) * 100
        ),
        "example_frequency_hz": float(
            example_bin * sample_rate / actual_frame_size
        ),
        "example_time_seconds": float(frame_times[example_frame]),
        "example_magnitude": example_magnitude,
        "example_noise_profile_magnitude": example_noise_profile,
        "example_estimated_noise_magnitude": float(example_estimated_noise),
        "example_raw_gain": float(example_raw_gain),
        "example_smoothed_gain": float(gain[example_bin, example_frame]),
    }
    return reduced[: x.size], frame_times, gain, details


def enhance_audio(
    noisy: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return band-passed audio, enhanced audio, and filter SOS coefficients."""
    bandpassed, cleaned, sos, _ = enhance_audio_with_calculations(
        noisy, sample_rate
    )
    return bandpassed, cleaned, sos


def enhance_audio_with_calculations(
    noisy: np.ndarray, sample_rate: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float | int],
]:
    """Enhance audio and return measured statistics from the spectral gate."""
    if noisy.ndim != 1 or noisy.size == 0:
        raise ValueError("Audio must be a non-empty mono signal.")
    if not np.all(np.isfinite(noisy)):
        raise ValueError("Audio contains non-finite sample values.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    highcut = min(HIGHCUT_HZ, 0.95 * (sample_rate / 2))
    if not 0 < LOWCUT_HZ < highcut:
        raise ValueError(
            f"Sample rate {sample_rate} Hz is too low for the "
            f"{LOWCUT_HZ:g}-{HIGHCUT_HZ:g} Hz bird-call band."
        )

    sos = signal.butter(
        FILTER_ORDER,
        [LOWCUT_HZ, highcut],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    try:
        bandpassed = signal.sosfiltfilt(sos, noisy)
    except ValueError as exc:
        raise ValueError(
            "The audio is too short for the selected Butterworth filter."
        ) from exc

    cleaned, _, _, gate_details = _spectral_gate_with_details(
        bandpassed, sample_rate
    )
    return bandpassed, cleaned, sos, gate_details


def _figure_as_base64(fig: plt.Figure) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_plots(
    noisy: np.ndarray,
    bandpassed: np.ndarray,
    cleaned: np.ndarray,
    sample_rate: int,
    sos: np.ndarray,
) -> dict[str, str]:
    """Render processing graphs to PNG data URIs for the browser."""
    plots: dict[str, str] = {}
    with _PLOT_LOCK:
        frequencies, power = signal.welch(
            noisy, fs=sample_rate, nperseg=min(8192, noisy.size)
        )
        fig, axis = plt.subplots(figsize=(9, 3.5))
        axis.semilogy(frequencies, np.maximum(power, 1e-15), color="#4298f5")
        axis.set_xlim(0, min(12000, sample_rate / 2))
        axis.set(title="Input frequency spectrum (Welch / DFT)", xlabel="Frequency (Hz)", ylabel="Power spectral density")
        fig.tight_layout()
        plots["frequency"] = _figure_as_base64(fig)

        response_frequencies, response = signal.sosfreqz(
            sos, worN=4096, fs=sample_rate
        )
        fig, axis = plt.subplots(figsize=(9, 3.5))
        axis.plot(
            response_frequencies,
            20 * np.log10(np.maximum(np.abs(response), 1e-7)),
            color="#8c6ee8",
        )
        axis.axvline(LOWCUT_HZ, linestyle="--", color="#ef795e")
        axis.axvline(min(HIGHCUT_HZ, 0.95 * sample_rate / 2), linestyle="--", color="#ef795e")
        axis.set(
            title="Butterworth IIR filter response",
            xlabel="Frequency (Hz)",
            ylabel="Magnitude (dB)",
            ylim=(-80, 5),
            xlim=(0, sample_rate / 2),
        )
        fig.tight_layout()
        plots["filter"] = _figure_as_base64(fig)

        frame_size = min(STFT_FRAME_SIZE, noisy.size)
        overlap = min(STFT_OVERLAP, frame_size - 1)
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True, sharey=True, layout="constrained"
        )
        for axis, audio, title in (
            (axes[0], noisy, "Input spectrogram"),
            (axes[1], cleaned, "Cleaned bird-focused spectrogram"),
        ):
            _, times, spectrum = signal.stft(
                audio,
                fs=sample_rate,
                nperseg=frame_size,
                noverlap=overlap,
            )
            magnitude_db = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-7))
            mesh = axis.pcolormesh(
                times,
                np.fft.rfftfreq(frame_size, 1 / sample_rate),
                magnitude_db,
                shading="auto",
                cmap="magma",
                vmin=-100,
                vmax=-25,
            )
            axis.set_ylim(0, min(12000, sample_rate / 2))
            axis.set_ylabel("Frequency (Hz)")
            axis.set_title(title)
        axes[-1].set_xlabel("Time (s)")
        fig.colorbar(mesh, ax=axes, label="Magnitude (dB)", pad=0.02)
        plots["spectrogram"] = _figure_as_base64(fig)

        times = np.arange(noisy.size) / sample_rate
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for axis, audio, title in (
            (axes[0], noisy, "Noisy input"),
            (axes[1], bandpassed, "After Butterworth band-pass"),
            (axes[2], cleaned, "Cleaned bird-focused output"),
        ):
            stride = max(1, audio.size // 100_000)
            axis.plot(times[::stride], audio[::stride], linewidth=0.5)
            axis.set_ylabel("Amplitude")
            axis.set_title(title)
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        plots["waveform"] = _figure_as_base64(fig)

    return plots
