"""Enhance bird calls in a noisy environmental recording.

Install dependencies with:
    python -m pip install numpy scipy matplotlib soundfile ipython

The input MP3 is decoded with libsndfile through python-soundfile. Output is
saved beside this script as a WAV file.
"""

# Run the # %% cells in order in VS Code, or execute this file as a script.

# %% [markdown]
# Birdsong Extraction from Environmental Noise
#
# This project processes the supplied recording of morning birds with
# environmental background noise. It uses a Butterworth IIR band-pass filter
# followed by STFT spectral gating to emphasize bird calls.
#
# Syllabus links:
# - Fourier transform / DFT: inspect the recording in the frequency domain.
# - Discrete-time LTI systems: the Butterworth filter is an IIR system whose
#   output is the convolution of its impulse response with the input.
# - Z-transform: the filter has a rational transfer function H(z); `sosfiltfilt`
#   applies it forward and backward for offline, near-zero-phase processing.
# - STFT: track and attenuate background energy over time and frequency.
#
# Filtering enhances bird-like frequency components; it cannot perfectly
# separate bird calls from traffic or other sounds that overlap them.

# %%
from pathlib import Path
import sys

PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd()
)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from signals_systems.processing import (
    FILTER_ORDER,
    HIGHCUT_HZ,
    LOWCUT_HZ,
    STFT_FRAME_SIZE,
    STFT_OVERLAP,
    enhance_audio,
)

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from IPython.display import Audio, display
from scipy import signal

plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["axes.grid"] = True

# Keep the example recording and generated WAV at the project root.
INPUT_AUDIO = PROJECT_DIR / "loswin23-morning-birds-chirping-499430.mp3"
OUTPUT_AUDIO = PROJECT_DIR / "birdsong_cleaned.wav"

# %% [markdown]
# 1. Load the supplied recording
#
# Stereo channels are averaged to a mono discrete-time signal. The original
# sample rate is retained so that playback speed and pitch are unchanged.

# %%
if not INPUT_AUDIO.is_file():
    raise FileNotFoundError(
        f"Input recording not found: {INPUT_AUDIO}\n"
        "Place the uploaded MP3 beside this script or update INPUT_AUDIO."
    )

audio, fs = sf.read(INPUT_AUDIO, dtype="float64", always_2d=True)
if audio.shape[0] == 0:
    raise ValueError(f"Input recording contains no samples: {INPUT_AUDIO}")

noisy = audio.mean(axis=1)
duration = len(noisy) / fs
t = np.arange(len(noisy)) / fs
print(
    f"Input: {INPUT_AUDIO.name} | {fs} Hz | "
    f"{audio.shape[1]} channel(s) | {duration:.2f} s"
)


# %% [markdown]
# 2. Inspect the noisy input using the Fourier transform
#
# The Welch estimate averages short, windowed periodograms (DFT magnitude
# squared) to show where signal energy is concentrated.

# %%
frequencies, power = signal.welch(noisy, fs=fs, nperseg=min(8192, len(noisy)))
plt.figure()
plt.semilogy(frequencies, np.maximum(power, 1e-15))
plt.xlim(0, min(12000, fs / 2))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power spectral density")
plt.title("Noisy recording — Welch frequency spectrum")
plt.tight_layout()
plt.show()


# %% [markdown]
# 3. Method A — Butterworth IIR band-pass filtering
#
# A band-pass filter attenuates low-frequency rumble/traffic and frequencies
# above the selected bird-call band. Its rational Z-transform response is
# designed as second-order sections for numerical stability. Forward-backward
# filtering is non-causal, so this is an offline processing method.

# %%
bandpassed, filtered, sos = enhance_audio(noisy, fs)
highcut = min(HIGHCUT_HZ, 0.95 * (fs / 2))

response_frequencies, response = signal.sosfreqz(sos, worN=4096, fs=fs)
plt.figure()
plt.plot(
    response_frequencies,
    20 * np.log10(np.maximum(np.abs(response), 1e-7)),
)
plt.axvline(LOWCUT_HZ, linestyle="--", label=f"Low cutoff: {LOWCUT_HZ:.0f} Hz")
plt.axvline(highcut, linestyle="--", label=f"High cutoff: {highcut:.0f} Hz")
plt.ylim(-80, 5)
plt.xlim(0, fs / 2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Butterworth band-pass frequency response")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# 4. Method B — STFT spectral noise reduction
#
# The shared processor estimates a background spectrum from a low percentile
# of each frequency bin and attenuates it with an STFT spectral gate.


# %% [markdown]
# 5. Compare noisy input and bird-focused output
#
# The two spectrograms use the same scale to make frequency/time changes
# visible. The output is an enhanced recording, not a guaranteed isolated stem.

# %%
def plot_spectrogram(x, sample_rate, title, vmin=-100, vmax=-25):
    frame_size = min(STFT_FRAME_SIZE, len(x))
    overlap = min(STFT_OVERLAP, frame_size - 1)
    frequencies, frame_times, spectrum = signal.stft(
        x,
        fs=sample_rate,
        nperseg=frame_size,
        noverlap=overlap,
    )
    magnitude_db = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-7))
    mesh = plt.pcolormesh(
        frame_times,
        frequencies,
        magnitude_db,
        shading="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    plt.ylim(0, min(12000, sample_rate / 2))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(mesh, label="Magnitude (dB)")


fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, sharey=True)
plt.sca(axes[0])
plot_spectrogram(noisy, fs, "Original noisy recording")
plt.sca(axes[1])
plot_spectrogram(filtered, fs, "Bird-focused output after IIR filter + STFT gate")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
axes[0].plot(t, noisy, linewidth=0.5)
axes[0].set_title("Noisy input waveform")
axes[1].plot(t, filtered, linewidth=0.5)
axes[1].set_title("Cleaned bird-focused waveform")
axes[1].set_xlabel("Time (s)")
for axis in axes:
    axis.set_ylabel("Amplitude")
plt.tight_layout()
plt.show()


# %% [markdown]
# 6. Save and listen to the result
#
# WAV is used for the cleaned output to avoid adding further lossy MP3
# compression. Peak normalization is only for convenient playback; it does not
# alter the filtering or spectral analysis above.

# %%
peak = np.max(np.abs(filtered))
if peak <= 1e-12:
    raise ValueError("The cleaned output is silent; check the filter settings.")

playback = 0.98 * filtered / peak
sf.write(OUTPUT_AUDIO, playback, fs, subtype="PCM_16")
print(f"Saved cleaned audio: {OUTPUT_AUDIO}")

print("Original noisy audio:")
display(Audio(noisy, rate=fs))
print("Cleaned bird-focused audio:")
display(Audio(playback, rate=fs))


# %% [markdown]
# Results and limitations
#
# Listen to both audio players and inspect the spectrograms. The band-pass
# filter suppresses energy outside the chosen bird-call band; the STFT gate
# reduces persistent spectral background. Since no isolated clean reference
# is available, clean-reference SNR, RMSE, and correlation are not reported.
# Traffic, wind, and birds can overlap in frequency, so the result may retain
# some background sound or remove quiet bird calls. Tune the band edges and
# gate parameters for a different recording.
