# Signals & Systems implementation

`processing.py` contains the shared, executable signal path used by the
website backend and the standalone course demonstration.

## Discrete-time signal model

For a recording \(x[n]\), the app first applies a Butterworth band-pass IIR
filter. Its difference equation is

\[
y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k].
\]

This is an LTI system with rational Z-transform response

\[
H(z) = \frac{\sum_{k=0}^{M} b_k z^{-k}}
            {1 + \sum_{k=1}^{N} a_k z^{-k}}.
\]

The filter is implemented in second-order sections for numerical stability.
Forward/backward filtering (`sosfiltfilt`) removes phase delay for this offline
project; it is non-causal and is not a real-time filter.

## Fourier analysis and STFT gate

The Welch plot averages windowed DFT periodograms to estimate the input power
spectrum. The spectrogram and spectral gate use the short-time Fourier
transform:

\[
X(m,k) = \sum_n x[n]\,w[n-mR]e^{-j2\pi kn/N},
\]

where \(N\) is the frame length and \(R\) is the hop size. The gate estimates
the persistent magnitude at each frequency from a low temporal percentile,
attenuates bins near that estimate, and reconstructs the signal with the
inverse STFT. A gain floor limits over-suppression.

The estimate assumes bird calls are intermittent compared with the background.
It cannot distinguish a bird call from traffic energy occupying the same
time-frequency region. With no isolated clean reference, the app does not
report a fabricated clean-reference SNR or RMSE.

## Demonstration

Run from the project root:

```powershell
python signals_systems\Birdsong_Extraction_Signals_Systems.py
```

The demonstration loads the example MP3 from the project root and runs the
same `enhance_audio` function used by the website.
