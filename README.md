# Songbird - Birdsong Cleaner

Upload a short field recording to reduce persistent background sound, compare
the original and bird-focused audio, inspect the generated signal plots, and
download the cleaned WAV.

## Project structure

```text
backend/
  app.py                         FastAPI API and static frontend routing
frontend/
  index.html                     Hero page, upload UI, player, and graphs
signals_systems/
  processing.py                  Reusable IIR filter, STFT gate, and graphs
  Birdsong_Extraction_Signals_Systems.py
                                 Runnable syllabus demonstration
  README.md                      Equations and course-concept guide
requirements.txt                 Runtime and demo dependencies
pyproject.toml                   Vercel Python entrypoint configuration
```

## Run locally

From the project root:

```powershell
python -m pip install -r requirements-dev.txt
python -m uvicorn backend.app:app --reload
```

Open <http://127.0.0.1:8000>. The frontend calls the same-origin
`POST /api/process` backend route.

## Deploy to Vercel

1. Put this project in a Git repository and import it as a Vercel project.
2. Set the Vercel **Root Directory** to the repository root (the folder
   containing `index.py`, `pyproject.toml`, and `requirements.txt`), not one
   of its three code folders.
3. Vercel detects the conventional root `index.py` FastAPI entrypoint and
   installs the runtime packages from `requirements.txt`. `.python-version`
   selects Python 3.12.
4. Deploy. The FastAPI static mount serves `frontend/` at `/`; API routes are
   registered before that mount.

After deployment, verify that `/api/health` responds with `{"status":"ok"}`.
If the page still fails, check Vercel **Project → Settings → General →
Root Directory** and set it to `.` (repository root), then check the latest
deployment's **Build Logs** and **Runtime Logs**. Confirm that the deployment
commit contains `index.py`, `backend/app.py`, `signals_systems/processing.py`,
`frontend/index.html`, and `requirements.txt`.

Alternatively, install the Vercel CLI and deploy from this directory:

```powershell
npm install --global vercel
vercel
```

The serverless uploader enforces Vercel's 4.5 MB function payload limit with a
4 MB file cap and a 40-second audio limit. Cleaned audio is resampled to mono
22.05 kHz when necessary so the audio and graph response remain within the
serverless response-size limit. **No `.env` file or environment variables are
required**; the app has no configurable secrets or external API keys.

## Run the course demonstration

With the supplied example MP3 in the project root:

```powershell
python signals_systems\Birdsong_Extraction_Signals_Systems.py
```

It prints the recording/sample-rate details, visualizes Fourier/Welch spectrum
and the Butterworth response, then plots and plays the noisy and cleaned
recording. It saves `birdsong_cleaned.wav` in the project root.

## Syllabus connection

- **Fourier transform / DFT:** Welch power spectrum and the STFT spectrograms.
- **Discrete-time LTI systems:** the Butterworth IIR filter shapes the input
  according to its impulse response.
- **Z-transform and frequency response:** the filter is represented by a
  rational transfer function `H(z)`; the plotted response shows its passband
  and attenuation.
- **STFT:** estimates persistent background energy across time-frequency bins
  and applies a spectral gate.

The result is bird-focused enhancement, not perfect source separation. Bird
calls and traffic can overlap in frequency, so background may remain and quiet
calls may be attenuated. See [signals_systems/README.md](./signals_systems/README.md)
for the equations and method notes.
