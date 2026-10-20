"""Backend API and frontend static-file routing for Birdsong Cleaner."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import soundfile as sf

PROJECT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from signals_systems.processing import (
    FILTER_ORDER,
    HIGHCUT_HZ,
    LOWCUT_HZ,
    create_plots,
    enhance_audio_with_calculations,
)

app = FastAPI(title="Birdsong Cleaner")
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_DURATION_SECONDS = 40


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))))


def _level_change_db(reference: float, measured: float) -> float | None:
    if reference <= 1e-12:
        return None
    return float(20 * np.log10(max(measured, 1e-12) / reference))


@app.get("/", include_in_schema=False)
def homepage() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/process")
async def process_audio(file: UploadFile = File(...)) -> dict[str, object]:
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Choose an audio file to process.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The audio file exceeds Vercel's 4 MB upload limit.",
        )

    try:
        audio, sample_rate = sf.read(
            BytesIO(payload), dtype="float64", always_2d=True
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not decode this file. Try WAV, MP3, FLAC, or another supported audio format.",
        ) from exc

    if audio.shape[0] == 0 or sample_rate <= 0:
        raise HTTPException(status_code=400, detail="The audio file is empty or invalid.")
    duration = audio.shape[0] / sample_rate
    if duration > MAX_DURATION_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=f"The recording is longer than {MAX_DURATION_SECONDS} seconds. Please upload a shorter clip.",
        )

    mono = audio.mean(axis=1)
    try:
        bandpassed, cleaned, sos, gate_details = enhance_audio_with_calculations(
            mono, sample_rate
        )
        plots = create_plots(mono, bandpassed, cleaned, sample_rate, sos)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    peak = float(np.max(np.abs(cleaned)))
    if not np.isfinite(peak) or peak <= 1e-12:
        raise HTTPException(
            status_code=422,
            detail="The processing produced silent audio. Try a different recording.",
        )

    cleaned_for_playback = 0.98 * cleaned / peak
    output_sample_rate = min(sample_rate, 22050)
    resample_up = 1
    resample_down = 1
    if output_sample_rate != sample_rate:
        from scipy.signal import resample_poly

        divisor = np.gcd(sample_rate, output_sample_rate)
        resample_up = int(output_sample_rate // divisor)
        resample_down = int(sample_rate // divisor)
        cleaned_for_playback = resample_poly(
            cleaned_for_playback,
            resample_up,
            resample_down,
        )
    output_buffer = BytesIO()
    sf.write(
        output_buffer,
        cleaned_for_playback,
        output_sample_rate,
        format="WAV",
        subtype="PCM_16",
    )

    output_payload = output_buffer.getvalue()
    output_audio, decoded_output_sample_rate = sf.read(
        BytesIO(output_payload), dtype="float64", always_2d=True
    )
    input_rms = _rms(mono)
    bandpassed_rms = _rms(bandpassed)
    cleaned_rms = _rms(cleaned)
    output_rms = _rms(output_audio)
    calculations = {
        "input": {
            "sample_count": int(audio.shape[0]),
            "sample_rate": int(sample_rate),
            "channels": int(audio.shape[1]),
            "duration_seconds": duration,
            "mono_formula": "x_mono[n] = (1 / C) * sum_c x_c[n]",
            "mono_rms": input_rms,
            "mono_peak": float(np.max(np.abs(mono))),
        },
        "filter": {
            "design": (
                "Butterworth IIR band-pass, SOS, forward-backward "
                "(zero-phase) filtering"
            ),
            "order": FILTER_ORDER,
            "low_cut_hz": LOWCUT_HZ,
            "high_cut_hz": min(HIGHCUT_HZ, 0.95 * (sample_rate / 2)),
            "sos_coefficients": sos.tolist(),
            "output_rms": bandpassed_rms,
            "rms_change_db_from_input": _level_change_db(
                input_rms, bandpassed_rms
            ),
        },
        "spectral_gate": {
            **gate_details,
            "input_rms": bandpassed_rms,
            "output_rms": cleaned_rms,
            "rms_change_db_from_bandpass": _level_change_db(
                bandpassed_rms, cleaned_rms
            ),
        },
        "output": {
            "normalization_peak_before": peak,
            "normalization_target_peak": 0.98,
            "normalization_gain": float(0.98 / peak),
            "resample_up": resample_up,
            "resample_down": resample_down,
            "sample_rate": int(decoded_output_sample_rate),
            "sample_count": int(output_audio.shape[0]),
            "duration_seconds": float(
                output_audio.shape[0] / decoded_output_sample_rate
            ),
            "rms": output_rms,
            "peak": float(np.max(np.abs(output_audio))),
            "bit_depth": 16,
            "rms_change_db_from_input": _level_change_db(input_rms, output_rms),
        },
    }

    result = {
        "filename": Path(file.filename or "uploaded-audio").name,
        "sample_rate": sample_rate,
        "output_sample_rate": output_sample_rate,
        "channels": audio.shape[1],
        "duration": duration,
        "plots": plots,
        "calculations": calculations,
        "cleaned_audio": base64.b64encode(output_payload).decode("ascii"),
        "note": (
            "This is bird-focused enhancement, not perfect source separation. "
            "Bird calls and traffic can overlap in frequency."
        ),
    }
    response_size = len(result["cleaned_audio"]) + sum(
        len(image) for image in plots.values()
    )
    if response_size > 4 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="The analysis result is too large for serverless response limits. Upload a shorter recording.",
        )
    return result


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.mount(
    "/asset",
    StaticFiles(directory=PROJECT_DIR / "asset"),
    name="asset",
)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
