"""Backend API and frontend static-file routing for Birdsong Cleaner."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
import numpy as np
import soundfile as sf

from signals_systems.processing import create_plots, enhance_audio

app = FastAPI(title="Birdsong Cleaner")
PROJECT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_DURATION_SECONDS = 40


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
        bandpassed, cleaned, sos = enhance_audio(mono, sample_rate)
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
    if output_sample_rate != sample_rate:
        from scipy.signal import resample_poly

        divisor = np.gcd(sample_rate, output_sample_rate)
        cleaned_for_playback = resample_poly(
            cleaned_for_playback,
            output_sample_rate // divisor,
            sample_rate // divisor,
        )
    output_buffer = BytesIO()
    sf.write(
        output_buffer,
        cleaned_for_playback,
        output_sample_rate,
        format="WAV",
        subtype="PCM_16",
    )

    result = {
        "filename": Path(file.filename or "uploaded-audio").name,
        "sample_rate": sample_rate,
        "output_sample_rate": output_sample_rate,
        "channels": audio.shape[1],
        "duration": duration,
        "plots": plots,
        "cleaned_audio": base64.b64encode(output_buffer.getvalue()).decode("ascii"),
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


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
