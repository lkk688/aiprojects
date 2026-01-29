from fastapi import FastAPI, Form, HTTPException, Response, BackgroundTasks, UploadFile, File
from .qwen3_tts.tts import Qwen3TTS
from .qwen_asr_api.asr import Qwen3ASR
from .whisper_asr.asr import WhisperASR
import io
import soundfile as sf
from pydantic import BaseModel
import os
import numpy as np
import asyncio
import datetime
import tempfile

app = FastAPI()

tts_model = Qwen3TTS()
try:
    qwen_asr_model = Qwen3ASR()
except ValueError as e:
    print(e)
    qwen_asr_model = None

whisper_asr_model = WhisperASR()


class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    speaker: str = "Ryan"
    instruct: str = ""

async def heartbeat():
    while True:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gpu_info = tts_model.get_gpu_info()
        system_info = tts_model.get_system_info()
        
        print(f"--- Heartbeat [{now}] ---")
        if gpu_info:
            print(f"GPU Usage: {gpu_info['gpu_utilization_percent']}% | Memory: {gpu_info['memory_used_gb']}/{gpu_info['memory_total_gb']} GB")
        else:
            print("GPU Info: Not Available")
            
        print(f"CPU Usage: {system_info['cpu_utilization_percent']}% | RAM: {system_info['ram_used_gb']}/{system_info['ram_total_gb']} GB")
        print(f"Major work running: {tts_model.is_processing}")
        print("-" * (25 + len(now)))
        
        await asyncio.sleep(120) # 2 minutes

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(heartbeat())

@app.get("/")
def read_root():
    return {"status": "online"}

@app.get("/gpu_info")
def get_gpu_info():
    gpu_info = tts_model.get_gpu_info()
    if gpu_info is None:
        return {"message": "No GPU available or an error occurred."}
    return gpu_info

@app.get("/system_info")
def get_system_info():
    return tts_model.get_system_info()


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        wav, sr, stats = tts_model.generate(
            request.text, request.language, request.speaker, request.instruct
        )
        
        # --- For debugging: save the file to disk ---
        sf.write("output/debug_output.wav", wav, sr, format='WAV')
        # -----------------------------------------

        # In-memory buffer to hold the audio file
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        
        # Get the bytes from the buffer
        wav_bytes = buffer.getvalue()
        
        # We add the stats to the response headers
        headers = {f"X-TTS-Stats-{k}": str(v) for k, v in stats.items()}
        
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qwen_asr")
async def audio_to_text_qwen(audio: UploadFile = File(...)):
    if not qwen_asr_model:
        raise HTTPException(status_code=500, detail="Qwen ASR model is not available. Please provide a DashScope API key.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        
        result = qwen_asr_model.transcribe(tmp_path)
        
        os.unlink(tmp_path)
        
        if result:
            return result
        else:
            raise HTTPException(status_code=500, detail="Transcription failed.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/whisper_asr")
async def audio_to_text_whisper(audio: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        
        text = whisper_asr_model.transcribe(tmp_path)
        
        os.unlink(tmp_path)
        
        if text:
            return {"text": text}
        else:
            raise HTTPException(status_code=500, detail="Transcription failed.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # This block is for running the script directly, e.g., for debugging.
    # It's not used when running with uvicorn.
    print("Running in debug mode. For production, use 'uvicorn backend.main:app'")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)