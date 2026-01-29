# AI Process Documentation

This document summarizes the steps taken to set up the project environment, from initializing a Git repository to running a text-to-speech (TTS) example and creating a FastAPI backend with both Qwen and Whisper ASR capabilities.

## 1. Git Repository Setup

To track changes and manage the project's source code, a Git repository was initialized.

### Commands Used:

```bash
# 1. Initialize a new Git repository
git init

# 2. Stage all existing files for the initial commit
git add .

# 3. Create the first commit
git commit -m "Initial commit"

# 4. Check the status of the working tree
git status
```

## 2. Qwen3-TTS Sample Code

A sample Python script was created to demonstrate the capabilities of the `Qwen3-TTS` model.

### Steps and Commands:

1.  **Install Dependencies:** The necessary Python packages were installed into the `py312` conda environment.

    ```bash
    conda run -n py312 pip install -U qwen-tts
    ```

2.  **Create the Example Script:** A file named `qwen3_tts_example.py` was created with Python code to load the `Qwen3-TTS` model and generate audio from text.

3.  **Run the Script:** The script was executed to generate the sample audio files.

    ```bash
    conda run -n py312 python qwen3_tts_example.py
    ```

## 3. Installing SoX and FFmpeg in the HPC Conda Environment

During the execution of the Python script, a warning indicated that `sox` (Sound eXchange) was not found. Later, a `pydub` warning highlighted a missing `ffmpeg` dependency. These are crucial for audio processing.

### How to Install SoX

To install `sox` into the existing `py312` conda environment, you can use the `conda` package manager, preferably sourcing the package from the `conda-forge` channel.

#### Recommended Command:

```bash
conda install -n py312 -c conda-forge sox
```

### How to Install FFmpeg

To install `ffmpeg` for audio processing (e.g., for `pydub`), use the following command:

```bash
conda install -n py312 -c conda-forge ffmpeg
```

## 4. FastAPI Backend for TTS and ASR

A FastAPI backend was created to serve both TTS and ASR models over an API, including a heartbeat for system monitoring.

### Project Structure

```
aiprojects/
├── backend/
│   ├── qwen3_tts/
│   │   ├── __init__.py
│   │   └── tts.py
│   ├── qwen_asr_api/  # Renamed from qwen3_asr to avoid naming conflicts
│   │   ├── __init__.py
│   │   └── asr.py
│   ├── whisper_asr/
│   │   ├── __init__.py
│   │   └── asr.py
│   ├── main.py
│   ├── test_qwen_asr_api.sh
│   ├── test_whisper_api.sh
│   └── ...
└── ...
```

### `qwen3_tts` Library

A library was created to encapsulate the TTS logic, including:
*   Model loading.
*   Text chunking and batching for long inputs.
*   GPU monitoring.

Key dependencies installed:
```bash
conda run -n py312 pip install pynvml numpy fastapi uvicorn python-multipart
```

### FastAPI Application

A FastAPI application was created in `backend/main.py` with the following endpoints:
*   `GET /`: Health check.
*   `GET /gpu_info`: Returns GPU memory and utilization information.
*   `GET /system_info`: Returns CPU and RAM utilization information.
*   `POST /tts`: Accepts text and returns the synthesized audio.
*   `POST /qwen_asr`: Accepts an audio file and returns the transcription using the Qwen ASR API (requires a DashScope API key).
*   `POST /whisper_asr`: Accepts an audio file and returns the transcription using a local Whisper model.

### Running the Server

To run the FastAPI server, use the following command from the project root:

```bash
conda run -n py312 uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## 5. Troubleshooting Audio File Download

A problem was encountered where the audio files downloaded via `curl` were corrupted (7 bytes). After extensive debugging, it was determined that the issue was likely with the client-side `curl` environment, as the server was confirmed to be generating and sending the audio data correctly. The solution was to use a different client or investigate the local `curl` configuration. A Python script `test_whisper_api.py` was created to perform reliable API testing.

## 6. Integrating Local Whisper ASR

To add local ASR capabilities, the `openai/whisper-large-v3` model was integrated into the backend.

### Model Export to ONNX

For performance optimization, the Whisper model was exported to the ONNX format using the `optimum` library.

1.  **Install Dependencies:**

    ```bash
    conda run -n py312 pip install "optimum[onnxruntime-gpu]"
    ```

2.  **Export Script:** An export script `export_whisper_onnx.py` was created to handle the conversion.

3.  **Run Export:** The export was performed by running the script:

    ```bash
    conda run -n py312 python export_whisper_onnx.py
    ```
    This created the `whisper_onnx_model` directory with the optimized model.

### Whisper ASR Library

A new library `backend/whisper_asr/asr.py` was created to load the ONNX model and perform transcriptions.

### Testing the Whisper API

A new test script `backend/test_whisper_api.sh` was created to test the `/whisper_asr` endpoint.

To run the test script:
```bash
./backend/test_whisper_api.sh
```
This script sends an audio file from the `output` folder to the `/whisper_asr` endpoint and attempts to print the transcription.

## 7. Addressing Import Errors and Dependency Conflicts

Several critical issues arose related to Python module imports and package dependencies.

### Resolving `ModuleNotFoundError` for Qwen ASR

Initially, `ModuleNotFoundError: No module named 'qwen3_tts'` (and similar for ASR) occurred because Python couldn't find the custom modules within the `backend` directory.

*   **Attempted `sys.path` modification:** A temporary fix was attempted by modifying `sys.path` in `backend/main.py`. This was later deemed an improper solution.
*   **Renaming Local Directory:** The local directory `backend/qwen3_asr` was renamed to `backend/qwen_asr_api` to avoid a naming conflict with the installed `qwen3-asr-toolkit` package.
*   **Correcting Imports in `backend/main.py`:** Import statements in `backend/main.py` were changed to relative imports (e.g., `from .qwen_asr_api.asr import Qwen3ASR`). This is the standard practice for modules within a package.
*   **Correcting Imports in `backend/qwen_asr_api/asr.py`:** The internal import within `backend/qwen_asr_api/asr.py` for the installed `qwen3-asr-toolkit` was corrected from `from qwen3_asr import transcribe` to `from qwen3_asr_toolkit.qwen3asr import QwenASR`, reflecting the actual structure of the installed package.
*   **Verification of `qwen3-asr-toolkit`:** Repeated checks confirmed that `qwen3-asr-toolkit` was installed in the environment, but Python's import mechanism was struggling to locate it correctly, even after reinstallation. The final correction to the import statement within `qwen_asr_api/asr.py` should resolve this.

### Whisper Model Loading and GPU Incompatibility

After successfully exporting the `whisper-large-v3` model to ONNX, issues arose during its loading:

*   **`pydub` `RuntimeWarning`:** The `pydub` library issued a warning about not finding `ffmpeg` or `avconv`, which was resolved by installing `ffmpeg` via `conda`.
*   **ONNX Model File Naming:** The `optimum` library initially failed to find the ONNX files because it expected a `decoder_model_merged.onnx` file, but the export produced separate `encoder_model.onnx`, `decoder_model.onnx`, and `decoder_with_past_model.onnx` files. This was fixed by explicitly specifying these filenames in `backend/whisper_asr/asr.py`.
*   **`ConnectionResetError` (Server Crash):** The server repeatedly crashed with `ConnectionResetError` during Whisper transcription. This was attributed to potential Out of Memory (OOM) issues due to the large size of `whisper-large-v3`.
*   **Quantization Attempt and `NotImplemented` Error:** To mitigate OOM, the Whisper ONNX model was quantized (`export_whisper_onnx_quantized.py`) using `int8` dynamic quantization. However, loading this quantized model with `CPUExecutionProvider` resulted in a `NotImplemented` error for `ConvInteger` operations, indicating incompatibility.
*   **Reverted to Unquantized with CUDA:** The `WhisperASR` was reverted to use the unquantized model (`whisper_onnx_model`) with `CUDAExecutionProvider` to determine if the base model was compatible. The model initialized successfully, confirming that the issue was specific to the quantization process's compatibility with the execution providers. `ConnectionResetError` (OOM) still occurs during actual inference with the unquantized model on GPU, confirming severe resource constraints for `whisper-large-v3` on the current system.

### VibeVoice Integration and Dependency Conflicts

An attempt to integrate `VibeVoice` TTS services led to severe dependency conflicts:

*   **Installation of `vibevoice`:** Installing `vibevoice` through `pip` successfully installed the package.
*   **Downgrade of Core Libraries:** `vibevoice` forced a downgrade of `accelerate` and `transformers` to older versions (`accelerate==1.6.0`, `transformers==4.51.3`).
*   **Incompatibility:** These older versions conflict directly with the requirements of `qwen-tts` and `trl` (which require `accelerate==1.12.0` and `transformers>=4.56.1` respectively), rendering them non-functional if `vibevoice` is present in the same environment.
*   **Conclusion:** Running `vibevoice` alongside `qwen-tts` and `trl` in the same environment is currently infeasible due to these irreconcilable dependency conflicts. Options were presented to the user, including using separate environments or abandoning `VibeVoice` integration.

## 8. Output Folder

A dedicated `output` folder was created in the project root to store all generated audio files, and the code was modified to direct future outputs into this folder.

### Command Used:
```bash
mkdir output
mv *.wav output/
mv backend/*.wav output/
```
The following files were updated to save output to `output/`:
*   `qwen3_tts_example.py`
*   `backend/qwen3_tts/tts.py`
*   `backend/main.py`
*   `backend/test_api.sh`
*   `backend/test_qwen_asr_api.sh`
*   `backend/test_whisper_api.sh`
