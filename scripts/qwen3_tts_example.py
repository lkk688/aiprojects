
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Check if a CUDA-enabled GPU is available, otherwise use CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the pre-trained Qwen3-TTS model
# Using flash_attention_2 for better performance if available
try:
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
except ImportError:
    print("flash_attention_2 not found, using default attention mechanism.")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device,
        dtype=torch.bfloat16,
    )


# --- Single Inference Example ---
print("\n--- Running Single Inference Example ---")
text_to_speak = "其实我真的有发现，我是一个特别善于观察别人情绪的人。"
output_filename = "output/output_custom_voice.wav"

print(f"Generating audio for the text: '{text_to_speak}'")
wavs, sr = model.generate_custom_voice(
    text=text_to_speak,
    language="Chinese",
    speaker="Vivian",
    instruct="用特别愤怒的语气说",
)

# Save the generated audio to a file
sf.write(output_filename, wavs[0], sr)
print(f"Audio saved to {output_filename}")

# --- Batch Inference Example ---
print("\n--- Running Batch Inference Example ---")
batch_texts = [
    "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    "She said she would be here by noon."
]
batch_languages = ["Chinese", "English"]
batch_speakers = ["Vivian", "Ryan"]
batch_instructs = ["", "Very happy."]
output_filename_1 = "output/output_custom_voice_1.wav"
output_filename_2 = "output_custom_voice_2.wav"

print("Generating audio for the batch of texts...")
wavs, sr = model.generate_custom_voice(
    text=batch_texts,
    language=batch_languages,
    speaker=batch_speakers,
    instruct=batch_instructs,
)

# Save the generated audio files
sf.write(output_filename_1, wavs[0], sr)
print(f"Audio 1 saved to {output_filename_1}")
sf.write(output_filename_2, wavs[1], sr)
print(f"Audio 2 saved to {output_filename_2}")

print("\nSample code execution finished.")
