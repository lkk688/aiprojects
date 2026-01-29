import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa # Still needed for loading audio
import os # For checking if audio file exists

class WhisperASR:
    def __init__(self, model_id="openai/whisper-large-v3"):
        """
        Initializes the WhisperASR model using Hugging Face transformers pipeline.
        """
        print("Initializing WhisperASR with Hugging Face pipeline...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        print("WhisperASR initialized.")

    def transcribe(self, audio_path: str):
        """
        Transcribes the given audio file using the Hugging Face pipeline.
        
        Args:
            audio_path (str): The path to the audio file.
            
        Returns:
            str: The transcribed text.
        """
        print(f"Starting transcription for: {audio_path}")
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return None
            
        try:
            # The pipeline handles loading and preprocessing
            print("Running transcription pipeline...")
            result = self.pipe(audio_path)
            print("Transcription pipeline finished.")
            
            return result["text"]
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == '__main__':
    print("WhisperASR class is defined. Please use it within the FastAPI application.")