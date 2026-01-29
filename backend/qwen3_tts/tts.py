import torch
from qwen_tts import Qwen3TTSModel
import pynvml
import time
import numpy as np
import re
import psutil

class Qwen3TTS:
    def __init__(self, model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", batch_size=4):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_name)
        self.batch_size = batch_size
        self._init_pynvml()
        self.is_processing = False

    def _init_pynvml(self):
        if self.device.startswith("cuda"):
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.device.split(":")[1]))
            except pynvml.NVMLError:
                self.handle = None
        else:
            self.handle = None

    def _load_model(self, model_name):
        print(f"Loading model: {model_name} on device: {self.device}")
        try:
            model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=self.device,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        except (ImportError, RuntimeError):
            print("flash_attention_2 not found or not supported, using default attention mechanism.")
            model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=self.device,
                dtype=torch.bfloat16,
            )
        return model

    def get_gpu_info(self):
        if not self.handle:
            return None
        
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        
        return {
            "memory_total_gb": round(info.total / (1024**3), 2),
            "memory_free_gb": round(info.free / (1024**3), 2),
            "memory_used_gb": round(info.used / (1024**3), 2),
            "gpu_utilization_percent": utilization.gpu,
            "memory_utilization_percent": utilization.memory,
        }

    def get_system_info(self):
        return {
            "cpu_utilization_percent": psutil.cpu_percent(),
            "ram_utilization_percent": psutil.virtual_memory().percent,
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        }

    def _chunk_text(self, text):
        # Split text by sentences, keeping the delimiters.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s]

    def generate(self, text, language, speaker, instruct=""):
        self.is_processing = True
        start_time = time.time()
        
        text_chunks = self._chunk_text(text)
        
        all_wavs = []
        
        gpu_info_before = self.get_gpu_info()

        for i in range(0, len(text_chunks), self.batch_size):
            batch = text_chunks[i:i+self.batch_size]
            
            # Prepare batch parameters
            languages = [language] * len(batch)
            speakers = [speaker] * len(batch)
            instructs = [instruct] * len(batch)

            wavs, sr = self.model.generate_custom_voice(
                text=batch,
                language=languages,
                speaker=speakers,
                instruct=instructs,
            )
            all_wavs.extend(wavs)

        # Concatenate all audio chunks into a single audio array
        if all_wavs:
            final_wav = np.concatenate(all_wavs)
        else:
            final_wav = np.array([])
            sr = self.model.config.sampling_rate

        gpu_info_after = self.get_gpu_info()
        end_time = time.time()
        
        self.is_processing = False
        return final_wav, sr, {
            "generation_time": end_time - start_time,
            "gpu_info_before": gpu_info_before,
            "gpu_info_after": gpu_info_after,
            "num_chunks": len(text_chunks),
            "num_batches": (len(text_chunks) + self.batch_size - 1) // self.batch_size
        }

    def __del__(self):
        if self.handle:
            pynvml.nvmlShutdown()

if __name__ == '__main__':
    tts = Qwen3TTS()
    text_to_speak = "This is the first sentence. This is the second sentence! And this is a third one? Let's add a fourth one to make a batch. This is the fifth sentence, starting a new batch. The sixth. The seventh. And the eighth."
    
    final_wav, sr, stats = tts.generate(text_to_speak, "English", "Ryan")
    
    import soundfile as sf
    output_filename = "output/merged_output.wav"
    sf.write(output_filename, final_wav, sr)
    print(f"Saved merged audio to {output_filename}")
        
    print(stats)