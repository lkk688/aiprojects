import os
from qwen3_asr_toolkit.qwen3asr import QwenASR # Corrected import

class Qwen3ASR:
    def __init__(self, api_key=None):
        """
        Initializes the Qwen3ASR client.
        
        Args:
            api_key (str, optional): Your DashScope API key. 
                                     If not provided, it will be read from the 
                                     DASHSCOPE_API_KEY environment variable.
        """
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
            
        if not self.api_key:
            raise ValueError("DashScope API key not found. Please provide it as an argument or set the DASHSCOPE_API_KEY environment variable.")
            
        os.environ["DASHSCOPE_API_KEY"] = self.api_key
        self._qwen_asr_client = QwenASR() # Instantiate the client

    def transcribe(self, audio_path: str):
        """
        Transcribes the given audio file.
        
        Args:
            audio_path (str): The path to the audio file.
            
        Returns:
            dict: The transcription result.
        """
        try:
            language, text = self._qwen_asr_client.asr(wav_url=audio_path) # Corrected call
            return {"language": language, "text": text}
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            return None

if __name__ == '__main__':
    # This is an example of how to use the Qwen3ASR class.
    # You need to have a valid DashScope API key and an audio file.
    
    # Example usage:
    #asr = Qwen3ASR()
    #result = asr.transcribe("path/to/your/audio.wav")
    #if result:
    #    print(result['text'])
    
    print("Qwen3ASR class is defined. Please use it within the FastAPI application.")
    print("You need to set the DASHSCOPE_API_KEY environment variable.")
