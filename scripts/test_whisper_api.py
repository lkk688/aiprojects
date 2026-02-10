import requests

def test_whisper_asr(file_path="output/test_output.wav"):
    """
    Tests the /whisper_asr endpoint by sending an audio file.
    """
    url = "http://127.0.0.1:8000/whisper_asr"
    
    with open(file_path, "rb") as f:
        files = {"audio": (file_path, f, "audio/wav")}
        
        try:
            # Set a long timeout (e.g., 300 seconds = 5 minutes)
            response = requests.post(url, files=files, timeout=300)
            
            if response.status_code == 200:
                print("Whisper ASR Transcription:")
                print(response.json())
            else:
                print(f"Error: Server returned status code {response.status_code}")
                print(response.text)
                
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_whisper_asr()