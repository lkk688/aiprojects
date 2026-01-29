#!/bin/bash

BASE_URL="http://127.0.0.1:8000"
AUDIO_FILE="output/test_output.wav"

echo "### Testing Whisper ASR Endpoint with $AUDIO_FILE ###"

curl -X POST "$BASE_URL/whisper_asr" \
    -F "audio=@$AUDIO_FILE;type=audio/wav" \
    --verbose
