#!/bin/bash

export no_proxy="127.0.0.1,localhost"

if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "WARNING: DASHSCOPE_API_KEY environment variable is not set."
    echo "The ASR API call is expected to fail."
fi

BASE_URL="http://127.0.0.1:8000"
AUDIO_FILE="output/test_output.wav"

echo "### Testing Qwen ASR Endpoint with $AUDIO_FILE ###"

curl -X POST "$BASE_URL/qwen_asr" \
    -F "audio=@$AUDIO_FILE;type=audio/wav" \
    --verbose