#!/bin/bash

export no_proxy="127.0.0.1,localhost"

BASE_URL="http://127.0.0.1:8000"

echo "### Testing Health Check Endpoint ###"
curl "$BASE_URL/"
echo -e "\n"

echo "### Testing GPU Info Endpoint ###"
curl "$BASE_URL/gpu_info"
echo -e "\n"


echo "### Testing TTS Endpoint ###"
TEXT="Hello, this is a test of our new text-to-speech API."
LANGUAGE="English"
SPEAKER="Ryan"
OUTPUT_FILE="output/test_output.wav"

# Use a heredoc to pass the JSON data to curl
curl --http1.1 -L \
    -H "Content-Type: application/json" \
    -d @- \
    -o $OUTPUT_FILE --verbose "$BASE_URL/tts" <<EOF
{
    "text": "$TEXT",
    "language": "$LANGUAGE",
    "speaker": "$SPEAKER"
}
EOF

echo "TTS audio saved to $OUTPUT_FILE"
