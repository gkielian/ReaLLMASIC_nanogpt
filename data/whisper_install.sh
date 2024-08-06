#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "This script will install whisper.cpp"

# Check if whisper.cpp exists, prompt to initialize submodule
if [ ! -d "$script_dir/template/whisper.cpp/models" ]; then
    read -p "whisper.cpp module not initialized. Download with git? [Y/n] " response
    response=${response,,} # Convert to lowercase

    if [[ "$response" != "n" ]]; then
        # Initialize whisper.cpp submodule
        git submodule update --init --recursive
    else
        echo "Exiting. Whisper.cpp module required."
        exit 1
    fi
fi

pushd "$script_dir/template/whisper.cpp" > /dev/null

# Download one of the Whisper models and convert to ggml format
bash ./models/download-ggml-model.sh base.en

make clean
make

popd > /dev/null

echo "whisper.cpp module installed successfully."
