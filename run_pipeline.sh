#!/bin/bash
# ==============================================================================
# PLEASE EDIT STEP 3 TO CONFIGURE THE SCRIPT BEFORE RUNNING!!
# ==============================================================================
# Speech-Speaker-Recognition Batch Processor
#
# This script processes all .mp3 files in a specified directory. For each file,
# it splits it into smaller chunks, runs the speech-pipeline on each chunk,
# and then merges the resulting SRT files into a single, time-corrected output.
# ==============================================================================

set -e

# =====================================================
# 1. Make the script executable
# =====================================================
chmod +x "$0"

# =====================================================
# 2. Setup .env file
# =====================================================
# Create .env from example if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "Creating .env from .env.example..."
    cp ".env.example" ".env"
fi

# Load Defaults from .env file
if [ -f ".env" ]; then
    echo "Loading default variables from .env file..."
    set -o allexport
    source ".env"
    set +o allexport
fi

# =====================================================
# 3. USER CONFIGURATION
# =====================================================
# --- EDIT THE VARIABLES BELOW ---
# These variables will override any values set in the .env file.

# 1. (Required) Your Hugging Face API token, refer to README.md of this project to see the guideline
export HUGGINGFACE_TOKEN="your_token_here"

# 2. The absolute path to the Speech-Speaker-Recognition project directory, default to current file's directory.
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

# 3. (Required) Default input directory containing .mp3 files. Can be overridden by the 1st command-line argument.
INPUT_AUDIO_DIR=""

# 4. Default output directory, default to INPUT_AUDIO_DIR/output Can be overridden by the 2nd command-line argument.
OUTPUT_DIR="${INPUT_AUDIO_DIR}/output"

# 5. Minimum number of speakers. Can be overridden by the 3rd command-line argument.
MIN_SPEAKERS=2

# 6. Maximum number of speakers. Can be overridden by the 4th command-line argument.
MAX_SPEAKERS=3

# 7. Chunk length in seconds. Can be overridden by the 5th command-line argument.
CHUNK_LENGTH=60

# --- END OF USER CONFIGURATION ---
# =====================================================

export TORCH_AUDIO_BACKEND="soundfile"

# Override config with command-line arguments (highest precedence)
INPUT_AUDIO_DIR=${1:-$INPUT_AUDIO_DIR}
OUTPUT_DIR=${2:-$OUTPUT_DIR}
MIN_SPEAKERS=${3:-$MIN_SPEAKERS}
MAX_SPEAKERS=${4:-$MAX_SPEAKERS}
CHUNK_LENGTH=${5:-$CHUNK_LENGTH}

# =====================================================
# 4. Validate Configuration
# =====================================================
if [ "$HUGGINGFACE_TOKEN" == "your_token_here" ] || [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "ERROR: Hugging Face token not set. Edit this script to set the HUGGINGFACE_TOKEN variable."
    echo "You must also accept the pyannote license at: https://huggingface.co/pyannote/speaker-diarization-3.1"
    exit 1
fi

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: Project source directory not found at '$SRC_DIR'. Please edit the variable."
    exit 1
fi

if [ -z "$INPUT_AUDIO_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <path/to/input_directory> <path/to/output_directory> [min_speakers] [max_speakers] [chunk_length_in_seconds]"
    echo "Alternatively, edit the INPUT_AUDIO_DIR, OUTPUT_DIR, MIN_SPEAKERS, MAX_SPEAKERS, and CHUNK_LENGTH variables inside the script."
    exit 1
fi

if [ ! -d "$INPUT_AUDIO_DIR" ]; then
    echo "ERROR: Input directory '$INPUT_AUDIO_DIR' not found."
    exit 1
fi

# Check for ffmpeg installation
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found. Attempting to install..."
    OS="$(uname -s)"
    case "${OS}" in
        Linux*)
            if command -v apt-get >/dev/null 2>&1; then
                sudo apt-get update && sudo apt-get install -y ffmpeg
            elif command -v yum >/dev/null 2>&1; then
                sudo yum install -y ffmpeg
            elif command -v dnf >/dev/null 2>&1; then
                sudo dnf install -y ffmpeg
            elif command -v pacman >/dev/null 2>&1; then
                sudo pacman -S --noconfirm ffmpeg
            elif command -v apk >/dev/null 2>&1; then
                sudo apk add ffmpeg
            else
                echo "Could not detect package manager. Please install ffmpeg manually."
                exit 1
            fi
            ;;
        Darwin*)
            if command -v brew >/dev/null 2>&1; then
                brew install ffmpeg
            else
                echo "Homebrew not found. Please install Homebrew or ffmpeg manually."
                exit 1
            fi
            ;;
        CYGWIN*|MINGW*|MSYS*)
            if command -v choco >/dev/null 2>&1; then
                choco install ffmpeg
            elif command -v winget >/dev/null 2>&1; then
                winget install ffmpeg
            else
                echo "Please install ffmpeg manually on Windows."
                exit 1
            fi
            ;;
        *)
            echo "Unsupported OS: ${OS}. Please install ffmpeg manually."
            exit 1
            ;;
    esac

    # Verify installation
    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "ffmpeg installation failed. Please install it manually."
        exit 1
    fi
fi

# =====================================================
# 5. Setup Environment
# =====================================================
echo "Changing to source directory: $SRC_DIR"
cd "$SRC_DIR"

if ! command -v uv >/dev/null 2>&1; then
   echo "uv not found. Installing..."
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
fi

echo "Syncing dependencies..."
uv sync

echo "Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    echo "Error: Virtual environment activation script not found."
    exit 1
fi

# =====================================================
# 6. Process All MP3 Files in Directory
# =====================================================
for mp3_file in "$INPUT_AUDIO_DIR"/*.mp3; do
    # If no mp3 files are found, the loop will run once with a non-existent path
    if [ ! -f "$mp3_file" ]; then
        echo "No .mp3 files found in $INPUT_AUDIO_DIR. Exiting."
        continue
    fi

    echo "-----------------------------------------------------------------"
    echo "Processing file: $mp3_file"
    echo "-----------------------------------------------------------------"

    FILENAME=$(basename "$mp3_file")
    BASENAME="${FILENAME%.*}"
    MERGED_DIR="$OUTPUT_DIR/final_srt"

    CHUNK_DIR="$SRC_DIR/chunks_temp"
    SRT_DIR="$SRC_DIR/srt_temp"
    mkdir -p "$CHUNK_DIR" "$SRT_DIR" "$MERGED_DIR"

    echo "Splitting $mp3_file into ${CHUNK_LENGTH}s chunks..."
    ffmpeg -i "$mp3_file" -f segment -segment_time "$CHUNK_LENGTH" -c copy "$CHUNK_DIR/${BASENAME}_%03d.mp3"

    echo "Running speech-pipeline on each chunk..."
    for chunk in $(ls "$CHUNK_DIR/${BASENAME}"_*.mp3 | sort); do
        [[ -f "$chunk" ]] || continue
        CHUNK_BASENAME=$(basename "$chunk" .mp3)
        echo "→ Processing: $CHUNK_BASENAME"
        speech-pipeline process "$chunk" --output "$SRT_DIR/${CHUNK_BASENAME}.srt" --min-speakers "$MIN_SPEAKERS" --max-speakers "$MAX_SPEAKERS"
    done

    # --- Merge SRTs with corrected timestamps ---
    OUTPUT_FILE="$MERGED_DIR/$BASENAME.srt"
    echo "Merging all SRTs into $OUTPUT_FILE"
    > "$OUTPUT_FILE"

    counter=1
    offset_ms=0

    to_ms() { local t=$1; local h=${t:0:2} m=${t:3:2} s=${t:6:2} ms=${t:9:3}; echo $((10#$h*3600000 + 10#$m*60000 + 10#$s*1000 + 10#$ms)); }
    to_time() { local t=$1; local h=$((t/3600000)) m=$(((t%3600000)/60000)) s=$(((t%60000)/1000)) ms=$((t%1000)); printf "%02d:%02d:%02d,%03d" $h $m $s $ms; }

    for srt_file in $(ls "$SRT_DIR/${BASENAME}"_*.srt | sort); do
        echo "Adding $srt_file"
        chunk_mp3_file="$CHUNK_DIR/$(basename "$srt_file" .srt).mp3"
        if [ ! -f "$chunk_mp3_file" ]; then continue; fi

        while IFS= read -r line; do
            if [[ "$line" =~ ^[0-9]+$ ]]; then
                echo "$counter" >> "$OUTPUT_FILE"; ((counter++))
            elif [[ "$line" =~ ^([0-9:,]+)\ --\>\ ([0-9:,]+)$ ]]; then
                start_ms=$(to_ms "${BASH_REMATCH[1]}")
                end_ms=$(to_ms "${BASH_REMATCH[2]}")
                new_start_ms=$((start_ms + offset_ms))
                new_end_ms=$((end_ms + offset_ms))
                echo "$(to_time $new_start_ms) --> $(to_time $new_end_ms)" >> "$OUTPUT_FILE"
            else
                echo "$line" >> "$OUTPUT_FILE"
            fi
        done < "$srt_file"
        echo "" >> "$OUTPUT_FILE"

        chunk_duration_ms=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$chunk_mp3_file" | awk '{printf("%d",$1*1000)}')
        offset_ms=$((offset_ms + chunk_duration_ms))
    done

    echo "Merged SRT saved to: $OUTPUT_FILE"

    # --- Clean up temporary files for the processed mp3 ---
    echo "Cleaning up temporary files for $BASENAME..."
    rm -rf "$CHUNK_DIR"
    rm -rf "$SRT_DIR"

    echo "Finished processing $BASENAME."
done

echo "======================================"
echo "All files have been processed."
