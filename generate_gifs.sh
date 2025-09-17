#!/bin/bash

# Default FPS value
FPS=50

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fps)
            FPS="$2"
            shift 2
            ;;
        *)
            DIR="$1"
            shift
            ;;
    esac
done

# Check if directory is provided as argument
if [ -z "$DIR" ]; then
    echo "Please provide directory path as argument"
    exit 1
fi

# Create temporary file for sorted filenames
find "$DIR" -name "*.png" | sort -V > sorted_files.txt

# Create a file with input files in FFmpeg format
sed 's/^/file /' sorted_files.txt > ffmpeg_files.txt

# Generate GIF using convert
convert -delay $(echo "100/$FPS" | bc) -loop 0 $(cat sorted_files.txt) "$DIR/animation.gif"

#compress gif using gifsicle
gifsicle -O3 --lossy=30 "$DIR/animation.gif" -o "$DIR/animation_compressed.gif"

# Clean up temporary files
rm sorted_files.txt ffmpeg_files.txt
rm "$DIR/animation.gif"
