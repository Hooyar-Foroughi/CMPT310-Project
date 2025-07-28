"""
This script filters WAV and RTTM files based on the number of speakers.
"""

import os
import shutil

# Constants for speaker count filtering
MIN_SPEAKERS = 2
MAX_SPEAKERS = 5

RTTM_DIR = "../data/rttm"
WAV_DIR = "../data/wav"
REMOVED_RTTM_DIR = "../data/removed/rttm"
REMOVED_WAV_DIR = "../data/removed/wav"

# Create directories if they don't exist
os.makedirs(REMOVED_RTTM_DIR, exist_ok=True)
os.makedirs(REMOVED_WAV_DIR, exist_ok=True)

def count_speakers(rttm_path):
    speakers = set()
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                speakers.add(parts[7])
    return len(speakers)

for rttm_file in os.listdir(RTTM_DIR):
    if not rttm_file.endswith(".rttm"):
        continue

    rttm_path = os.path.join(RTTM_DIR, rttm_file)
    num_speakers = count_speakers(rttm_path)

    if num_speakers < MIN_SPEAKERS or num_speakers > MAX_SPEAKERS:
        print(f"[REMOVE] {rttm_file} with {num_speakers} speakers")

        # Move RTTM file
        shutil.move(rttm_path, os.path.join(REMOVED_RTTM_DIR, rttm_file))

        # Move corresponding WAV file if exists
        wav_filename = rttm_file.replace(".rttm", ".wav")
        wav_path = os.path.join(WAV_DIR, wav_filename)
        if os.path.exists(wav_path):
            shutil.move(wav_path, os.path.join(REMOVED_WAV_DIR, wav_filename))