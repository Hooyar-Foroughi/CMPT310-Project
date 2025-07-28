import os
from subprocess import call
import gettimestamp



# --- Configuration ---
AUDIO_DIRECTORY = "audio"
TIMESTAMP_DIRECTORY = "timestamp"
AUDIO_FILE_EXTENSION = ".wav"
TIMESTAMP_FILE_EXTENSION = ".csv"
# -------------------

def run_timestamp_check():
    """
    Scans the audio directory for audio files, checks for corresponding
    timestamp CSVs, and runs a specified script if a CSV is missing.
    """
    print(f"Starting timestamp check...")
    print(f"Audio directory: {AUDIO_DIRECTORY}")
    print(f"Timestamp directory: {TIMESTAMP_DIRECTORY}")

    if not os.path.isdir(AUDIO_DIRECTORY):
        print(f"Error: Audio directory '{AUDIO_DIRECTORY}' not found.")
        print("Please create it and place your audio files inside.")
        return

    if not os.path.exists(TIMESTAMP_DIRECTORY):
        print(f"Timestamp directory '{TIMESTAMP_DIRECTORY}' not found. Creating it...")
        os.makedirs(TIMESTAMP_DIRECTORY)
    elif not os.path.isdir(TIMESTAMP_DIRECTORY):
        print(f"Error: A file named '{TIMESTAMP_DIRECTORY}' exists, but it's not a directory.")
        print("Please resolve this conflict.")
        return

    # Get a list of all files in the audio directory
    all_audio_files = [f for f in os.listdir(AUDIO_DIRECTORY) if f.endswith(AUDIO_FILE_EXTENSION)]

    if not all_audio_files:
        print(f"No '{AUDIO_FILE_EXTENSION}' files found in '{AUDIO_DIRECTORY}'.")
        return

    print(f"Found {len(all_audio_files)} audio files to check.")

    for audio_file_name in all_audio_files:
        # Extract the base name (e.g., "hiyis" from "hiyis.wav")
        base_name = os.path.splitext(audio_file_name)[0]

        # Construct the full path to the audio file
        full_audio_path = os.path.join(AUDIO_DIRECTORY, audio_file_name)

        # Construct the expected path for the corresponding CSV file
        expected_csv_file = f"{base_name}{TIMESTAMP_FILE_EXTENSION}"
        full_csv_path = os.path.join(TIMESTAMP_DIRECTORY, expected_csv_file)

        print(f"\nChecking: {audio_file_name}")
        print(f"  Expected CSV: {full_csv_path}")

        # Check if the CSV file exists
        if not os.path.exists(full_csv_path):
            print(f"  CSV file NOT found for '{audio_file_name}'.")
            gettimestamp.maketimestamp(full_audio_path)
        else:
            print(f"  CSV file already exists for '{audio_file_name}'. Skipping.")

    print("\nTimestamp check complete.")

if __name__ == "__main__":
    run_timestamp_check()