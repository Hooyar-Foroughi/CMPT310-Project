import os
import src.gettimestamp as gettimestamp



# --- Configuration ---
AUDIO_DIRECTORY = "data/wav"
AGTIMESTAMP_DIRECTORY = "data/agtimestamp"
SPTIMESTAMP_DIRECTORY = "data/sptimestamp"
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
    print(f"agTimestamp directory: {AGTIMESTAMP_DIRECTORY}")
    print(f"spTimestamp directory: {SPTIMESTAMP_DIRECTORY}")

    if not os.path.isdir(AUDIO_DIRECTORY):
        print(f"Error: Audio directory '{AUDIO_DIRECTORY}' not found.")
        print("Please create it and place your audio files inside.")
        return

    if not os.path.exists(AGTIMESTAMP_DIRECTORY):
        print(f"Timestamp directory '{AGTIMESTAMP_DIRECTORY}' not found. Creating it...")
        os.makedirs(AGTIMESTAMP_DIRECTORY)
    elif not os.path.isdir(AGTIMESTAMP_DIRECTORY):
        print(f"Error: A file named '{AGTIMESTAMP_DIRECTORY}' exists, but it's not a directory.")
        print("Please resolve this conflict.")
        return
    
    if not os.path.exists(SPTIMESTAMP_DIRECTORY):
        print(f"Timestamp directory '{SPTIMESTAMP_DIRECTORY}' not found. Creating it...")
        os.makedirs(SPTIMESTAMP_DIRECTORY)
    elif not os.path.isdir(SPTIMESTAMP_DIRECTORY):
        print(f"Error: A file named '{SPTIMESTAMP_DIRECTORY}' exists, but it's not a directory.")
        print("Please resolve this conflict.")
        return

    # Get a list of all files in the audio directory
    all_audio_files = [f for f in os.listdir(AUDIO_DIRECTORY) if f.endswith(AUDIO_FILE_EXTENSION)]

    if not all_audio_files:
        print(f"No '{AUDIO_FILE_EXTENSION}' files found in '{AUDIO_DIRECTORY}'.")
        return

    print(f"Found {len(all_audio_files)} audio files to check.")

    for audio_file_name in all_audio_files:
        for dir in [AGTIMESTAMP_DIRECTORY,SPTIMESTAMP_DIRECTORY]:
            base_name = os.path.splitext(audio_file_name)[0]
            full_audio_path = os.path.join(AUDIO_DIRECTORY, audio_file_name)

            expected_csv_file = f"{base_name}{TIMESTAMP_FILE_EXTENSION}"
            full_csv_path = os.path.join(dir, expected_csv_file)
            print(f"\nChecking: {audio_file_name}")
            print(f"  Expected CSV: {full_csv_path}")


            if not os.path.exists(full_csv_path):
                print(f"  CSV file NOT found for '{audio_file_name}'.")
                if dir==AGTIMESTAMP_DIRECTORY:
                    gettimestamp.maketimestamp(full_audio_path)
                else:
                    gettimestamp.maketimestamp(full_audio_path,'sp')
            else:
                print(f"  CSV file already exists for '{audio_file_name}'. Skipping.")


    print("\nTimestamp check complete.")

if __name__ == "__main__":
    run_timestamp_check()