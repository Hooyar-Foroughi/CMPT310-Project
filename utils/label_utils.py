
def count_unique_speakers(rttm_file):
    speaker_ids = set()
    with open(rttm_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                speaker_ids.add(parts[7])
    return len(speaker_ids)