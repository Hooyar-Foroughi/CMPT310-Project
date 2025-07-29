
import numpy as np
import sys
import os

# Add src to path 
sys.path.append('src')

from learning_env import SpeakerEnvironment

def test_environment():
    """Test the RL environment with a simple example"""
    
    print(" Testing RL Speaker Diarization Environment")

    
    # Initialize environment
    audio_file = "audio/jyirt.wav"
    if not os.path.exists(audio_file):
        print(" Audio file {audio_file} not found!")
        return
    
    print(f" Loading audio file: {audio_file}")
    env = SpeakerEnvironment(audio_file, segment_duration=1.0)
    
    print(f"Environment created with {env.n_segments} segments")
    print(f"  Audio duration: {env.duration:.2f} seconds")
    
    # Test reset
    print("\n Testing reset...")
    initial_state = env.reset()
    print(f" Initial state shape: {initial_state.shape}")
    print(f" Initial state values: {initial_state[:5]}...")  
    
    print("\n Testing environment steps...")
    for step in range(3):  # Test first 3 steps
        print(f"\n--- Step {step + 1} ---")
        
        # Show current state
        current_state = env._get_state()
        print(f" Current state shape: {current_state.shape}")
        
        # Take a random action (speaker assignment)
        action = np.random.randint(0, 5)  # Random speaker 0-4
        print(f"Agent chooses speaker: {action}")
        
        # Take step
        next_state, reward, done = env.step(action)
        
        print(f" Reward: {reward:.3f}")
        print(f" Done: {done}")
        print(f"Speaker assignments so far: {env.speaker_assignments}")
        
        if done:
            print("Episode finished!")
            break
    
   
    print("\n Final Results:")
    print(f" Total speaker assignments: {len(env.speaker_assignments)}")
    print(f" Unique speakers used: {len(set(env.speaker_assignments))}")
    print(f" Speaker distribution: {dict(zip(*np.unique(env.speaker_assignments, return_counts=True)))}")

def test_feature_extraction():
    """Test just the feature extraction"""
    print("\n Testing Feature Extraction")
    
    
    audio_file = "audio/jyirt.wav"
    if not os.path.exists(audio_file):
        print(f" Audio file {audio_file} not found!")
        return
    
    env = SpeakerEnvironment(audio_file, segment_duration=1.0)
    
    # Test feature extraction on first segment
    if len(env.segments) > 0:
        first_segment = env.segments[0]
        features = first_segment['features']
        
        print(f" Feature vector shape: {features.shape}")
        print(f"Feature values: {features}")
        
        # Show what each feature represents
        feature_names = [
            "MFCC Mean (13 values)",
            "MFCC Std (13 values)", 
            "Spectral Centroid",
            "Spectral Rolloff",
            "Spectral Bandwidth",
            "Pitch Mean",
            "Pitch Std"
        ]
        
        print("\n Feature breakdown:")
        start_idx = 0
        for i, name in enumerate(feature_names):
            if i < 2:  # MFCC features have 13 values each
                end_idx = start_idx + 13
                print(f"  {name}: {features[start_idx:end_idx][:3]}...")  # Show first 3
                start_idx = end_idx
            else:  # Single values
                print(f"  {name}: {features[start_idx]:.3f}")
                start_idx += 1

if __name__ == "__main__":
    test_environment()
    test_feature_extraction() 