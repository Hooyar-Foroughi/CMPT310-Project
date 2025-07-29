"""
Reinforcement Learning Environment for Speaker Diarization
Each segment becomes a "state" for the RL agent
"""

import numpy as np
import librosa
import os
from utils.audio_utils import extract_segments, embed_audio
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class SpeakerEnvironment:  
    def __init__(self, audio_path, segment_duration=1.0, sample_rate=16000):
        self.audio_path = audio_path
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate

        # Load and preprocess audio
        self.y, self.sr = librosa.load(self.audio_path, sr=self.sample_rate)
        self.duration = len(self.y) / self.sr

        # Extract features for each segment
        self.segments = self._extract_segments()
        self.n_segments = len(self.segments)
        
        # Initialize environment state variables
        self.current_step = 0
        self.speaker_assignments = []
        self.speaker_embeddings = []
        
        print(f"Environment initialized with {self.n_segments} segments")


    def _extract_segments(self):
        """Extract audio segments with features"""
        segments = []
        segment_length = int(self.segment_duration * self.sr)
        
        for i in range(0, len(self.y) - segment_length, segment_length // 2):  # 50% overlap
            segment_audio = self.y[i:i + segment_length]
            
            # Extract features for this segment
            features = self._extract_features(segment_audio)
            
            segments.append({
                'audio': segment_audio,
                'features': features,
                'start_time': i / self.sr,
                'end_time': (i + segment_length) / self.sr
            })
        
        return segments

    def _extract_features(self, segment):
        """Extract features from a segment"""
        mfcc = librosa.feature.mfcc(y=segment, sr=self.sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=self.sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=self.sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=self.sr)
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=segment, sr=self.sr)
        pitch_mean = np.mean(pitches[magnitudes > 0.1]) if np.any(magnitudes > 0.1) else 0
        pitch_std = np.std(pitches[magnitudes > 0.1]) if np.any(magnitudes > 0.1) else 0

        # Combine features - fix the concatenation
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        spec_centroid_mean = np.mean(spectral_centroid)
        spec_rolloff_mean = np.mean(spectral_rolloff)
        spec_bandwidth_mean = np.mean(spectral_bandwidth)
        
        # Convert all to numpy arrays for concatenation
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            np.array([spec_centroid_mean]),
            np.array([spec_rolloff_mean]),
            np.array([spec_bandwidth_mean]),
            np.array([pitch_mean]),
            np.array([pitch_std])
        ])

        return features
        
    def reset(self):
        # Reset the environment
        self.current_step = 0
        self.speaker_assignments = []
        self.speaker_embeddings = []
        return self._get_state()

    def _get_state(self):
        """Get current state of the environment"""
        if self.current_step >= self.n_segments:
            return None
        
        # Get current segment
        current_segment = self.segments[self.current_step]

        # Extract features - pass the audio data, not the segment dictionary
        feature = self._extract_features(current_segment['audio'])
        context = self._get_context()

        state = np.concatenate([feature, context])

        return state

    

    def _get_context(self):
        """Get context of the environment"""
        return np.array([self.current_step, self.n_segments])

    def step(self, action): 
        """Take action and return next state, reward, done"""
        if self.current_step >= self.n_segments:
            return None, 0, True
        
        # Assign speaker
        self.speaker_assignments.append(action)
        
        # Get features from current segment
        current_segment = self.segments[self.current_step]
        self.speaker_embeddings.append(current_segment['features'])
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.n_segments
        
        next_state = self._get_state()
        return next_state, reward, done
    
    def _calculate_reward(self, action):
        """Calculate reward for the action"""
        if len(self.speaker_assignments) < 2:
            return 0  # No reward for first assignment
        
        # Reward based on clustering quality
        if len(self.speaker_embeddings) >= 3:  # Need at least 3 samples for silhouette
            embeddings_array = np.array(self.speaker_embeddings)
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(embeddings_array)
            labels = kmeans.labels_
            
            try:
                silhouette = silhouette_score(embeddings_array, labels)
                reward = silhouette * 10  # Scale up for better gradients
            except ValueError:
                # If silhouette score fails, use a simple reward
                reward = 0.1  # Small positive reward for continuing
                
            # Penalty for too many speakers
            n_speakers = len(set(labels))
            if n_speakers > 8:  # Penalty for too many speakers
                reward -= (n_speakers - 8) * 0.5
                
            # Bonus for consistent assignments
            if len(self.speaker_assignments) >= 3:
                recent_assignments = self.speaker_assignments[-3:]
                if len(set(recent_assignments)) == 1:  # Same speaker for 3 consecutive segments
                    reward += 0.5
                    
            return reward
        else:
            # For 2 samples, give a small reward to encourage continuation
            return 0.05
        
    def get_results(self):
        """Get results of the environment"""
        timestamps = []
        for i, (assignment, segment) in enumerate(zip(self.speaker_assignments, self.segments)):
            timestamps.append((assignment, segment['start_time'], segment['end_time']))
        return self.speaker_assignments
    
    
    
    