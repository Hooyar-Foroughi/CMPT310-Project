"""
Optimized Reinforcement Learning Environment for Speaker Diarization
Uses Resemblyzer embeddings for fast and effective feature extraction
"""

import numpy as np
import librosa
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
from resemblyzer import preprocess_wav, VoiceEncoder
from scipy.stats import skew, kurtosis
from sklearn.metrics.pairwise import cosine_distances

# Global encoder (load once, reuse)
encoder = VoiceEncoder("cpu")
EMB_DIM = 512

class SpeakerEnvironment:
    def __init__(self, audio_path, segment_duration=0.25, sample_rate=16000):
        self.audio_path = audio_path
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate

        
        # Load audio once
        self.y, self.sr = librosa.load(self.audio_path, sr=self.sample_rate)
        self.duration = len(self.y) / self.sr

        # Pre-compute ALL features at initialization using Resemblyzer
        self.segments = self._precompute_all_features()
        self.n_segments = len(self.segments)
        
        # Initialize environment state variables
        self.current_step = 0
        self.speaker_assignments = []
        self.speaker_embeddings = []
        
     
        
    def _precompute_all_features(self):
        """Pre-compute features for ALL segments using Resemblyzer"""
        segments = []
        segment_length = int(self.segment_duration * self.sr)
        
        # Process all segments in batch
        segment_audios = []
        segment_times = []
        
        for i in range(0, len(self.y) - segment_length, segment_length // 2):  # 50% overlap
            segment_audio = self.y[i:i + segment_length]
            segment_audios.append(segment_audio)
            segment_times.append((i / self.sr, (i + segment_length) / self.sr))
        
        # Batch feature extraction using Resemblyzer
       
        features = self._extract_features_batch_resemblyzer(segment_audios)
        
        # Create segments with pre-computed features
        for i, (feature, (start_time, end_time)) in enumerate(zip(features, segment_times)):
            segments.append({
                'features': feature,
                'start_time': start_time,
                'end_time': end_time,
                'index': i
            })
        
        return segments

    def _extract_features_batch_resemblyzer(self, segment_audios):
        """Extract features for multiple segments using Resemblyzer"""
        features = []
        
        for segment_audio in segment_audios:
            # Use Resemblyzer for fast and effective feature extraction
            feature = self._extract_features_resemblyzer(segment_audio)
            features.append(feature)
        
        return features

    def _extract_features_resemblyzer(self, segment):
        """Extract features using Resemblyzer (simplified version)"""
        # Convert to proper format for Resemblyzer
        if len(segment) == 0:
            # Handle empty segment
            return np.zeros(128, dtype=np.float32)  # Reduced feature size
        
        # Get embeddings from Resemblyzer
        try:
            # Get frame embeddings
            _, embeds, _ = encoder.embed_utterance(
                segment, return_partials=True, rate=8
            )
            
            if len(embeds) == 0:
                # No embeddings found, return zeros
                return np.zeros(128, dtype=np.float32)
            
            # Convert to simplified pooled vector (128 features)
            pooled_feature = self._pooled_vector_simple(embeds)
            return pooled_feature
            
        except Exception as e:
            print(f"Warning: Resemblyzer failed for segment, using fallback: {e}")
            # Fallback to simple features
            return self._extract_features_fallback(segment)

    def _pooled_vector(self, embeds):
        """
        Produce a fixed‑length feature vector per clip (6147 features)
        Based on the unsupervised learning pipeline
        """
        # Core per‑dimension stats
        mu   = embeds.mean(axis=0)
        sd   = embeds.std(axis=0)
        med  = np.median(embeds, axis=0)
        mn   = embeds.min(axis=0)
        mx   = embeds.max(axis=0)
        
        # Handle potential NaN values in skew and kurtosis
        try:
            sk   = skew(embeds, axis=0, bias=False)
            ku   = kurtosis(embeds, axis=0, bias=False)
        except:
            sk = np.zeros(embeds.shape[1])
            ku = np.zeros(embeds.shape[1])

        # Percentiles (5 × 512)
        p10, p25, p50, p75, p90 = np.percentile(
            embeds, [10, 25, 50, 75, 90], axis=0
        )

        # First‑order Δ embeddings
        if len(embeds) > 1:
            delta = np.abs(np.diff(embeds, axis=0))
            d_mu  = delta.mean(axis=0)
            d_sd  = delta.std(axis=0)
        else:  # single frame
            d_mu = np.zeros(embeds.shape[1], dtype=np.float32)
            d_sd = np.zeros_like(d_mu)

        # Pairwise diversity
        pd_mean, pd_std = self._frame_distance_stats(embeds)

        dur   = np.array([len(embeds)], dtype=np.float32)

        vec = np.hstack(
            [
                mu, sd, med, mn, mx, sk, ku,
                p10, p25, p50, p75, p90,
                d_mu, d_sd,
                [pd_mean, pd_std],
                dur,
            ]
        ).astype(np.float32)

        # Replace any NaN values with zeros
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        
        return vec

    def _pooled_vector_simple(self, embeds):
        """
        Produce a simplified feature vector (128 features)
        Much more stable for neural networks
        """
        # Basic statistics (mean, std, min, max)
        mu = embeds.mean(axis=0)
        sd = embeds.std(axis=0)
        mn = embeds.min(axis=0)
        mx = embeds.max(axis=0)
        
        # Take first 32 dimensions to reduce size
        mu = mu[:32]
        sd = sd[:32]
        mn = mn[:32]
        mx = mx[:32]
        
        # Simple diversity measure
        if len(embeds) > 1:
            diversity = np.std(embeds[:32], axis=0)
        else:
            diversity = np.zeros(32)
        
        # Duration and basic stats
        duration = np.array([len(embeds)], dtype=np.float32)
        
        vec = np.concatenate([
            mu, sd, mn, mx, diversity, duration
        ]).astype(np.float32)
        
        # Ensure exact size
        if len(vec) < 128:
            vec = np.pad(vec, (0, 128 - len(vec)), 'constant')
        elif len(vec) > 128:
            vec = vec[:128]
        
        # Replace any NaN values with zeros
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        
        return vec

    def _frame_distance_stats(self, embeds):
        """Mean and standard deviation of pairwise cosine distances"""
        if len(embeds) < 2:
            return 0.0, 0.0
        dists = cosine_distances(embeds)
        iu = np.triu_indices_from(dists, k=1)
        flat = dists[iu]
        return float(flat.mean()), float(flat.std())

    def _extract_features_fallback(self, segment):
        """Fallback feature extraction if Resemblyzer fails"""
        # Simple features as fallback
        features = np.concatenate([
            np.array([np.mean(segment)]),  # Mean amplitude
            np.array([np.std(segment)]),   # Std amplitude
            np.array([len(segment)])       # Duration
        ])
        
        # Pad to match expected size (128 features)
        padded = np.zeros(128, dtype=np.float32)
        padded[:len(features)] = features
        return padded
        
    def reset(self):
        """Reset the environment"""
        self.current_step = 0
        self.speaker_assignments = []
        self.speaker_embeddings = []
        return self._get_state()

    def _get_state(self):
        """Get current state (uses pre-computed features)"""
        if self.current_step >= self.n_segments:
            return None
        
        # Get pre-computed features
        current_segment = self.segments[self.current_step]
        feature = current_segment['features']
        context = self._get_context()

        state = np.concatenate([feature, context])
        return state

    def _get_context(self):
        """Get context information"""
        return np.array([self.current_step, self.n_segments])

    def step(self, action): 
        """Take action and return next state, reward, done"""
        if self.current_step >= self.n_segments:
            return None, 0, True
        
        # Assign speaker
        self.speaker_assignments.append(action)
        
        # Get pre-computed features
        current_segment = self.segments[self.current_step]
        self.speaker_embeddings.append(current_segment['features'])
        
        # Calculate reward efficiently
        reward = self._calculate_reward_fast(action)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.n_segments
        
        next_state = self._get_state()
        return next_state, reward, done
    
    def _calculate_reward_fast(self, action):
        """Fast reward calculation"""
        if len(self.speaker_assignments) < 2:
            return 0.1  # Small positive reward for continuing
        
        # Simplified reward based on speaker consistency
        if len(self.speaker_assignments) >= 3:
            recent_assignments = self.speaker_assignments[-3:]
            if len(set(recent_assignments)) == 1:  # Same speaker
                return 1.0  # High reward for consistency
            else:
                return 0.1  # Low reward for changes
        
        return 0.1  # Default small reward
    
    def get_results(self):
        """Get results of the environment"""
        return self.speaker_assignments 