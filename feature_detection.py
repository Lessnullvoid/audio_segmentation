import librosa
import numpy as np
import matplotlib.pyplot as plt

def detect_transients(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    transients = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    return transients

def detect_beats(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return tempo, beat_times

def detect_spectral_features(y, sr):
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    times = librosa.times_like(spectral_centroid[0], sr=sr)
    return {
        "spectral_centroid": (times, spectral_centroid[0]),
        "spectral_rolloff": (times, spectral_rolloff[0]),
        "spectral_bandwidth": (times, spectral_bandwidth[0]),
    }

def detect_features(audio_file):
    """Detect various audio features."""
    y, sr = librosa.load(audio_file)
    
    # Store audio file path in features
    features = {
        "audio_file": audio_file
    }
    
    # Detect beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    features["beats"] = librosa.frames_to_time(beats, sr=sr)
    features["tempo"] = tempo
    
    # Detect transients
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    transients = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    features["transients"] = librosa.frames_to_time(transients, sr=sr)
    
    # Spectral features
    times = librosa.times_like(onset_env, sr=sr)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid"] = (times, spectral_centroids)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["spectral_rolloff"] = (times, spectral_rolloff)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth"] = (times, spectral_bandwidth)
    
    return features

def plot_features(y, sr):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(y) / sr, num=len(y)), y, alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()