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
    y, sr = librosa.load(audio_file)
    transients = detect_transients(y, sr)
    tempo, beat_times = detect_beats(y, sr)
    spectral_features = detect_spectral_features(y, sr)
    return {
        "transients": transients,
        "beats": beat_times,
        "tempo": tempo,
        **spectral_features
    }

def plot_features(y, sr):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(y) / sr, num=len(y)), y, alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()