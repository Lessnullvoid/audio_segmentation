import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import librosa

def plot_features(audio_file, features):
    y, sr = librosa.load(audio_file)
    plt.figure(figsize=(15, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.title("Waveform with Features")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    for transient in features["transients"]:
        plt.axvline(transient, color='r', linestyle='--', label='Transient' if transient == features["transients"][0] else "")
    for beat in features["beats"]:
        plt.axvline(beat, color='g', linestyle=':', label='Beat' if beat == features["beats"][0] else "")

    times, spectral_centroid = features["spectral_centroid"]
    plt.plot(times, spectral_centroid / max(spectral_centroid) * max(y), color='b', label="Spectral Centroid")

    plt.legend(loc="upper right")
    plt.show()

def simplified_waveform_with_segments(audio_file, segments):
    """
    Plot a simplified waveform with segment markers.
    """
    # Load audio and downsample
    y, sr = librosa.load(audio_file, sr=None)
    downsample_rate = max(1, len(y) // 5000)  # Downsample for efficiency
    y_downsampled = y[::downsample_rate]
    times = np.linspace(0, len(y) / sr, len(y_downsampled))

    # Plot waveform
    plt.figure(figsize=(12, 2))  # Simplified compact view
    plt.plot(times, y_downsampled, color="orange", linewidth=0.8)
    plt.title("Simplified Waveform with Segments")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.ylim([-1, 1])  # Normalize amplitude for consistency

    # Add segment markers
    for i, (start, end) in enumerate(segments):
        plt.axvline(x=start, color="cyan", linestyle="--", linewidth=0.7)
        plt.text(start, 0.8, f"{i+1}", color="blue", fontsize=8, ha="center")

    plt.tight_layout()
    plt.show()