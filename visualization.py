import librosa.display
import matplotlib.pyplot as plt

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