from pydub import AudioSegment
import os
import librosa
import numpy as np

def frequency_to_note(frequency):
    """
    Convert a frequency to its corresponding musical note.
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if frequency <= 0:
        return "Unknown"
    A4 = 440.0
    semitones = 12 * np.log2(frequency / A4)
    note_index = int(round(semitones)) % 12
    return note_names[note_index]

def chop_audio_with_metadata(audio_file, segments, clusters=None):
    """
    Chop the audio file into segments and save them with metadata in structured directories.
    """
    audio = AudioSegment.from_wav(audio_file)
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_dir = f"{base_name}_segmented"
    os.makedirs(output_dir, exist_ok=True)

    for i, (start, end) in enumerate(segments):
        segment = audio[start * 1000:end * 1000]
        
        # Compute frequency and note metadata
        y, sr = librosa.load(audio_file, offset=start, duration=(end - start))
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        note = frequency_to_note(spectral_centroid)
        
        # Determine subfolder based on cluster
        if clusters is not None and len(clusters) > i:
            cluster_label = f"cluster_{clusters[i]}"
        else:
            cluster_label = "unclustered"

        cluster_folder = os.path.join(output_dir, cluster_label)
        os.makedirs(cluster_folder, exist_ok=True)

        # Save segment with metadata in the filename
        filename = f"seg{i+1}_freq{int(spectral_centroid)}_note{note}.wav"
        segment.export(os.path.join(cluster_folder, filename), format="wav")

def extract_features(segment_file):
    y, sr = librosa.load(segment_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)