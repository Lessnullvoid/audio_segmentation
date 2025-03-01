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
    Chop the audio file into segments and save them with metadata.
    If clusters is provided, organize in cluster folders, otherwise save in a single folder.
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
        
        # Determine save location based on clustering
        if clusters is not None and len(clusters) > i:
            # Save in cluster subfolder
            cluster_folder = os.path.join(output_dir, f"cluster_{clusters[i]}")
            os.makedirs(cluster_folder, exist_ok=True)
            save_path = cluster_folder
        else:
            # Save in main folder
            save_path = output_dir

        # Create filename with metadata
        filename = f"seg{i+1}_freq{int(spectral_centroid)}_note{note}.wav"
        segment.export(os.path.join(save_path, filename), format="wav")
        
        # Print progress
        if clusters is not None:
            print(f"Saved segment {i+1} in cluster {clusters[i]}")
        else:
            print(f"Saved segment {i+1}")

    total_segments = len(segments)
    print(f"\nSuccessfully saved {total_segments} segments")
    if clusters is not None:
        num_clusters = len(set(clusters))
        print(f"Organized into {num_clusters} clusters")
    print(f"Output directory: {output_dir}")

def extract_features(segment_file):
    y, sr = librosa.load(segment_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

def is_silent_segment(audio_file, start, end, threshold_db=-60):
    """
    Check if a segment is silent
    Parameters:
        audio_file: path to audio file
        start: start time in seconds
        end: end time in seconds
        threshold_db: silence threshold in dB (default: -60dB)
    Returns:
        bool: True if segment is silent
    """
    y, sr = librosa.load(audio_file)
    
    # Get segment samples
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = y[start_sample:end_sample]
    
    if len(segment) == 0:
        return True
        
    # Calculate RMS energy in dB
    rms = librosa.feature.rms(y=segment)
    db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Check if the average dB is below threshold
    return np.mean(db) < threshold_db