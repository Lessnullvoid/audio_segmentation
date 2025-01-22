from pydub import AudioSegment
import librosa
import numpy as np

def chop_audio(audio_file, segments):
    audio = AudioSegment.from_wav(audio_file)
    for i, (start, end) in enumerate(segments):
        chunk = audio[start * 1000:end * 1000]
        chunk.export(f"segment_{i+1}.wav", format="wav")

def extract_features(segment_file):
    y, sr = librosa.load(segment_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)