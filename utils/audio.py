"""Audio preprocessing utilities."""

import numpy as np
import librosa
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


# Exact spec_transform from IBNet helpers.py
spec_transform = nn.Sequential(
    MelSpectrogram(n_fft=400, sample_rate=16000, hop_length=160, n_mels=64),
    AmplitudeToDB(stype="power", top_db=80)
)


def load_audio(audio_path, sr=16000, max_duration=30):
    """Load and preprocess audio file.

    Args:
        audio_path: Path to audio file
        sr: Sample rate (Hz)
        max_duration: Maximum audio duration in seconds

    Returns:
        Audio waveform as numpy array
    """
    y, _ = librosa.load(audio_path, sr=sr)

    # Trim to max duration
    max_samples = int(sr * max_duration)
    if len(y) > max_samples:
        y = y[:max_samples]

    return y


def normalize_audio(audio, target_db=-20.0):
    """Normalize audio to target loudness."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio / rms
    return audio
