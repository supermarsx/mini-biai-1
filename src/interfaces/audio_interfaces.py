"""
Audio processing interfaces for the brain-inspired AI system.

This module defines interfaces for audio processing capabilities including
speech recognition, text-to-speech, audio classification, and music analysis.

Interfaces follow the established pattern using dataclasses with comprehensive
type hints and validation for all audio-related operations.

Key Components:
    - AudioData: Core audio data structure
    - SpeechRecognition: ASR interface with transcript and confidence
    - TextToSpeech: TTS interface with audio output
    - AudioClassification: Audio classification results
    - MusicAnalysis: Music-specific analysis results
    - AudioEnhancement: Audio preprocessing interface

Architecture Benefits:
    - Type-safe audio operations
    - Async operation support
    - Hardware-accelerated processing
    - Multi-format audio support
    - Real-time processing capabilities

Version: 1.0.0
Author: mini-biai-1 Team
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, BinaryIO
from enum import Enum
import numpy as np
import torch


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    AAC = "aac"
    OGG = "ogg"
    M4A = "m4a"


class AudioSampleRate(Enum):
    """Standard audio sample rates."""
    RATE_8KHZ = 8000
    RATE_16KHZ = 16000
    RATE_22KHZ = 22050
    RATE_44KHZ = 44100
    RATE_48KHZ = 48000


@dataclass
class AudioData:
    """
    Core audio data structure for all audio operations.
    
    Attributes:
        waveform: Audio waveform as numpy array
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels
        format: Audio format
        duration: Duration in seconds
        metadata: Additional audio metadata
    """
    waveform: np.ndarray
    sample_rate: int
    channels: int
    format: AudioFormat
    duration: float
    metadata: Dict[str, Any]


@dataclass
class SpeechRecognition:
    """
    Speech recognition result interface.
    
    Attributes:
        transcript: Recognized text
        confidence: Confidence score (0.0-1.0)
        language: Detected language code
        timestamps: Word-level timestamps
        alternatives: Alternative transcription hypotheses
        audio_features: Extracted audio features
    """
    transcript: str
    confidence: float
    language: str
    timestamps: Optional[List[Dict[str, float]]] = None
    alternatives: Optional[List[Dict[str, Any]]] = None
    audio_features: Optional[np.ndarray] = None


@dataclass
class TextToSpeech:
    """
    Text-to-speech synthesis result interface.
    
    Attributes:
        audio_data: Generated audio waveform
        voice_id: Voice identifier used
        speaking_rate: Speech rate factor
        pitch_shift: Pitch adjustment
        duration: Synthesis duration
        phonemes: Phoneme sequence
    """
    audio_data: np.ndarray
    voice_id: str
    speaking_rate: float
    pitch_shift: float
    duration: float
    phonemes: Optional[List[str]] = None


@dataclass
class AudioClassification:
    """
    Audio classification result interface.
    
    Attributes:
        labels: Predicted class labels
        probabilities: Class probability scores
        features: Extracted audio features
        spectrogram: Audio spectrogram representation
        mfcc: Mel-frequency cepstral coefficients
    """
    labels: List[str]
    probabilities: List[float]
    features: np.ndarray
    spectrogram: Optional[np.ndarray] = None
    mfcc: Optional[np.ndarray] = None


@dataclass
class MusicAnalysis:
    """
    Music analysis result interface.
    
    Attributes:
        genre: Predicted music genre
        tempo: Beats per minute (BPM)
        key: Musical key
        time_signature: Time signature (e.g., 4/4)
        energy: Energy level (0.0-1.0)
        valence: Musical valence (0.0-1.0)
        danceability: Danceability score (0.0-1.0)
        instruments: Detected instruments
        sections: Musical sections (verse, chorus, etc.)
    """
    genre: str
    tempo: float
    key: str
    time_signature: str
    energy: float
    valence: float
    danceability: float
    instruments: List[str]
    sections: Optional[List[Dict[str, Any]]] = None


@dataclass
class AudioEnhancement:
    """
    Audio enhancement and preprocessing interface.
    
    Attributes:
        enhanced_audio: Processed audio data
        noise_reduction: Noise reduction parameters
        normalization: Audio normalization settings
        filtering: Applied filters (high-pass, low-pass, etc.)
        loudness: Loudness normalization
        dynamic_range: Dynamic range compression
    """
    enhanced_audio: AudioData
    noise_reduction: Dict[str, Any]
    normalization: Dict[str, float]
    filtering: List[Dict[str, str]]
    loudness: float
    dynamic_range: Dict[str, float]


@dataclass
class AudioStream:
    """
    Real-time audio streaming interface.
    
    Attributes:
        stream_id: Unique stream identifier
        chunk_size: Audio chunk size in samples
        buffer_size: Buffer size for streaming
        is_active: Stream active status
        latency: Processing latency
        quality_metrics: Audio quality indicators
    """
    stream_id: str
    chunk_size: int
    buffer_size: int
    is_active: bool
    latency: float
    quality_metrics: Dict[str, float]


@dataclass
class AudioConfig:
    """
    Audio processing configuration interface.
    
    Attributes:
        model_name: Audio model identifier
        device: Processing device (cpu, cuda, etc.)
        batch_size: Processing batch size
        timeout: Processing timeout
        quality: Audio quality settings
        enhancement: Audio enhancement options
        feature_extraction: Feature extraction parameters
    """
    model_name: str
    device: str
    batch_size: int
    timeout: float
    quality: Dict[str, Any]
    enhancement: Dict[str, Any]
    feature_extraction: Dict[str, Any]


# Export all interfaces
__all__ = [
    'AudioFormat',
    'AudioSampleRate',
    'AudioData',
    'SpeechRecognition',
    'TextToSpeech',
    'AudioClassification',
    'MusicAnalysis',
    'AudioEnhancement',
    'AudioStream',
    'AudioConfig'
]