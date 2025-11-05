"""
Comprehensive Affect Detection and Modulation System

This advanced affect analysis system provides comprehensive emotion detection,
modulation, and tracking capabilities specifically designed for brain-inspired
AI systems. The module implements cutting-edge affective computing techniques
for understanding and responding to emotional states in human-AI interactions.

Affect Detection Architecture:
    The affect system follows a multi-layered approach combining psychological
    models with machine learning techniques:

    ┌─────────────────────────────────────────────────────────────┐
    │                    Affect System                           │
    └─────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Emotion   │ │   Affect    │ │    Affect   │
    │  Detector   │ │   Modulation│ │   Logger    │
    │             │ │   Hooks     │ │             │
    │ • Multi-modal│ │ • Context   │ │ • State     │
    │   Analysis  │ │   Adaptation│ │   Tracking  │
    │ • VAC Model │ │   Modulation│ │ • Analytics │
    │             │ │             │ │ • Reporting │
    └─────────────┘ └─────────────┘ └─────────────┘

Core Components:

EmotionDetector (Multi-Modal Affect Analysis):
    - Advanced emotion detection from text, voice, and contextual signals
    - Valence-Arousal-Certainty (VAC) model implementation
    - Multi-language emotion recognition capabilities
    - Real-time emotion state tracking and classification
    - Uncertainty quantification and confidence scoring
    - Alternative emotion hypothesis generation
    - Contextual adaptation for different domains and cultures

AffectModulationHooks (Adaptive Response System):
    - Context-aware affect modulation based on detected emotional states
    - Dynamic response adaptation for improved user experience
    - Emotion-driven content filtering and adaptation
    - Real-time affect state monitoring and intervention
    - Integration hooks for AI model affect modulation
    - Safety mechanisms for emotional content handling

AffectLogger (Comprehensive Tracking System):
    - Detailed affect state logging and analysis
    - Historical emotion pattern recognition and trends
    - Performance metrics for affect detection accuracy
    - Real-time affect dashboard and visualization
    - Compliance reporting for emotional AI systems
    - Research data export and analysis capabilities

AffectTypes (Data Structures and Models):
    - Comprehensive affect state definitions and taxonomies
    - Valence-Arousal-Certainty signal representations
    - Affect intensity and duration modeling
    - Multi-modal affect context and metadata structures.

Key Features:
    - Multi-modal emotion detection (text, voice, facial expressions)
    - Real-time affect state tracking and classification
    - Cross-cultural emotion recognition and adaptation
    - Uncertainty quantification with confidence intervals
    - Alternative emotion hypothesis generation
    - Context-aware affect modulation and response adaptation
    - Comprehensive logging and analytics system
    - Integration hooks for mini-biai-1 architectures

Psychological Models:
    - Valence-Arousal-Certainty (VAC) model for affect representation
    - Plutchik's emotion wheel for emotion taxonomy
    - Circumplex model of affect for emotion positioning
    - Cognitive appraisal theory for emotion generation
    - Multi-dimensional affect scaling and measurement.

Version: 1.0.0
Author: mini-biai-1 Team
License: MIT
"""

from .emotion_detector import EmotionDetector
from .modulation_hooks import AffectModulationHooks
from .affect_types import AffectState, ValenceArousalCertainty
from .affect_logger import AffectLogger

# Enhanced Integration Components
from .enhanced_affect_integration import (
    EnhancedAffectIntegration,
    RouterTemperatureSettings,
    MemoryThresholdSettings, 
    RewardShapingConfig,
    ResponseAdaptationSettings,
    create_enhanced_affect_system
)

# A/B Testing Framework
from .ab_testing_framework import (
    AffectABTestingFramework,
    TestHypothesis,
    TestVariant,
    TestMetric
)

# Comprehensive Demo
from .comprehensive_enhanced_demo import main as demo_enhanced_affect_integration

__all__ = [
    # Core Components
    'EmotionDetector',
    'AffectModulationHooks',
    'AffectState',
    'ValenceArousalCertainty',
    'AffectLogger',
    
    # Enhanced Integration
    'EnhancedAffectIntegration',
    'RouterTemperatureSettings',
    'MemoryThresholdSettings',
    'RewardShapingConfig',
    'ResponseAdaptationSettings',
    'create_enhanced_affect_system',
    
    # A/B Testing Framework
    'AffectABTestingFramework',
    'TestHypothesis',
    'TestVariant', 
    'TestMetric',
    
    # Demo
    'demo_enhanced_affect_integration'
]

__version__ = "1.0.0"
__author__ = "mini-biai-1 Team"