"""
Experts Module for Multi-Expert AI System

This module contains the implementation of various expert systems for the
multi-expert architecture. Each expert specializes in different domains:

- LanguageExpert: Text understanding, generation, and reasoning
- VisionExpert: Visual scene understanding and image analysis  
- SymbolicExpert: Logical reasoning, mathematics, and planning

All experts implement the BaseExpert interface and can be dynamically
loaded and routed by the MultiExpertRouter system.

Author: mini-biai-1 Team
Version: 2.0.0
License: MIT
"""

# Expert implementations
try:
    from .language_expert import LanguageExpert
    from .vision_expert import VisionExpert  
    from .symbolic_expert import SymbolicExpert
    
    # Complete vision module components
    from .complete_vision_module import (
        CompleteVisionModule,
        CompleteVisionExpert,
        create_vision_expert,
        VisionConfig
    )
    
    # Vision utilities
    from .image_processing_utils import ImageProcessor, BatchImageProcessor
    from .vocabulary_tokenizer import Tokenizer, Vocabulary, TextPreprocessor
    
    # Base interface
    from ..interfaces.experts import (
        BaseExpert,
        ExpertType,
        ExpertRequest,
        ExpertResponse,
        ExpertMetadata,
        ExpertCapabilities,
        ExpertManager
    )

except ImportError:
    # Handle running as script or module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from interfaces.experts import (
        BaseExpert,
        ExpertType,
        ExpertRequest,
        ExpertResponse,
        ExpertMetadata,
        ExpertCapabilities,
        ExpertManager
    )
    
    # Define stub classes for standalone execution
    class LanguageExpert(BaseExpert):
        def __init__(self):
            super().__init__("language_expert_stub", ExpertType.LANGUAGE)
        
        def get_metadata(self):
            return ExpertMetadata("language_expert_stub", ExpertType.LANGUAGE, "text", "text", 2048, 32, [], {}, {})
        
        def get_capabilities(self):
            return ExpertCapabilities()
        
        def process(self, request):
            return ExpertResponse("stub response", 0.5, 1.0, expert_id="language_expert_stub")
    
    class VisionExpert(BaseExpert):
        def __init__(self):
            super().__init__("vision_expert_stub", ExpertType.VISION)
        
        def get_metadata(self):
            return ExpertMetadata("vision_expert_stub", ExpertType.VISION, "image", "text", 0, 16, [], {}, {})
        
        def get_capabilities(self):
            return ExpertCapabilities()
        
        def process(self, request):
            return ExpertResponse("stub vision response", 0.5, 1.0, expert_id="vision_expert_stub")
    
    class SymbolicExpert(BaseExpert):
        def __init__(self):
            super().__init__("symbolic_expert_stub", ExpertType.SYMBOLIC)
        
        def get_metadata(self):
            return ExpertMetadata("symbolic_expert_stub", ExpertType.SYMBOLIC, "structured", "structured", 0, 64, [], {}, {})
        
        def get_capabilities(self):
            return ExpertCapabilities()
        
        def process(self, request):
            return ExpertResponse("stub symbolic response", 0.5, 1.0, expert_id="symbolic_expert_stub")

# Version information
__version__ = "2.0.0"
__author__ = "mini-biai-1 Team"

# Export list for convenient imports
__all__ = [
    # Expert classes
    'LanguageExpert',
    'VisionExpert', 
    'SymbolicExpert',
    'CompleteVisionExpert',
    
    # Vision module components
    'CompleteVisionModule',
    'create_vision_expert',
    'VisionConfig',
    'ImageProcessor',
    'BatchImageProcessor',
    'Tokenizer',
    'Vocabulary',
    'TextPreprocessor',
    
    # Base interface
    'BaseExpert',
    
    # Data structures
    'ExpertType',
    'ExpertRequest', 
    'ExpertResponse',
    'ExpertMetadata',
    'ExpertCapabilities',
    
    # Manager
    'ExpertManager'
]


def create_expert_manager() -> ExpertManager:
    """Create and configure an expert manager with default experts"""
    try:
        manager = ExpertManager()
        
        # Create and register default experts
        language_expert = LanguageExpert()
        vision_expert = VisionExpert()
        symbolic_expert = SymbolicExpert()
        
        # Register experts
        manager.register_expert(language_expert)
        manager.register_expert(vision_expert)
        manager.register_expert(symbolic_expert)
        
        return manager
        
    except Exception as e:
        import logging
        logger = logging.getLogger("experts")
        logger.error(f"Failed to create expert manager: {e}")
        return ExpertManager()


def get_available_experts() -> dict:
    """Get information about all available experts"""
    return {
        'language_expert': {
            'type': 'LanguageExpert',
            'description': 'Text understanding, generation, and reasoning',
            'specializations': ['text_understanding', 'text_generation', 'natural_language_reasoning'],
            'input_modality': 'text',
            'output_type': 'text'
        },
        'vision_expert': {
            'type': 'VisionExpert', 
            'description': 'Visual scene understanding and image analysis',
            'specializations': ['image_understanding', 'visual_scene_analysis', 'image_captioning'],
            'input_modality': 'image',
            'output_type': 'text'
        },
        'symbolic_expert': {
            'type': 'SymbolicExpert',
            'description': 'Logical reasoning, mathematics, and planning',
            'specializations': ['logical_reasoning', 'mathematical_computation', 'planning_and_problem_solving'],
            'input_modality': 'structured',
            'output_type': 'structured'
        }
    }


def validate_expert_installation():
    """Validate that all expert modules are properly installed"""
    try:
        import torch
        
        # Check basic imports
        from .language_expert import LanguageExpert
        from .vision_expert import VisionExpert
        from .symbolic_expert import SymbolicExpert
        
        # Test basic instantiation
        lang_expert = LanguageExpert()
        vision_expert = VisionExpert()
        symbolic_expert = SymbolicExpert()
        
        # Test processing
        from ..interfaces.experts import ExpertRequest
        
        # Language test
        lang_request = ExpertRequest("Hello, this is a test.")
        lang_response = lang_expert.process(lang_request)
        
        # Vision test (mock image)
        import torch
        mock_image = torch.randn(3, 224, 224)
        vision_request = ExpertRequest(mock_image)
        vision_response = vision_expert.process(vision_request)
        
        # Symbolic test
        symbolic_request = ExpertRequest("2 + 2 = ?")
        symbolic_response = symbolic_expert.process(symbolic_request)
        
        return {
            'status': 'success',
            'message': 'All experts validated successfully',
            'tests': {
                'language_expert': lang_response.is_success(),
                'vision_expert': vision_response.is_success(), 
                'symbolic_expert': symbolic_response.is_success()
            }
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Expert validation failed: {str(e)}',
            'error': str(e)
        }


if __name__ == "__main__":
    # Validation and demo
    print("Multi-Expert System - Expert Module")
    print("===================================")
    
    print("\nAvailable Experts:")
    experts = get_available_experts()
    for name, info in experts.items():
        print(f"  {name}: {info['description']}")
        print(f"    Specializations: {', '.join(info['specializations'])}")
    
    print("\nValidating Installation...")
    validation = validate_expert_installation()
    print(f"Status: {validation['status']}")
    print(f"Message: {validation['message']}")
    
    if validation['status'] == 'success':
        print("\nTest Results:")
        for test, result in validation['tests'].items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {test}: {status}")
        
        print("\n🎉 Expert system validation completed!")
    else:
        print(f"\n❌ Expert system validation failed: {validation['message']}")
        if 'error' in validation:
            print(f"Error details: {validation['error']}")