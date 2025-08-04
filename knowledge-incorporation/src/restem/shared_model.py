# knowledge-incorporation/src/restem/shared_model.py

"""
Shared Model Manager for ARC RestEM Pipeline

This module provides a singleton pattern for sharing MLX models across
multiple components to avoid loading multiple instances and causing OOM.
"""

import logging
from typing import Optional, Tuple, Any
import gc

LOG = logging.getLogger(__name__)


class SharedModelManager:
    """Singleton model manager to share MLX models across components."""
    
    _instance: Optional['SharedModelManager'] = None
    _model: Optional[Any] = None
    _tokenizer: Optional[Any] = None
    _model_name: Optional[str] = None
    
    def __new__(cls) -> 'SharedModelManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load or return cached model and tokenizer."""
        if self._model is None or self._model_name != model_name:
            LOG.info(f"Loading shared model: {model_name}")
            
            # Clean up previous model if exists
            if self._model is not None:
                del self._model, self._tokenizer
                gc.collect()
            
            # Load new model
            from mlx_lm import load
            self._model, self._tokenizer = load(model_name)
            self._model_name = model_name
            
            LOG.info(f"Shared model loaded: {model_name}")
        else:
            LOG.info(f"Reusing cached model: {model_name}")
            
        return self._model, self._tokenizer
    
    def get_model(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Get current model and tokenizer if loaded."""
        return self._model, self._tokenizer
    
    def clear_model(self):
        """Clear the cached model to free memory."""
        if self._model is not None:
            LOG.info("Clearing shared model from memory")
            del self._model, self._tokenizer
            self._model = None
            self._tokenizer = None
            self._model_name = None
            gc.collect()


# Global instance
shared_model_manager = SharedModelManager()


def get_shared_model(model_name: str) -> Tuple[Any, Any]:
    """Convenience function to get shared model."""
    return shared_model_manager.load_model(model_name)


def clear_shared_model():
    """Convenience function to clear shared model."""
    shared_model_manager.clear_model()
