"""
Lazy loader for the DistilBERT prompt-injection classifier.

The model is hosted at HuggingFace Hub: agentguard/prompt-injection-detector
It is only downloaded on first use when use_ml=True is passed to PromptShield.

Requirements: pip install agentguard[ml]
"""
import warnings
from typing import Any, Optional

_classifier: Optional[Any] = None
_HF_MODEL_ID = "agentguard/prompt-injection-detector"


def load_injection_classifier() -> Optional[Any]:
    """Return a cached HuggingFace text-classification pipeline, or None on failure."""
    global _classifier

    if _classifier is not None:
        return _classifier

    try:
        from transformers import pipeline  # type: ignore
    except ImportError:
        warnings.warn(
            "AgentGuard: transformers is not installed. "
            "ML injection detection disabled. "
            "Install with: pip install agentguard[ml]",
            stacklevel=3,
        )
        return None

    try:
        _classifier = pipeline(
            "text-classification",
            model=_HF_MODEL_ID,
            device=-1,  # CPU; set device=0 for GPU
        )
        return _classifier
    except Exception as exc:
        warnings.warn(
            f"AgentGuard: Could not load ML classifier from '{_HF_MODEL_ID}': {exc}. "
            "The model may not be published yet — see training/train_injection_classifier.py. "
            "Falling back to rule-based detection only.",
            stacklevel=3,
        )
        return None
