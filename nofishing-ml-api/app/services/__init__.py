# -*- coding: utf-8 -*-
"""
NoFishing ML API - Services
"""
from app.services.hybrid_classifier import (
    HybridPhishingClassifier,
    LocalURLClassifier,
    RemoteLLMAnalyzer,
    get_hybrid_classifier
)

__all__ = [
    'HybridPhishingClassifier',
    'LocalURLClassifier',
    'RemoteLLMAnalyzer',
    'get_hybrid_classifier'
]
