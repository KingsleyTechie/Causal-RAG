"""
Causal-RAG: A framework for causally-augmented retrieval to reduce hallucinations in clinical AI systems.
"""

__version__ = "0.1.0"
__author__ = "Causal-RAG Contributors"
__email__ = "your-email@example.com"

from causal_rag.core.causal_analyzer import CausalAnalyzer
from causal_rag.core.retriever import CausalRetriever
from causal_rag.core.generator import CausalGenerator
from causal_rag.core.pipeline import CausalRAGPipeline

__all__ = [
    "CausalAnalyzer",
    "CausalRetriever", 
    "CausalGenerator",
    "CausalRAGPipeline"
]
