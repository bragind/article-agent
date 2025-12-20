# src/__init__.py
"""
Пакет article-agent - AI RAG-агент для поиска статей
"""

from .core.rag import RAGAgent, SearchScope
from .generation.llm_client import LLMClient

__version__ = "1.0.0"
__all__ = ["RAGAgent", "SearchScope", "LLMClient"]