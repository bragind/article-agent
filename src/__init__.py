# src/__init__.py
"""
Пакет article-agent - AI RAG-агент для поиска статей
"""

from .core.rag_orchestrator import RAGOrchestrator as RAGAgent, SearchScope
from .generation.llm_client import LLMClient

__version__ = "1.0.0"
__all__ = ["RAGAgent", "SearchScope", "LLMClient"]