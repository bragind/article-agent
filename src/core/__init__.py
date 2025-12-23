"""
Пакет core - ядро RAG системы
"""

from .rag_orchestrator import RAGOrchestrator
from .search_engine import SearchEngine, SearchScope
from .recommender import Recommender
from .document_manager import DocumentManager

# Для обратной совместимости
RAGAgent = RAGOrchestrator

__version__ = "2.0.0"
__all__ = [
    "RAGOrchestrator", 
    "RAGAgent",
    "SearchEngine", 
    "SearchScope", 
    "Recommender", 
    "DocumentManager"
]