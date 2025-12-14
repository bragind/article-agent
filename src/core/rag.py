"""
Основной RAG класс для поиска и генерации ответов
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

from src.ingest.embedder import Embedder
from src.ingest.vector_store import VectorStore
from src.generation.llm_client import LLMClient

logger = logging.getLogger(__name__)


class RAGSystem:
    """Основная система RAG для поиска статей и генерации ответов"""
    
    def __init__(self, 
                 vector_db_path: str = "chroma_full_db",
                 llm_model: str = "llama3.2:3b",
                 llm_base_url: str = "http://localhost:11434"):
        """
        Инициализация RAG системы
        
        Args:
            vector_db_path: Путь к векторной базе данных
            llm_model: Название модели для LLM
            llm_base_url: URL сервера Ollama
        """
        self.embedder = Embedder()
        self.vector_store = VectorStore(path=vector_db_path)
        self.llm_client = LLMClient(base_url=llm_base_url, model=llm_model)
        
        if not self.llm_client.test_connection():
            logger.warning(f"Не удается подключиться к LLM серверу {llm_base_url}. "
                          f"Убедитесь, что Ollama запущен и модель '{llm_model}' загружена.")
    
    def search(self, query: str, top_k: int = 5) -> Dict[str, List]:
        """
        Поиск статей по запросу (базовый векторный поиск)
        
        Args:
            query: Текст запроса
            top_k: Количество возвращаемых результатов
            
        Returns:
            Результаты поиска
        """
        query_embedding = self.embedder.embed([query])[0]
        
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return results
    
    def generate_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Полный RAG-пайплайн: поиск + генерация ответа
        
        Args:
            query: Вопрос пользователя
            top_k: Количество релевантных чанков для использования
            
        Returns:
            Полный ответ с источниками и вопросами
        """
        logger.info(f"Обработка запроса: '{query}'")
        
        search_results = self.search(query, top_k=top_k)
        
        if not search_results or not search_results.get("documents"):
            return {
                "answer": f"По запросу '{query}' ничего не найдено.",
                "sources": [],
                "questions": [],
                "similar_articles": []
            }
        
        documents = search_results["documents"][0]
        metadatas = search_results["metadatas"][0]
        
        context_chunks = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            context_chunks.append({
                "text": doc,
                "metadata": meta,
                "relevance_score": 1.0 - (i * 0.1)
            })
        
        llm_result = self.llm_client.generate_answer_with_context(
            query=query,
            context=context_chunks,
            top_k=min(top_k, len(context_chunks))
        )
        
        questions = []
        if documents:
            top_article_text = documents[0]
            questions = self.llm_client.generate_questions(
                article_text=top_article_text,
                num_questions=3
            )
        
        similar_articles = self._get_similar_articles(search_results)
        
        return {
            "answer": llm_result["answer"],
            "sources": llm_result["sources"],
            "questions": questions,
            "similar_articles": similar_articles,
            "search_stats": {
                "total_found": len(documents),
                "context_used": llm_result["context_used"]
            }
        }
    
    def _get_similar_articles(self, search_results: Dict[str, List]) -> List[Dict[str, str]]:
        """
        Получение похожих статей на основе результатов поиска
        
        Args:
            search_results: Результаты поиска
            
        Returns:
            Список похожих статей
        """
        if not search_results or not search_results.get("metadatas"):
            return []
        
        metadatas = search_results["metadatas"][0]
        similar_articles = []
        
        for i, meta in enumerate(metadatas[:3]):
            similar_articles.append({
                "title": meta.get("title", "Без названия"),
                "url": meta.get("url", ""),
                "author": meta.get("author", "Неизвестен"),
                "source": meta.get("source", "Unknown"),
                "similarity_rank": i + 1
            })
        
        return similar_articles
    
    def get_filtered_results(self, 
                           search_results: Dict[str, List], 
                           author: Optional[str] = None,
                           source: Optional[str] = None,
                           min_date: Optional[str] = None,
                           max_date: Optional[str] = None) -> Dict[str, List]:
        """
        Фильтрация результатов поиска
        
        Args:
            search_results: Результаты поиска
            author: Фильтр по автору
            source: Фильтр по источнику
            min_date: Минимальная дата (формат YYYY-MM-DD)
            max_date: Максимальная дата (формат YYYY-MM-DD)
            
        Returns:
            Отфильтрованные результаты
        """
        if not search_results or not search_results.get("documents"):
            return search_results
        
        filtered_docs = []
        filtered_metas = []
        
        for doc, meta in zip(search_results["documents"][0], search_results["metadatas"][0]):
            include = True
            
            if author and author.lower() not in meta.get("author", "").lower():
                include = False
            
            if source and source.lower() not in meta.get("source", "").lower():
                include = False
            
            date_str = meta.get("date", "")
            if date_str:
                try:
                    if min_date and date_str < min_date:
                        include = False
                    if max_date and date_str > max_date:
                        include = False
                except:
                    pass
            
            if include:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
        
        return {
            "documents": [filtered_docs],
            "metadatas": [filtered_metas]
        }
    
    def get_available_sources(self) -> List[str]:
        """
        Получение уникальных источников в базе
        
        Returns:
            Список уникальных источников
        """
        return ["Habr", "CNews", "Rusbase", "Uploaded PDF"]
    
    def check_llm_connection(self) -> Dict[str, Any]:
        """
        Проверка состояния подключения к LLM
        
        Returns:
            Словарь с информацией о подключении
        """
        is_connected = self.llm_client.test_connection()
        models = self.llm_client.get_available_models() if is_connected else []
        
        return {
            "connected": is_connected,
            "current_model": self.llm_client.model,
            "available_models": models,
            "llm_base_url": self.llm_client.base_url
        }