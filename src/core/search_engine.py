"""
Поисковая система с векторным и текстовым поиском
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SearchScope(Enum):
    """Области поиска"""
    ALL = "all"
    USER_ONLY = "user"
    HABR_ONLY = "habr"


class SearchEngine:
    """Поисковая система с поддержкой фильтров"""
    
    def __init__(self, vector_store, embedder, all_articles: List[Dict]):
        """
        Args:
            vector_store: Векторное хранилище
            embedder: Генератор эмбеддингов
            all_articles: Все статьи для текстового поиска (fallback)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.all_articles = all_articles
    
    def search(self, 
               query: str, 
               user_id: str, 
               scope: SearchScope = SearchScope.ALL, 
               limit: int = 10,
               tags: Optional[List[str]] = None,
               author: Optional[str] = None,
               date_from: Optional[str] = None,
               date_to: Optional[str] = None) -> List[Dict]:
        """
        Умный поиск с фильтрами
        
        Args:
            query: Запрос пользователя
            user_id: ID пользователя для фильтрации документов
            scope: Область поиска
            limit: Количество результатов
            tags: Фильтр по тегам
            author: Фильтр по автору
            date_from: Дата от
            date_to: Дата до
            
        Returns:
            Список найденных статей с релевантностью
        """
        logger.info(f"Поиск: '{query}', scope={scope}, user={user_id}")
        
        # Векторный поиск если доступен
        if self.vector_store and self.embedder:
            try:
                results = self._vector_search_with_filters(
                    query, user_id, scope, limit, 
                    tags, author, date_from, date_to
                )
                if results:
                    return results
            except Exception as e:
                logger.error(f"Ошибка векторного поиска: {e}")
        
        # Fallback на текстовый поиск
        return self._text_search_with_filters(
            query, user_id, scope, limit, 
            tags, author, date_from, date_to
        )
    
    def _vector_search_with_filters(self, 
                                   query: str, 
                                   user_id: str, 
                                   scope: SearchScope,
                                   limit: int,
                                   tags: Optional[List[str]] = None,
                                   author: Optional[str] = None,
                                   date_from: Optional[str] = None,
                                   date_to: Optional[str] = None) -> List[Dict]:
        """Векторный поиск с фильтрами"""
        if not self.embedder or not self.vector_store:
            return []
        
        try:
            # Генерация эмбеддинга запроса
            query_embedding = self.embedder.embed([query])
            if query_embedding is None or len(query_embedding) == 0:
                return []
            
            # Построение фильтра для ChromaDB
            where_filter = self._build_chroma_filter(user_id, scope)
            
            # Поиск с увеличенным лимитом для последующей фильтрации
            search_results = self.vector_store.search_with_filters(
                query_embedding[0], 
                top_k=limit * 3,
                tags=tags,
                author=author,
                date_from=date_from,
                date_to=date_to,
                where=where_filter
            )
            
            if not search_results.get("documents"):
                return []
            
            return self._format_vector_results(search_results, limit)
            
        except Exception as e:
            logger.error(f"Ошибка векторного поиска с фильтрами: {e}")
            return []
    
    def _build_chroma_filter(self, user_id: str, scope: SearchScope) -> Optional[Dict]:
        """Построение фильтра для ChromaDB"""
        if scope == SearchScope.HABR_ONLY:
            return {"source_type": {"$eq": "habr"}}
        
        elif scope == SearchScope.USER_ONLY:
            return {
                "$and": [
                    {"source_type": {"$eq": "user"}},
                    {"uploaded_by": {"$eq": user_id}}
                ]
            }
        
        elif scope == SearchScope.ALL:
            return {
                "$or": [
                    {"source_type": {"$eq": "habr"}},
                    {
                        "$and": [
                            {"source_type": {"$eq": "user"}},
                            {"uploaded_by": {"$eq": user_id}}
                        ]
                    }
                ]
            }
        
        return None
    
    def _format_vector_results(self, search_results: Dict, limit: int) -> List[Dict]:
        """Форматирование результатов векторного поиска"""
        formatted = []
        
        docs = search_results["documents"][0] if search_results.get("documents") else []
        metas = search_results["metadatas"][0] if search_results.get("metadatas") else []
        distances = search_results.get("distances", [[]])[0] if search_results.get("distances") else []
        
        seen_articles = set()
        
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            article_id = meta.get("article_id")
            if not article_id or article_id in seen_articles:
                continue
            
            # Вычисляем релевантность из расстояния
            score = 1.0
            if distances and i < len(distances):
                score = max(0.0, 1.0 - distances[i])
            
            formatted.append({
                "article": {
                    "id": article_id,
                    "title": meta.get("title", "Без названия"),
                    "text": doc,
                    "author": meta.get("author", "Неизвестен"),
                    "url": meta.get("url", ""),
                    "source": meta.get("source", "Unknown"),
                    "date": meta.get("date", ""),
                    "tags": self._parse_tags(meta.get("tags", "")),
                    "text_length": len(doc),
                    "is_user_document": meta.get("is_user_document", False),
                    "uploaded_by": meta.get("uploaded_by", "")
                },
                "score": score,
                "is_user_document": meta.get("is_user_document", False),
                "search_type": "vector"
            })
            
            seen_articles.add(article_id)
            if len(formatted) >= limit:
                break
        
        return sorted(formatted, key=lambda x: x["score"], reverse=True)
    
    def _text_search_with_filters(self, 
                                 query: str, 
                                 user_id: str, 
                                 scope: SearchScope,
                                 limit: int,
                                 tags: Optional[List[str]] = None,
                                 author: Optional[str] = None,
                                 date_from: Optional[str] = None,
                                 date_to: Optional[str] = None) -> List[Dict]:
        """Текстовый поиск с фильтрами (fallback)"""
        query_lower = query.lower()
        results = []
        
        # Фильтрация статей по области поиска
        articles_to_search = self._filter_articles_by_scope(user_id, scope)
        
        for article in articles_to_search:
            # Применение фильтров
            if not self._passes_filters(article, tags, author, date_from, date_to):
                continue
            
            # Вычисление релевантности
            score = self._calculate_text_relevance(article, query_lower)
            
            if score > 0:
                results.append({
                    "article": article,
                    "score": score,
                    "is_user_document": article.get("is_user_document", False),
                    "search_type": "text"
                })
        
        # Сортировка и ограничение результатов
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def _filter_articles_by_scope(self, user_id: str, scope: SearchScope) -> List[Dict]:
        """Фильтрация статей по области поиска"""
        if scope == SearchScope.HABR_ONLY:
            return [a for a in self.all_articles if not a.get("is_user_document")]
        
        elif scope == SearchScope.USER_ONLY:
            return [a for a in self.all_articles 
                   if a.get("is_user_document") and a.get("uploaded_by") == user_id]
        
        elif scope == SearchScope.ALL:
            return [a for a in self.all_articles 
                   if not a.get("is_user_document") or 
                   (a.get("is_user_document") and a.get("uploaded_by") == user_id)]
        
        return []
    
    def _passes_filters(self, article: Dict, 
                       tags: Optional[List[str]], 
                       author: Optional[str],
                       date_from: Optional[str],
                       date_to: Optional[str]) -> bool:
        """Проверка статьи на соответствие фильтрам"""
        # Фильтр по тегам
        if tags:
            article_tags = self._normalize_tags(article.get("tags", []))
            if not any(tag in article_tags for tag in tags):
                return False
        
        # Фильтр по автору
        if author:
            article_author = str(article.get("author", "")).lower()
            if author.lower() not in article_author:
                return False
        
        # Фильтр по дате
        article_date = article.get("date", "")
        if article_date and (date_from or date_to):
            try:
                date_obj = datetime.fromisoformat(article_date[:10] + "T00:00:00")
                
                if date_from:
                    filter_from = datetime.fromisoformat(date_from[:10] + "T00:00:00")
                    if date_obj < filter_from:
                        return False
                
                if date_to:
                    filter_to = datetime.fromisoformat(date_to[:10] + "T00:00:00")
                    if date_obj > filter_to:
                        return False
            except:
                pass
        
        return True
    
    def _calculate_text_relevance(self, article: Dict, query_lower: str) -> float:
        """Вычисление релевантности для текстового поиска"""
        score = 0.0
        
        # Поиск в заголовке (самый весомый)
        title = article.get("title", "").lower()
        if query_lower in title:
            score += 10.0
        
        # Поиск в тексте
        text = article.get("text", "").lower()
        if query_lower in text:
            occurrences = text.count(query_lower)
            score += occurrences * 3.0
        
        # Поиск в тегах
        tags = article.get("tags", [])
        for tag in tags:
            if query_lower in str(tag).lower():
                score += 2.0
        
        # Дополнительные факторы
        if len(text) > 0:
            # Более длинные статьи получают небольшой бонус
            score += min(len(text) / 10000, 1.0)
        
        return score
    
    def _normalize_tags(self, tags) -> List[str]:
        """Нормализация тегов"""
        if not tags:
            return []
        
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        elif isinstance(tags, list):
            tags = [str(tag).strip() for tag in tags if tag]
        
        return list(dict.fromkeys(tags))  # Удаляем дубликаты с сохранением порядка
    
    def _parse_tags(self, tags_str: str) -> List[str]:
        """Парсинг тегов из строки"""
        if not tags_str:
            return []
        return [tag.strip() for tag in tags_str.split(",") if tag.strip()]