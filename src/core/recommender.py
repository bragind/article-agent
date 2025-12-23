"""
Рекомендательная система для похожих статей
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class Recommender:
    """Рекомендательная система на основе контента и метаданных"""
    
    def __init__(self, vector_store, embedder, all_articles: List[Dict]):
        self.vector_store = vector_store
        self.embedder = embedder
        self.all_articles = all_articles
        self.article_index = {a["id"]: a for a in all_articles}
    
    def get_similar_articles(self, 
                            article_id: str, 
                            user_id: str = None, 
                            limit: int = 5,
                            min_similarity: float = 0.3) -> List[Dict]:
        """
        Поиск похожих статей
        
        Args:
            article_id: ID целевой статьи
            user_id: ID пользователя для фильтрации
            limit: Количество рекомендаций
            min_similarity: Минимальный порог схожести
            
        Returns:
            Список похожих статей с оценкой схожести
        """
        # Находим целевую статью
        target_article = self.article_index.get(article_id)
        if not target_article:
            logger.warning(f"Статья {article_id} не найдена")
            return []
        
        # Используем векторный поиск если доступен
        if self.vector_store and self.embedder:
            vector_results = self._vector_similarity(target_article, user_id, limit * 2, min_similarity)
            if vector_results:
                return self._format_and_limit_results(vector_results, limit)
        
        # Fallback на метаданные
        return self._metadata_similarity(target_article, user_id, limit)
    
    def _vector_similarity(self, 
                          target_article: Dict, 
                          user_id: str, 
                          limit: int,
                          min_similarity: float) -> List[Dict]:
        """Поиск похожих статей через векторное пространство"""
        try:
            # Подготавливаем текст для эмбеддинга
            text_for_embedding = self._prepare_text_for_embedding(target_article)
            
            # Генерируем эмбеддинг
            embedding = self.embedder.embed([text_for_embedding])
            if embedding is None or len(embedding) == 0:
                return []
            
            # Фильтр по пользователю
            where_filter = None
            if user_id and target_article.get("is_user_document"):
                where_filter = {
                    "$and": [
                        {"source_type": {"$eq": "user"}},
                        {"uploaded_by": {"$eq": user_id}}
                    ]
                }
            
            # Поиск похожих статей
            results = self.vector_store.search(
                embedding[0], 
                top_k=limit,
                where=where_filter
            )
            
            if not results.get("documents"):
                return []
            
            return self._process_vector_results(
                results, target_article["id"], min_similarity
            )
            
        except Exception as e:
            logger.error(f"Ошибка векторного поиска похожих статей: {e}")
            return []
    
    def _prepare_text_for_embedding(self, article: Dict) -> str:
        """Подготовка текста статьи для эмбеддинга"""
        title = article.get("title", "")
        text = article.get("text", "")[:1000]  # Первые 1000 символов
        tags = article.get("tags", [])
        
        # Форматируем теги
        if isinstance(tags, list):
            tags_text = " ".join([str(tag) for tag in tags[:10] if tag])
        elif isinstance(tags, str):
            tags_text = tags
        else:
            tags_text = ""
        
        # Комбинируем компоненты
        combined = f"{title} {tags_text} {text}"
        
        # Очищаем от лишних пробелов
        import re
        combined = re.sub(r'\s+', ' ', combined).strip()
        
        return combined
    
    def _process_vector_results(self, 
                               results: Dict, 
                               target_id: str,
                               min_similarity: float) -> List[Dict]:
        """Обработка результатов векторного поиска"""
        similar_articles = []
        seen_ids = {target_id}
        
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            article_id = meta.get("article_id")
            
            # Пропускаем целевую статью и дубликаты
            if not article_id or article_id in seen_ids:
                continue
            
            # Получаем статью
            article = self.article_index.get(article_id)
            if not article:
                continue
            
            # Вычисляем схожесть
            similarity = 1.0
            if distances and i < len(distances):
                similarity = max(0.0, 1.0 - distances[i])
            
            if similarity < min_similarity:
                continue
            
            # Определяем причину схожести
            reason = self._get_similarity_reason(article)
            
            similar_articles.append({
                "article": article,
                "similarity_score": similarity,
                "reason": reason,
                "source": "vector"
            })
            
            seen_ids.add(article_id)
        
        return similar_articles
    
    def _metadata_similarity(self, 
                            target_article: Dict, 
                            user_id: str, 
                            limit: int) -> List[Dict]:
        """Поиск похожих статей по метаданным"""
        target_tags = set(self._normalize_tags(target_article.get("tags", [])))
        target_author = str(target_article.get("author", "")).lower().strip()
        target_source = target_article.get("source", "")
        
        candidates = []
        
        for article in self.all_articles:
            # Пропускаем целевую статью
            if article["id"] == target_article["id"]:
                continue
            
            # Фильтрация по пользователю
            if user_id and article.get("is_user_document"):
                if article.get("uploaded_by") != user_id:
                    continue
            
            # Вычисляем схожесть по метаданным
            similarity_score = 0.0
            reasons = []
            
            # Схожесть по тегам
            article_tags = set(self._normalize_tags(article.get("tags", [])))
            common_tags = target_tags.intersection(article_tags)
            if common_tags:
                tag_similarity = len(common_tags) / max(len(target_tags), 1) * 0.5
                similarity_score += min(tag_similarity, 0.5)
                reasons.append(f"общие теги: {', '.join(list(common_tags)[:2])}")
            
            # Схожесть по автору
            article_author = str(article.get("author", "")).lower().strip()
            if target_author and article_author and target_author == article_author:
                similarity_score += 0.3
                reasons.append(f"один автор: {target_author}")
            
            # Схожесть по источнику
            if target_source and article.get("source") == target_source:
                similarity_score += 0.1
                reasons.append(f"один источник: {target_source}")
            
            # Схожесть по дате
            try:
                target_date = target_article.get("date", "")[:10]
                article_date = article.get("date", "")[:10]
                if target_date and article_date:
                    if target_date[:7] == article_date[:7]:  # Год-месяц
                        similarity_score += 0.05
                        if not reasons:
                            reasons.append("публикация в одном месяце")
                    elif target_date[:4] == article_date[:4]:  # Год
                        similarity_score += 0.02
            except:
                pass
            
            if similarity_score >= 0.2:
                candidates.append({
                    "article": article,
                    "similarity_score": min(similarity_score, 1.0),
                    "reason": "; ".join(reasons) if reasons else "тематическая схожесть",
                    "source": "metadata"
                })
        
        # Сортировка и ограничение
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:limit]
    
    def _normalize_tags(self, tags) -> List[str]:
        """Нормализация тегов"""
        if not tags:
            return []
        
        if isinstance(tags, str):
            import re
            tags = [tag.strip() for tag in re.split(r'[,;]', tags) if tag.strip()]
        elif isinstance(tags, list):
            tags = [str(tag).strip() for tag in tags if tag]
        
        # Убираем дубликаты
        unique_tags = []
        seen = set()
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower not in seen:
                seen.add(tag_lower)
                unique_tags.append(tag)
        
        return unique_tags[:15]
    
    def _get_similarity_reason(self, article: Dict) -> str:
        """Определение причины схожести для пользователя"""
        reasons = []
        
        if article.get("tags"):
            tags = article["tags"]
            if isinstance(tags, list) and tags:
                reasons.append(f"теги: {', '.join(tags[:2])}")
        
        author = article.get("author")
        if author:
            reasons.append(f"автор: {author}")
        
        source = article.get("source")
        if source:
            reasons.append(f"источник: {source}")
        
        return "; ".join(reasons) if reasons else "тематическая схожесть"
    
    def _format_and_limit_results(self, results: List[Dict], limit: int) -> List[Dict]:
        """Форматирование и ограничение результатов"""
        # Сортировка по схожести
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Возвращаем указанное количество результатов
        return results[:limit]