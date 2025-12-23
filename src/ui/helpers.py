"""
Вспомогательные функции
"""

import streamlit as st
import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def group_search_results(search_results: List[Dict]) -> Dict[str, Dict]:
    """Группирует результаты поиска по статьям"""
    grouped_results = {}
    
    for result in search_results:
        article = result["article"]
        article_id = article.get("id", "")
        
        if not article_id:
            title_key = article.get("title", "").replace(" ", "_")[:50]
            author_key = article.get("author", "").replace(" ", "_")[:20]
            article_id = f"{title_key}_{author_key}"
        
        if article_id not in grouped_results:
            grouped_results[article_id] = {
                "metadata": {
                    "id": article_id,
                    "title": article.get("title", "Без названия"),
                    "author": article.get("author", "Не указан"),
                    "url": article.get("url", ""),
                    "source": article.get("source", "Unknown"),
                    "date": article.get("date", ""),
                    "tags": article.get("tags", []),
                    "is_user_document": result.get("is_user_document", False),
                    "uploaded_by": article.get("uploaded_by", "system"),
                    "total_chunks": 0,
                    "total_characters": 0
                },
                "chunks": [],
                "total_score": 0.0
            }
        
        chunk_data = {
            "text": article.get("text", ""),
            "score": result.get("score", 0.0),
            "characters": len(article.get("text", ""))
        }
        
        if "chunk_index" in article:
            chunk_data["chunk_index"] = article.get("chunk_index", 0)
        
        grouped_results[article_id]["chunks"].append(chunk_data)
        grouped_results[article_id]["total_score"] += result.get("score", 0.0)
        grouped_results[article_id]["metadata"]["total_chunks"] += 1
        grouped_results[article_id]["metadata"]["total_characters"] += chunk_data["characters"]
    
    return grouped_results


def get_relevance_badge(score: float) -> str:
    """Возвращает текстовое обозначение релевантности"""
    from .constants import RELEVANCE_BADGES
    
    if score > RELEVANCE_BADGES["high"]["threshold"]:
        return RELEVANCE_BADGES["high"]["text"]
    elif score > RELEVANCE_BADGES["medium"]["threshold"]:
        return RELEVANCE_BADGES["medium"]["text"]
    else:
        return RELEVANCE_BADGES["low"]["text"]


def load_pdf_file(uploaded_file) -> Dict[str, Any]:
    """Обрабатывает загруженный PDF файл"""
    try:
        import pdfplumber
        from io import BytesIO

        file_bytes = uploaded_file.read()
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            full_text = ""
            pages_data = []

            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                full_text += f"\n\n--- Страница {page_num} ---\n{page_text}"

                pages_data.append({
                    "page_number": page_num,
                    "text": page_text,
                    "char_count": len(page_text),
                    "word_count": len(page_text.split()),
                })

            article_data = {
                "title": uploaded_file.name.replace(".pdf", "").replace("_", " ").title(),
                "author": "",
                "date": datetime.now().isoformat(),
                "text": full_text.strip(),
                "tags": [],
                "url": f"file://{uploaded_file.name}",
                "source": "User Upload",
                "views": 0,
                "rating": 0,
                "parsed_at": datetime.now().isoformat(),
                "text_length": len(full_text.strip()),
                "has_content": len(full_text.strip()) > 100,
                "pages": pages_data,
            }
            
            return article_data
            
    except Exception as e:
        st.error(f"Ошибка обработки PDF: {e}")
        logger.error(f"Ошибка обработки PDF: {e}")
        return None


def get_scope_enum(scope_str: str):
    """Конвертирует строку в SearchScope"""
    try:
        from src.core.search_engine import SearchScope
        
        if scope_str == "all":
            return SearchScope.ALL
        elif scope_str == "user":
            return SearchScope.USER_ONLY
        elif scope_str == "habr":
            return SearchScope.HABR_ONLY
        return SearchScope.ALL
    except ImportError as e:
        logger.error(f"Ошибка импорта SearchScope: {e}")
        # Возвращаем значение по умолчанию
        try:
            # Пробуем другой импорт
            from src.core.rag_orchestrator import SearchScope
            return SearchScope.ALL
        except:
            # Создаем простой класс для заглушки
            class SimpleScope:
                ALL = "all"
                USER_ONLY = "user"
                HABR_ONLY = "habr"
            return SimpleScope.ALL


def create_enhanced_statistics(rag_agent, user_id: str) -> Dict:
    """Создает расширенную статистику"""
    stats = rag_agent.get_statistics(user_id)
    all_articles = rag_agent.all_articles if hasattr(rag_agent, 'all_articles') else []
    
    # Базовые метрики
    enhanced_stats = {
        "total_articles": stats.get("total_articles", 0),
        "habr_articles": stats.get("habr_articles", 0),
        "user_articles": stats.get("user_articles", 0),
        "current_user_articles": stats.get("current_user_articles", 0),
        "total_chunks": stats.get("total_chunks", 0),
        "vector_search_enabled": stats.get("vector_search_enabled", False),
        "llm_enabled": stats.get("llm_enabled", False),
        "last_update": stats.get("last_update", ""),
    }
    
    # Дополнительная аналитика
    if all_articles:
        # Анализ тегов
        all_tags = {}
        for article in all_articles:
            tags = article.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    if tag:
                        tag_str = str(tag).strip()
                        all_tags[tag_str] = all_tags.get(tag_str, 0) + 1
            elif isinstance(tags, str) and tags:
                tag_str = tags.strip()
                all_tags[tag_str] = all_tags.get(tag_str, 0) + 1
        
        enhanced_stats["top_tags"] = dict(sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Анализ авторов
        authors = {}
        for article in all_articles:
            author = article.get("author", "")
            if author and author.strip() and author.lower() not in ["неизвестен", "unknown"]:
                authors[author] = authors.get(author, 0) + 1
        
        enhanced_stats["top_authors"] = dict(sorted(authors.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Анализ по датам
        dates = {}
        for article in all_articles:
            date = article.get("date", "")
            if date and len(date) >= 10:
                date_key = date[:7]  # Год-месяц
                dates[date_key] = dates.get(date_key, 0) + 1
        
        enhanced_stats["articles_by_month"] = dict(sorted(dates.items()))
        
        # Анализ длины статей
        article_lengths = [len(article.get("text", "")) for article in all_articles if article.get("text")]
        if article_lengths:
            enhanced_stats["avg_article_length"] = int(np.mean(article_lengths))
            enhanced_stats["max_article_length"] = max(article_lengths)
            enhanced_stats["min_article_length"] = min(article_lengths)
    
    return enhanced_stats


def migrate_existing_documents(project_root: Path):
    """Миграция существующих документов в новую структуру"""
    try:
        import shutil
        
        data_dir = project_root / "data"
        old_uploads_dir = data_dir / "user_uploads"
        
        if not old_uploads_dir.exists():
            return "Нет старых документов для миграции"
        
        migrated_count = 0
        error_count = 0
        
        # Обрабатываем все JSON файлы в корне user_uploads
        for filepath in old_uploads_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                
                # Определяем пользователя
                user_id = article.get("uploaded_by", "unknown")
                if user_id == "unknown":
                    # Пытаемся извлечь из имени файла
                    if "user_" in filepath.stem:
                        user_id = filepath.stem.split("user_")[-1].split("_")[0][:8]
                    else:
                        user_id = "legacy_user"
                
                # Создаем целевую директорию
                user_dir = old_uploads_dir / f"user_{user_id}"
                user_dir.mkdir(exist_ok=True)
                
                # Перемещаем файл
                target_path = user_dir / filepath.name
                shutil.move(str(filepath), str(target_path))
                
                # Обновляем метаданные
                article["uploaded_by"] = user_id
                with open(target_path, 'w', encoding='utf-8') as f:
                    json.dump(article, f, ensure_ascii=False, indent=2)
                
                migrated_count += 1
                
            except Exception as e:
                error_count += 1
                logger.error(f"Ошибка миграции {filepath.name}: {e}")
        
        return f"Миграция завершена: успешно {migrated_count}, ошибок {error_count}"
        
    except Exception as e:
        logger.error(f"Ошибка выполнения миграции: {e}")
        return f"Ошибка миграции: {e}"