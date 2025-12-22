# src/streamlit_app.py
"""
Streamlit UI - Простой и понятный интерфейс с широкими кнопками
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Подготовка пути к проекту
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Импорты RAG
try:
    from src.core.rag import RAGAgent, SearchScope
    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False
    RAG_IMPORT_ERROR = e
    st.error(f"Ошибка импорта RAG: {e}")

# Импорт модуля аутентификации
try:
    from src.auth.auth import streamlit_auth, auth_manager
    AUTH_AVAILABLE = True
except ImportError as e:
    AUTH_AVAILABLE = False
    AUTH_IMPORT_ERROR = e

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

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
    if score > 0.8:
        return "🟢 Высокая"
    elif score > 0.5:
        return "🟡 Средняя"
    else:
        return "🔵 Низкая"

def search_with_filters(query: str, user_id: str, scope: SearchScope, limit: int, 
                       tags: Optional[List[str]] = None, author: Optional[str] = None, 
                       date_from: Optional[str] = None, date_to: Optional[str] = None) -> Dict:
    """Поиск с фильтрами"""
    if not st.session_state.get("rag_agent"):
        st.error("RAG агент не инициализирован")
        return {}
    
    try:
        logger.info(f"Поиск с фильтрами: запрос='{query}', теги={tags}, автор={author}, дата от={date_from}, дата до={date_to}")
        
        # Нормализация параметров
        norm_tags = None
        if tags and isinstance(tags, list) and len(tags) > 0:
            norm_tags = [str(t).strip() for t in tags if t and str(t).strip()]
            if len(norm_tags) == 0:
                norm_tags = None
        
        norm_author = None
        if author and isinstance(author, str) and author.strip():
            norm_author = author.strip()
            if norm_author.lower() in ["все авторы", "all authors", "любой автор"]:
                norm_author = None
        
        norm_date_from = None
        if date_from and isinstance(date_from, str) and date_from.strip():
            try:
                datetime.strptime(date_from[:10], "%Y-%m-%d")
                norm_date_from = date_from[:10]
            except:
                norm_date_from = None
        
        norm_date_to = None
        if date_to and isinstance(date_to, str) and date_to.strip():
            try:
                datetime.strptime(date_to[:10], "%Y-%m-%d")
                norm_date_to = date_to[:10]
            except:
                norm_date_to = None
        
        # Вызываем поиск с фильтрами
        search_results = st.session_state.rag_agent.search_with_filters(
            query=query,
            user_id=user_id,
            scope=scope,
            limit=limit,
            tags=norm_tags,
            author=norm_author,
            date_from=norm_date_from,
            date_to=norm_date_to
        )
        
        # Группировка результатов
        grouped = group_search_results(search_results)
        
        sorted_articles = sorted(
            grouped.items(),
            key=lambda x: x[1]["total_score"],
            reverse=True
        )[:limit]
        
        return dict(sorted_articles)
        
    except Exception as e:
        logger.error(f"Ошибка при поиске с фильтрами: {e}")
        st.error(f"❌ **Ошибка при поиске:** {str(e)[:200]}")
        
        # Fallback: обычный поиск без фильтров
        try:
            search_results = st.session_state.rag_agent.search(
                query, user_id, scope, limit * 3
            )
            grouped = group_search_results(search_results)
            sorted_articles = sorted(
                grouped.items(),
                key=lambda x: x[1]["total_score"],
                reverse=True
            )[:limit]
            return dict(sorted_articles)
        except Exception as e2:
            logger.error(f"И обычный поиск не работает: {e2}")
            return {}

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
                "uploaded_by": st.session_state.user_id,
            }
            
            return article_data
            
    except Exception as e:
        st.error(f"Ошибка обработки PDF: {e}")
        logger.error(f"Ошибка обработки PDF: {e}")
        return None

def get_scope_enum(scope_str: str) -> Optional[SearchScope]:
    """Конвертирует строку в SearchScope"""
    if not RAG_AVAILABLE:
        return None

    if scope_str == "all":
        return SearchScope.ALL
    elif scope_str == "user":
        return SearchScope.USER_ONLY
    elif scope_str == "habr":
        return SearchScope.HABR_ONLY
    return SearchScope.ALL

def display_similar_articles(article_id: str, article_metadata: Dict, user_id: str):
    """Отображение похожих статей для конкретной статьи"""
    if not st.session_state.get("rag_agent"):
        st.error("RAG агент не инициализирован")
        return
    
    try:
        similar_articles = st.session_state.rag_agent.get_similar_articles(
            article_id, 
            user_id, 
            limit=5
        )
        
        if similar_articles:
            st.markdown(f"**Найдено {len(similar_articles)} похожих статей:**")
            
            for i, similar in enumerate(similar_articles, 1):
                similar_article = similar["article"]
                score = similar.get("similarity_score", 0)
                reason = similar.get("reason", "")
                
                with st.container():
                    col_title, col_score = st.columns([3, 1])
                    with col_title:
                        st.markdown(f"**{i}. {similar_article['title']}**")
                    with col_score:
                        if score > 0:
                            st.progress(
                                min(score, 1.0), 
                                text=f"Схожесть: {score:.0%}"
                            )
                    
                    if reason:
                        st.caption(f"📌 {reason}")
                    
                    source_type = "📁 Ваш документ" if similar_article.get("is_user_document") else "🌐 Статья Habr"
                    st.caption(
                        f"{source_type} • 👤 {similar_article.get('author', 'Неизвестен')}"
                    )
                    
                    tags = similar_article.get("tags", [])
                    if tags:
                        if isinstance(tags, str):
                            st.markdown(f"**Теги:** `{tags}`")
                        elif isinstance(tags, list):
                            tag_text = ", ".join([f"`{tag}`" for tag in tags[:3]])
                            if len(tags) > 3:
                                tag_text += f" и еще {len(tags) - 3}"
                            st.markdown(f"**Теги:** {tag_text}")
                    
                    preview = similar_article.get("text", "")[:150]
                    if preview:
                        st.markdown(f"*{preview}...*")
                    
                    # Кнопка для открытия статьи
                    if similar_article.get("url") and not similar_article.get("url", "").startswith("file://"):
                        st.link_button("🔗 Открыть статью", similar_article["url"])
                    
                    st.divider()
        else:
            st.info("Похожих статей не найдено.")
    
    except Exception as e:
        st.error(f"Ошибка получения похожих статей: {e}")

def display_questions_for_article(article: Dict):
    """Отображение вопросов для самопроверки по статье"""
    if not st.session_state.get("rag_agent"):
        st.error("RAG агент не инициализирован")
        return
    
    with st.spinner("🤔 Генерирую вопросы для самопроверки..."):
        try:
            # Используем текст статьи для генерации вопросов
            article_text = article.get("text", "")[:2000]
            
            if article_text:
                # Проверяем, доступен ли LLMClient
                if hasattr(st.session_state.rag_agent, 'llm_client') and st.session_state.rag_agent.llm_client:
                    questions = st.session_state.rag_agent.llm_client.generate_questions(
                        article_text, 
                        num_questions=5
                    )
                else:
                    # Если LLM недоступен, используем заглушку
                    questions = [
                        f"Какие ключевые идеи представлены в статье?",
                        f"Какую проблему решает автор статьи?",
                        f"Какие методы или технологии обсуждаются в статье?",
                        f"Какие выводы можно сделать из прочитанного?",
                        f"Как эти знания можно применить на практике?"
                    ]
                
                # Отображаем вопросы
                st.markdown("### ❓ Вопросы для самопроверки")
                
                for i, question in enumerate(questions, 1):
                    st.markdown(f"**{i}. {question}**")
                    st.markdown("")  # Пустая строка для разделения
                
                st.markdown("---")
                st.info("💡 Попробуйте ответить на вопросы самостоятельно, затем проверьте свои ответы в статье.")
                
            else:
                st.warning("Недостаточно текста для генерации вопросов.")
                
        except Exception as e:
            st.error(f"Ошибка генерации вопросов: {e}")
            logger.error(f"Ошибка генерации вопросов: {e}")

def display_article_details(article_metadata: Dict, chunks: List[Dict]):
    """Отображение подробной информации о статье"""
    st.markdown("### 📖 Подробная информация")
    
    # Статистика статьи
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего фрагментов", article_metadata['total_chunks'])
    with col2:
        st.metric("Всего символов", f"{article_metadata['total_characters']:,}")
    with col3:
        relevance_score = min(sum(chunk.get("score", 0) for chunk in chunks) / max(len(chunks), 1), 1.0)
        st.metric("Средняя релевантность", f"{relevance_score:.0%}")
    
    st.markdown("---")
    
    # Релевантные фрагменты
    st.markdown("**📄 Наиболее релевантные фрагменты:**")
    
    for chunk_idx, chunk in enumerate(chunks[:3], 1):
        st.markdown(f"**Фрагмент {chunk_idx}**")
        
        preview = chunk["text"]
        if len(preview) > 300:
            preview = preview[:300] + "..."
        
        st.markdown(f"*{preview}*")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.caption(f"📄 {chunk['characters']:,} символов")
        with col_info2:
            relevance = get_relevance_badge(chunk.get("score", 0))
            st.caption(f"**Релевантность:** {relevance}")
        
        if chunk_idx < min(3, len(chunks)):
            st.divider()

def display_article_results(grouped_results: Dict, user_id: str):
    """Отображает результаты поиска статей в новом формате"""
    for i, (article_id, article_data) in enumerate(grouped_results.items(), 1):
        metadata = article_data["metadata"]
        chunks = article_data["chunks"]
        
        # Сортируем чанки по релевантности
        chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        source_type = "📁 Ваш документ" if metadata.get("is_user_document") else "🌐 Статья Habr"
        relevance_score = min(article_data["total_score"] / max(len(chunks), 1), 1.0)
        
        with st.container():
            # Заголовок и метаданные
            st.markdown(f"### {i}. {metadata['title']}")
            st.markdown(f"**{source_type}**")
            
            # Информация об источнике в компактном формате
            st.markdown(f"👤 **Автор:** {metadata['author']}")
            
            if metadata.get('date'):
                st.markdown(f"📅 **Дата:** {metadata['date'][:10]}")
            
            st.markdown(f"📊 **Фрагментов:** {metadata['total_chunks']}")
            st.markdown(f"🎯 **Релевантность:** {relevance_score:.0%}")
            
            # Теги
            tags = metadata.get("tags", [])
            if tags:
                if isinstance(tags, str):
                    tags_display = tags
                elif isinstance(tags, list):
                    tags_display = ", ".join([f"`{tag}`" for tag in tags[:5]])
                st.markdown(f"🏷️ **Теги:** {tags_display}")
            
            st.markdown("---")
            
            # ШИРОКАЯ КНОПКА "ОТКРЫТЬ СТАТЬЮ"
            if metadata['url'] and not metadata['url'].startswith("file://"):
                st.link_button(
                    "🔗 Открыть статью", 
                    metadata['url'], 
                    use_container_width=True,
                    help="Открыть оригинал статьи в новом окне"
                )
            else:
                st.button(
                    "📄 Ваш документ", 
                    key=f"doc_{article_id}", 
                    disabled=True,
                    use_container_width=True,
                    help="Это ваш загруженный документ"
                )
            
            # ШИРОКАЯ КНОПКА "ВОПРОСЫ" с разворачивающимся содержимым
            with st.expander("❓ Вопросы для самопроверки", expanded=False):
                display_questions_for_article({
                    "title": metadata['title'],
                    "text": chunks[0]["text"] if chunks else "",
                    "author": metadata['author'],
                    "date": metadata['date']
                })
            
            # ШИРОКАЯ КНОПКА "ПОХОЖИЕ" с разворачивающимся содержимым
            with st.expander("🔍 Похожие статьи", expanded=False):
                display_similar_articles(article_id, metadata, user_id)
            
            # ШИРОКАЯ КНОПКА "ПОДРОБНЕЕ" с разворачивающимся содержимым
            with st.expander("📖 Подробная информация", expanded=False):
                display_article_details(metadata, chunks)
            
            # Разделитель между статьями
            st.markdown("<hr style='margin: 30px 0; border: 1px solid #ddd;'>", unsafe_allow_html=True)

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

def display_enhanced_statistics(rag_agent, user_id: str):
    """Отображает расширенную статистику"""
    if not rag_agent:
        st.error("RAG агент не инициализирован")
        return
    
    try:
        stats = create_enhanced_statistics(rag_agent, user_id)
        
        # Основные метрики в виде карточек
        st.subheader("📊 Основные метрики")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "📚 Всего статей", 
                stats["total_articles"],
                help="Всего статей в базе данных"
            )
        with col2:
            st.metric(
                "🌐 Статей Habr", 
                stats["habr_articles"],
                help="Статей из источников Habr"
            )
        with col3:
            st.metric(
                "📁 Пользовательских", 
                stats["user_articles"],
                help="Загруженных пользователями документов"
            )
        with col4:
            st.metric(
                "👤 Ваших документов", 
                stats["current_user_articles"],
                help="Документов, загруженных вами"
            )
        
        st.markdown("---")
        
        # Графики в двух колонках
        col_left, col_right = st.columns(2)
        
        with col_left:
            # График распределения по источникам
            if stats["habr_articles"] > 0 or stats["user_articles"] > 0:
                st.subheader("📈 Распределение по источникам")
                sources_data = {
                    "Habr": stats["habr_articles"],
                    "Пользовательские": stats["user_articles"]
                }
                
                fig1 = px.pie(
                    values=list(sources_data.values()), 
                    names=list(sources_data.keys()), 
                    title="Источники документов",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hole=0.3
                )
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)
            
            # Топ тегов
            if "top_tags" in stats and stats["top_tags"]:
                st.subheader("🏷️ Топ-10 тегов")
                tags_df = pd.DataFrame(
                    list(stats["top_tags"].items()), 
                    columns=["Тег", "Количество статей"]
                )
                fig2 = px.bar(
                    tags_df, 
                    x="Тег", 
                    y="Количество статей",
                    title="Популярные теги",
                    color="Количество статей",
                    color_continuous_scale="viridis"
                )
                fig2.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)
        
        with col_right:
            # Топ авторов
            if "top_authors" in stats and stats["top_authors"]:
                st.subheader("👥 Топ-10 авторов")
                authors_df = pd.DataFrame(
                    list(stats["top_authors"].items()), 
                    columns=["Автор", "Количество статей"]
                )
                fig3 = px.bar(
                    authors_df, 
                    x="Автор", 
                    y="Количество статей",
                    title="Популярные авторы",
                    color="Количество статей",
                    color_continuous_scale="plasma"
                )
                fig3.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig3, use_container_width=True)
            
            # Статьи по месяцам
            if "articles_by_month" in stats and stats["articles_by_month"]:
                st.subheader("📅 Статьи по месяцам")
                months_df = pd.DataFrame(
                    list(stats["articles_by_month"].items()), 
                    columns=["Месяц", "Количество статей"]
                )
                fig4 = px.line(
                    months_df, 
                    x="Месяц", 
                    y="Количество статей",
                    title="Динамика публикаций",
                    markers=True
                )
                fig4.update_traces(line=dict(color='royalblue', width=3))
                st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("---")
        
        # Дополнительная информация
        st.subheader("ℹ️ Системная информация")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.markdown("**📊 Производительность:**")
            st.markdown(f"• Векторный поиск: {'✅ Активен' if stats['vector_search_enabled'] else '❌ Недоступен'}")
            st.markdown(f"• LLM генерация: {'✅ Активна' if stats['llm_enabled'] else '❌ Недоступна'}")
            st.markdown(f"• Всего чанков: {stats.get('total_chunks', 0):,}")
        
        with info_col2:
            st.markdown("**📈 Аналитика статей:**")
            if "avg_article_length" in stats:
                st.markdown(f"• Средняя длина: {stats['avg_article_length']:,} симв.")
                st.markdown(f"• Максимальная: {stats['max_article_length']:,} симв.")
                st.markdown(f"• Минимальная: {stats['min_article_length']:,} симв.")
            else:
                st.markdown("• Данные о длине недоступны")
        
        with info_col3:
            st.markdown("**🔄 Обновление:**")
            st.markdown(f"• Последнее обновление: {stats['last_update'][:19]}")
            st.markdown(f"• ID пользователя: `{user_id}`")
            st.markdown(f"• Всего уникальных тегов: {len(stats.get('top_tags', {}))}")
        
        # Кнопка обновления статистики
        if st.button("🔄 Обновить статистику", key="refresh_stats"):
            st.rerun()
            
    except Exception as e:
        st.error(f"Ошибка отображения статистики: {e}")
        logger.error(f"Ошибка отображения статистики: {e}")

# ==================== ПРОВЕРКА АВТОРИЗАЦИИ ====================
if AUTH_AVAILABLE:
    if not streamlit_auth.check_auth():
        st.stop()
else:
    if "user_id" not in st.session_state:
        st.session_state.user_id = hashlib.md5("demo_user".encode()).hexdigest()[:8]
        st.session_state.username = "demo"
        st.session_state.user_role = "user"
        st.session_state.authenticated = True

# ==================== НАСТРОЙКИ СТРАНИЦЫ ====================
st.set_page_config(
    page_title="AI RAG-агент",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== ИНИЦИАЛИЗАЦИЯ СЕССИИ ====================
if "rag_agent" not in st.session_state:
    if RAG_AVAILABLE:
        try:
            with st.spinner("🤖 Инициализация RAG агента..."):
                st.session_state.rag_agent = RAGAgent(data_dir=str(PROJECT_ROOT / "data"))
                st.session_state.initialized = True
                logger.info("RAG агент успешно инициализирован")
        except Exception as e:
            st.session_state.initialized = False
            st.error(f"Ошибка инициализации RAGAgent: {e}")
            logger.error(f"Ошибка инициализации RAGAgent: {e}")
    else:
        st.session_state.initialized = False
        st.error(f"RAGAgent не импортирован: {RAG_IMPORT_ERROR}")

if "user_id" not in st.session_state:
    st.session_state.user_id = hashlib.md5("demo_user".encode()).hexdigest()[:8]

if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False

if "search_tags" not in st.session_state:
    st.session_state.search_tags = []

if "search_author" not in st.session_state:
    st.session_state.search_author = None

if "search_date_from" not in st.session_state:
    st.session_state.search_date_from = None

if "search_date_to" not in st.session_state:
    st.session_state.search_date_to = None

# ==================== САЙДБАР ====================
with st.sidebar:
    # Профиль пользователя
    if AUTH_AVAILABLE:
        streamlit_auth.show_user_profile()
    else:
        st.subheader("👤 Профиль (демо)")
        st.info(f"ID: `{st.session_state.user_id}`")
        if st.button("🚪 Выйти"):
            st.session_state.authenticated = False
            st.rerun()
    
    st.title("🤖 AI RAG-агент")
    st.markdown("---")
    
    # Загрузка документов
    if st.session_state.get("authenticated", True) and st.session_state.get("user_role", "user") in ["admin", "user"]:
        st.subheader("📤 Загрузить документы")
        uploaded_file = st.file_uploader(
            "Выберите PDF файл", 
            type=["pdf"], 
            help="Максимальный размер: 100MB",
            label_visibility="collapsed",
            key="pdf_uploader"
        )
        
        if uploaded_file is not None:
            if st.button("📥 Загрузить и обработать", key="sb_upload", use_container_width=True):
                with st.spinner("📄 Обработка PDF..."):
                    article_data = load_pdf_file(uploaded_file)
                    
                    if article_data and article_data.get("has_content"):
                        article_id = st.session_state.rag_agent.add_user_article(
                            article_data, st.session_state.user_id
                        )
                        
                        if article_id:
                            st.success(f"✅ Документ успешно добавлен!")
                            st.info(f"**Название:** {article_data['title']}")
                            st.info(f"**Страниц:** {len(article_data.get('pages', []))}")
                            st.info(f"**Символов:** {article_data['text_length']:,}")
                        else:
                            st.error("❌ Ошибка при добавлении документа")
                    else:
                        st.error("❌ В документе недостаточно текста")
    
    st.markdown("---")
    
    # Быстрые действия
    st.subheader("⚡ Быстрые действия")
    
    if st.button("🔄 Обновить систему", key="sb_refresh", use_container_width=True):
        st.rerun()
    
    if st.button("🗑️ Очистить фильтры", key="sb_clear_filters", use_container_width=True):
        for key in ["search_tags", "search_author", "search_date_from", "search_date_to"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    st.caption("© 2025 AI RAG-агент | DreamTeam")

# ==================== ГЛАВНАЯ СТРАНИЦА ====================
st.title("🔍 AI RAG-агент для поиска статей")

# Приветствие
if AUTH_AVAILABLE and st.session_state.get("authenticated"):
    user_role_display = {
        "admin": "👑 Администратор",
        "user": "👤 Пользователь",
        "guest": "👥 Гость"
    }
    role = user_role_display.get(st.session_state.user_role, "👤 Пользователь")
    st.markdown(f"### Привет, **{st.session_state.username}!** ({role})")

st.markdown(
    """
    *Интеллектуальный поиск и анализ IT-статей с вопросами для самопроверки*
    
    **✨ Возможности:**
    - 🔍 **Умный поиск** по статьям Habr и вашим документам
    - 🏷️ **Фильтрация** по тегам, автору, дате публикации
    - ❓ **Вопросы для самопроверки** по каждой статье
    - 🔍 **Поиск похожих статей** по тематике
    - 📊 **Расширенная статистика** и аналитика
    """
)

# ==================== СОЗДАНИЕ ВКЛАДОК ПО РОЛЯМ ====================
user_role = st.session_state.get("user_role", "user")

if user_role == "admin":
    tabs = st.tabs(["🔍 Поиск", "📚 Мои документы", "📊 Статистика", "👑 Админ"])
    tab1, tab3, tab4, tab5 = tabs
elif user_role == "user":
    tabs = st.tabs(["🔍 Поиск", "📚 Мои документы", "📊 Статистика"])
    tab1, tab3, tab4 = tabs
    tab5 = None
else:  # guest
    tabs = st.tabs(["🔍 Поиск", "📊 Статистика"])
    tab1, tab4 = tabs
    tab3, tab5 = None, None

# ==================== ВКЛАДКА 1: ПОИСК ====================
with tab1:
    # Заголовок и описание
    st.header("🔍 Поиск статей")
    
    # КОНТЕЙНЕР ПАРАМЕТРОВ ПОИСКА
    with st.container():
        st.subheader("⚙️ Параметры поиска")
        
        # Область поиска и количество результатов
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🌍 Область поиска**")
            if st.session_state.get("authenticated", True) and st.session_state.get("user_role", "user") != "guest":
                search_scope = st.radio(
                    "Где искать:",
                    options=["🔎 Везде", "📁 Только мои документы", "🌐 Только Habr"],
                    index=0,
                    key="search_scope_radio",
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                if search_scope == "🔎 Везде":
                    st.session_state.search_scope = "all"
                elif search_scope == "📁 Только мои документы":
                    st.session_state.search_scope = "user"
                elif search_scope == "🌐 Только Habr":
                    st.session_state.search_scope = "habr"
            else:
                st.info("👥 **Гостевой доступ:** поиск только по Habr")
                st.session_state.search_scope = "habr"
        
        with col2:
            st.markdown("**📈 Количество результатов**")
            search_limit = st.slider(
                "Сколько результатов показывать:",
                min_value=3,
                max_value=20,
                value=10,
                key="search_limit_slider",
                label_visibility="collapsed"
            )
        
        st.divider()
        
        # ФИЛЬТРЫ ПОИСКА
        st.markdown("**🎯 Дополнительные фильтры**")
        
        # Получаем доступные теги и авторов
        all_tags = []
        all_authors = []
        if st.session_state.get("rag_agent"):
            try:
                all_tags = st.session_state.rag_agent.get_all_tags(st.session_state.user_id)
                all_authors = st.session_state.rag_agent.get_all_authors(st.session_state.user_id)
            except Exception as e:
                logger.error(f"Ошибка загрузки фильтров: {e}")
        
        # ФИЛЬТРЫ В КОЛОНКАХ
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            with st.expander("🏷️ Теги", expanded=False):
                if all_tags:
                    tag_search = st.text_input(
                        "Поиск тегов:",
                        placeholder="Введите тег...",
                        key="tag_search_input",
                        help="Начните вводить текст, чтобы найти тег"
                    )
                    
                    filtered_tags = all_tags
                    if tag_search:
                        filtered_tags = [tag for tag in all_tags if tag_search.lower() in tag.lower()]
                    
                    if filtered_tags:
                        selected_tags = st.multiselect(
                            "Выберите теги:",
                            options=filtered_tags,
                            default=st.session_state.get("search_tags", []),
                            help="Выберите один или несколько тегов",
                            key="tag_filter_select"
                        )
                        if selected_tags:
                            st.session_state.search_tags = selected_tags
                        else:
                            st.session_state.search_tags = []
                        
                        st.caption(f"Найдено тегов: {len(filtered_tags)} из {len(all_tags)}")
                    else:
                        st.info(f"Теги не найдены по запросу: '{tag_search}'")
                        st.session_state.search_tags = []
                else:
                    st.info("Нет тегов для фильтрации")
                    st.session_state.search_tags = []
        
        with filter_col2:
            with st.expander("👤 Автор", expanded=False):
                if all_authors:
                    author_search = st.text_input(
                        "Поиск авторов:",
                        placeholder="Введите имя автора...",
                        key="author_search_input",
                        help="Начните вводить имя автора"
                    )
                    
                    filtered_authors = all_authors
                    if author_search:
                        filtered_authors = [author for author in all_authors 
                                        if author_search.lower() in author.lower()]
                    
                    if filtered_authors:
                        selected_author = st.selectbox(
                            "Выберите автора:",
                            options=["Все авторы"] + filtered_authors,
                            index=0,
                            help="Выберите автора для фильтрации",
                            key="author_filter_select"
                        )
                        if selected_author and selected_author != "Все авторов":
                            st.session_state.search_author = selected_author
                        else:
                            st.session_state.search_author = None
                        
                        st.caption(f"Найдено авторов: {len(filtered_authors)} из {len(all_authors)}")
                    else:
                        st.info(f"Авторы не найдены по запросу: '{author_search}'")
                        st.session_state.search_author = None
                else:
                    st.info("Авторы не найдены")
                    st.session_state.search_author = None
        
        with filter_col3:
            with st.expander("📅 Дата публикации", expanded=False):
                date_col1, date_col2 = st.columns(2)
                with date_col1:
                    date_from = st.date_input(
                        "От:",
                        value=None,
                        help="Статьи, опубликованные после этой даты",
                        key="date_from_filter"
                    )
                    st.session_state.search_date_from = date_from.isoformat() if date_from else None
                
                with date_col2:
                    date_to = st.date_input(
                        "До:",
                        value=None,
                        help="Статьи, опубликованные до этой даты",
                        key="date_to_filter"
                    )
                    st.session_state.search_date_to = date_to.isoformat() if date_to else None
        
        # Индикатор активных фильтров
        active_filters = []
        
        if st.session_state.get("search_tags") and len(st.session_state.search_tags) > 0:
            tags_display = ", ".join(st.session_state.search_tags[:3])
            if len(st.session_state.search_tags) > 3:
                tags_display += f" (+{len(st.session_state.search_tags) - 3})"
            active_filters.append(f"Теги: {tags_display}")
        
        if st.session_state.get("search_author"):
            active_filters.append(f"Автор: {st.session_state.search_author}")
        
        date_range = ""
        if st.session_state.get("search_date_from"):
            date_range += f"от {st.session_state.search_date_from[:10]} "
        if st.session_state.get("search_date_to"):
            date_range += f"до {st.session_state.search_date_to[:10]}"
        if date_range.strip():
            active_filters.append(f"Дата: {date_range.strip()}")
        
        if active_filters:
            st.info("**🎯 Активные фильтры:** " + " • ".join(active_filters))
        
        st.divider()
    
    # ПОЛЕ ПОИСКА И КНОПКА
    st.subheader("🔍 Введите запрос")
    
    search_col1, search_col2 = st.columns([4, 1])
    
    with search_col1:
        search_query = st.text_input(
            "Введите ваш запрос:",
            placeholder="Например: RAG архитектура, машинное обучение, Python...",
            key="search_input",
            label_visibility="collapsed"
        )
    
    with search_col2:
        search_button = st.button("🚀 Найти", key="search_button", type="primary", use_container_width=True)
    
    # ОБЛАСТЬ РЕЗУЛЬТАТОВ ПОИСКА
    if search_query and (search_button or st.session_state.get("search_triggered")):
        if search_button:
            st.session_state.search_triggered = True
            st.session_state.last_search_query = search_query
        
        # Определяем область поиска
        if st.session_state.get("authenticated", True) and st.session_state.get("user_role", "user") != "guest":
            scope = get_scope_enum(st.session_state.get("search_scope", "all"))
        else:
            scope = SearchScope.HABR_ONLY
        
        if scope:
            with st.spinner(f"🔍 Ищу по запросу: '{search_query}'..."):
                grouped_results = search_with_filters(
                    search_query,
                    st.session_state.user_id,
                    scope=scope,
                    limit=search_limit,
                    tags=st.session_state.get("search_tags", []),
                    author=st.session_state.get("search_author"),
                    date_from=st.session_state.get("search_date_from"),
                    date_to=st.session_state.get("search_date_to")
                )
                
                if not grouped_results:
                    st.warning("📭 По вашему запросу ничего не найдено")
                    
                    with st.expander("💡 Советы по улучшению поиска", expanded=False):
                        st.markdown("""
                        **🎯 Попробуйте:**
                        1. Использовать другие ключевые слова
                        2. Проверить правильность написания
                        3. Убрать некоторые фильтры
                        4. Изменить область поиска
                        5. Использовать более общий запрос
                        """)
                else:
                    # Краткая сводка от RAG агента
                    with st.expander("📝 Краткая сводка по запросу", expanded=True):
                        try:
                            result = st.session_state.rag_agent.generate_answer(
                                search_query, st.session_state.user_id, scope
                            )
                            st.markdown(result.get("answer", ""))
                            
                            # Показываем источники
                            if result.get("sources"):
                                st.markdown("**📚 Использованные источники:**")
                                for i, source in enumerate(result["sources"][:3], 1):
                                    icon = "📁" if source.get("is_user_document") else "🌐"
                                    st.markdown(f"{i}. {icon} **{source['title']}** - {source['author']}")
                        except Exception as e:
                            st.error(f"Ошибка генерации сводки: {e}")
                    
                    st.success(f"✅ Найдено {len(grouped_results)} статей")
                    
                    # Отображение результатов
                    display_article_results(grouped_results, st.session_state.user_id)
        
        # Кнопка нового поиска
        if st.session_state.get("search_triggered"):
            if st.button("🔄 Новый поиск", key="new_search", type="secondary"):
                st.session_state.search_triggered = False
                st.session_state.last_search_query = ""
                st.rerun()
    
    elif not st.session_state.get("search_triggered"):
        # Примеры запросов
        st.info("""
        **🎯 Примеры запросов:**
        - *Машинное обучение в медицине*
        - *Лучшие практики Python*
        - *DevOps инструменты 2024*
        - *Базы данных и оптимизация*
        - *Веб-разработка на React*
        """)

# ==================== ВКЛАДКА 3: МОИ ДОКУМЕНТЫ (если доступен) ====================
if tab3:
    with tab3:
        st.header("📚 Мои документы")
        
        if st.session_state.get("authenticated", True) and st.session_state.get("user_role", "user") in ["admin", "user"]:
            user_docs = st.session_state.rag_agent.get_user_articles(st.session_state.user_id)
            st.session_state.user_documents = user_docs
            
            if not user_docs:
                st.info("""
                **📭 У вас пока нет загруженных документов.**
                
                **📥 Как добавить документы:**
                1. Перейдите в боковую панель
                2. Выберите PDF файл в разделе "Загрузить документы"
                3. Нажмите "Загрузить и обработать"
                4. Документ появится в этой вкладке
                """)
            else:
                # Статистика
                total_chars = sum(len(doc.get("text", "")) for doc in user_docs)
                avg_chars = total_chars // len(user_docs) if user_docs else 0
                total_pages = sum(len(doc.get("pages", [])) for doc in user_docs)
                
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.metric("📄 Всего документов", len(user_docs))
                with stat_col2:
                    st.metric("📑 Всего страниц", total_pages)
                with stat_col3:
                    st.metric("🔤 Всего символов", f"{total_chars:,}")
                with stat_col4:
                    st.metric("📊 Средний размер", f"{avg_chars:,}")
                
                # Быстрый поиск по документам
                st.subheader("🔍 Поиск в ваших документах")
                quick_search = st.text_input(
                    "Введите ключевые слова:",
                    placeholder="Например: Python, алгоритмы, базы данных...",
                    key="quick_search"
                )
                
                if quick_search:
                    search_results = st.session_state.rag_agent.search(
                        quick_search, 
                        st.session_state.user_id, 
                        scope=SearchScope.USER_ONLY, 
                        limit=10
                    )
                    
                    if search_results:
                        grouped_results = group_search_results(search_results)
                        if grouped_results:
                            st.success(f"✅ Найдено {len(grouped_results)} документов с совпадениями")
                            
                            for article_id, article_data in grouped_results.items():
                                metadata = article_data["metadata"]
                                
                                with st.expander(f"📄 {metadata['title']}", expanded=False):
                                    st.markdown(f"**📊 Статистика:**")
                                    st.markdown(f"- Символов: {metadata['total_characters']:,}")
                                    st.markdown(f"- Фрагментов с совпадениями: {metadata['total_chunks']}")
                                    
                                    # Показать контекст первого совпадения
                                    if article_data["chunks"]:
                                        chunk = article_data["chunks"][0]
                                        text = chunk["text"]
                                        if quick_search.lower() in text.lower():
                                            idx = text.lower().find(quick_search.lower())
                                            start = max(0, idx - 100)
                                            end = min(len(text), idx + 100)
                                            context = text[start:end]
                                            if start > 0:
                                                context = "..." + context
                                            if end < len(text):
                                                context = context + "..."
                                            st.markdown(f"**Контекст:** *{context}*")
                    else:
                        st.info("📭 Совпадений не найдено")
                
                # Список всех документов
                st.subheader("📋 Все ваши документы")
                
                docs_data = []
                for doc in user_docs:
                    docs_data.append({
                        "📄 Название": doc.get("title", "Без названия"),
                        "👤 Автор": doc.get("author", "Не указан"),
                        "📅 Дата": doc.get("uploaded_at", doc.get("date", ""))[:10],
                        "🔤 Символов": doc.get("text_length", 0),
                        "📑 Страниц": len(doc.get("pages", [])),
                        "🏷️ Теги": ", ".join(doc.get("tags", [])[:3]) if doc.get("tags") else "",
                    })
                
                if docs_data:
                    df = pd.DataFrame(docs_data)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "📄 Название": st.column_config.TextColumn(width="large"),
                            "🔤 Символов": st.column_config.NumberColumn(format="%d"),
                            "📑 Страниц": st.column_config.NumberColumn(format="%d"),
                        }
                    )
        else:
            st.info("📚 Управление документами доступно только для зарегистрированных пользователей")

# ==================== ВКЛАДКА 4: СТАТИСТИКА ====================
with tab4:
    st.header("📊 Расширенная статистика")
    
    if st.session_state.get("initialized", False) and st.session_state.rag_agent:
        display_enhanced_statistics(st.session_state.rag_agent, st.session_state.user_id)
    else:
        st.error("❌ Система не инициализирована")

# ==================== ВКЛАДКА 5: АДМИН-ПАНЕЛЬ (только для админов) ====================
if tab5 and st.session_state.get("user_role") == "admin" and AUTH_AVAILABLE:
    with tab5:
        st.header("👑 Административная панель")
        
        admin_tabs = st.tabs(["👥 Пользователи", "⚙️ Система", "🔧 Утилиты"])
        
        with admin_tabs[0]:
            st.subheader("👥 Управление пользователями")
            users_data = auth_manager.get_users_dataframe()
            
            if users_data:
                df_users = pd.DataFrame(users_data)
                st.dataframe(df_users, use_container_width=True, hide_index=True)
            else:
                st.info("Нет зарегистрированных пользователей")
        
        with admin_tabs[1]:
            st.subheader("⚙️ Управление системой")
            
            if st.session_state.get("initialized", False):
                rag_stats = st.session_state.rag_agent.get_statistics(st.session_state.username)
                
                # Статус системы
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📚 Всего статей", rag_stats["total_articles"])
                with col2:
                    status = "🟢 Активен" if rag_stats.get("vector_search_enabled", False) else "🔴 Ошибка"
                    st.metric("🔍 Векторный поиск", status)
                with col3:
                    st.metric("🧩 Всего чанков", rag_stats.get("total_chunks", 0))
                
                # Действия администратора
                st.subheader("⚡ Действия")
                
                action_col1, action_col2, action_col3 = st.columns(3)
                with action_col1:
                    if st.button("🔄 Перестроить индекс", use_container_width=True):
                        with st.spinner("Перестраиваю векторный индекс..."):
                            success, message = st.session_state.rag_agent.rebuild_index()
                            if success:
                                st.success(f"✅ {message}")
                            else:
                                st.error(f"❌ {message}")
                
                with action_col2:
                    if st.button("🧹 Очистить кэш", use_container_width=True):
                        st.success("✅ Кэш очищен")
                
                with action_col3:
                    if st.button("📊 Обновить статистику", use_container_width=True):
                        st.rerun()
            else:
                st.error("RAG система не инициализирована")
        
        with admin_tabs[2]:
            st.subheader("🔧 Утилиты")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📁 Директории:**")
                st.code(f"""
                Проект: {PROJECT_ROOT}
                Данные: {PROJECT_ROOT / 'data'}
                Логи: {PROJECT_ROOT / 'app.log'}
                """)
            
            with col2:
                st.markdown("**⚙️ Настройки:**")
                if st.button("Проверить соединения", key="test_connections"):
                    try:
                        status = st.session_state.rag_agent.get_status()
                        st.json(status)
                    except Exception as e:
                        st.error(f"Ошибка: {e}")

# ==================== ФУТЕР ====================
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 0.9em; padding: 20px;">
    <p>🤖 <strong>AI RAG-агент для интеллектуального поиска статей</strong></p>
    <p>🔍 Поиск • ❓ Самопроверка • 📊 Аналитика</p>
    <p>📚 Версия 2.0 | © 2025 DreamTeam AI Agent</p>
</div>
""",
    unsafe_allow_html=True,
)