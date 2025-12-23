"""
UI компоненты приложения
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

from . import helpers
from . import constants

logger = logging.getLogger(__name__)


class UIComponents:
    """Класс UI компонентов"""
    
    @staticmethod
    def create_sidebar(auth_manager, rag_agent):
        """Создание сайдбара"""
        with st.sidebar:
            # Профиль пользователя
            auth_manager.show_user_profile()
            
            st.title("🤖 AI RAG-агент")
            st.markdown("---")
            
            # Быстрые действия
            st.subheader("⚡ Быстрые действия")
            
            if st.button("🔄 Обновить систему", key="sb_refresh", use_container_width=True):
                st.rerun()
            
            if st.button("🗑️ Очистить фильтры", key="sb_clear_filters", use_container_width=True):
                UIComponents.clear_search_filters()
                st.rerun()
            
            # Информация о системе
            st.markdown("---")
            st.subheader("ℹ️ О системе")
            
            if rag_agent:
                stats = rag_agent.get_statistics(auth_manager.get_user_id())
                st.caption(f"📚 Статей: {stats.get('total_articles', 0)}")
                st.caption(f"📁 Ваших документов: {stats.get('current_user_articles', 0)}")
                st.caption(f"🔍 Поиск: {'✅ Векторный' if stats.get('vector_search_enabled') else '📝 Текстовый'}")
            
            st.markdown("---")
            st.caption("© 2025 AI RAG-агент | DreamTeam")
    
    @staticmethod
    def clear_search_filters():
        """Очистка фильтров поиска"""
        filter_keys = [
            "search_tags", "search_author", "search_date_from", 
            "search_date_to", "search_triggered", "search_query"
        ]
        
        for key in filter_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        # Восстанавливаем дефолтные значения
        st.session_state.search_tags = []
        st.session_state.search_author = None
        st.session_state.search_date_from = None
        st.session_state.search_date_to = None
        st.session_state.search_triggered = False
    
    @staticmethod
    def display_header(auth_manager):
        """Отображение заголовка"""
        st.title("🔍 AI RAG-агент для поиска статей")
        
        # Приветствие
        if auth_manager.is_authenticated():
            from .constants import USER_ROLES
            role = USER_ROLES.get(auth_manager.get_user_role(), "👤 Пользователь")
            username = auth_manager.get_username()
            
            if username != "guest":
                st.markdown(f"### Привет, **{username}!** ({role})")
            else:
                st.markdown(f"### Гостевой доступ ({role})")
        
        st.markdown(
            """
            *Интеллектуальный поиск и анализ IT-статей с вопросами для самопроверки*
            
            **✨ Возможности:**
            - 🔍 **Умный поиск** по статьям Habr и вашим документам
            - 🏷️ **Фильтрация** по теги, автору, дате публикации
            - ❓ **Вопросы для самопроверки** по каждой статье
            - 🔍 **Поиск похожих статей** по тематике
            - 📊 **Расширенная статистика** и аналитика
            - 📤 **Загрузка PDF-документов** в вашу коллекцию
            """
        )
    
    @staticmethod
    def create_tabs(auth_manager):
        """Создание вкладок по ролям"""
        user_role = auth_manager.get_user_role()
        
        if user_role == "admin":
            tabs = st.tabs(["🔍 Поиск", "📚 Мои документы", "📊 Статистика", "👑 Админ"])
            return tabs[0], tabs[1], tabs[2], tabs[3]
        elif user_role == "user":
            tabs = st.tabs(["🔍 Поиск", "📚 Мои документы", "📊 Статистика"])
            return tabs[0], tabs[1], tabs[2], None
        else:  # guest
            tabs = st.tabs(["🔍 Поиск", "📊 Статистика"])
            return tabs[0], None, tabs[1], None
    
    @staticmethod
    def display_similar_articles(article_id: str, article_metadata: Dict, user_id: str, rag_agent):
        """Отображение похожих статей для конкретной статьи"""
        if not rag_agent:
            st.error("RAG агент не инициализирован")
            return
        
        try:
            similar_articles = rag_agent.get_similar_articles(
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
                        
                        from .constants import SOURCE_TYPES
                        source_type = SOURCE_TYPES.get("user") if similar_article.get("is_user_document") else SOURCE_TYPES.get("habr")
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
    
    @staticmethod
    def display_questions_for_article(article: Dict, rag_agent):
        """Отображение вопросов для самопроверки по статье"""
        if not rag_agent:
            st.error("RAG агент не инициализирован")
            return
        
        with st.spinner("🤔 Генерирую вопросы для самопроверки..."):
            try:
                # Используем текст статьи для генерации вопросов
                article_text = article.get("text", "")[:2000]
                
                if article_text:
                    # Проверяем, доступен ли LLMClient
                    if hasattr(rag_agent, 'llm_client') and rag_agent.llm_client:
                        questions = rag_agent.llm_client.generate_questions(
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
    
    @staticmethod
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
                relevance = helpers.get_relevance_badge(chunk.get("score", 0))
                st.caption(f"**Релевантность:** {relevance}")
            
            if chunk_idx < min(3, len(chunks)):
                st.divider()
    
    @staticmethod
    def display_article_results(grouped_results: Dict, user_id: str, rag_agent):
        """Отображает результаты поиска статей в новом формате"""
        for i, (article_id, article_data) in enumerate(grouped_results.items(), 1):
            metadata = article_data["metadata"]
            chunks = article_data["chunks"]
            
            # Сортируем чанки по релевантности
            chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            from .constants import SOURCE_TYPES
            source_type = SOURCE_TYPES.get("user") if metadata.get("is_user_document") else SOURCE_TYPES.get("habr")
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
                    UIComponents.display_questions_for_article({
                        "title": metadata['title'],
                        "text": chunks[0]["text"] if chunks else "",
                        "author": metadata['author'],
                        "date": metadata['date']
                    }, rag_agent)
                
                # ШИРОКАЯ КНОПКА "ПОХОЖИЕ" с разворачивающимся содержимым
                with st.expander("🔍 Похожие статьи", expanded=False):
                    UIComponents.display_similar_articles(article_id, metadata, user_id, rag_agent)
                
                # ШИРОКАЯ КНОПКА "ПОДРОБНЕЕ" с разворачивающимся содержимым
                with st.expander("📖 Подробная информация", expanded=False):
                    UIComponents.display_article_details(metadata, chunks)
                
                # Разделитель между статьями
                st.markdown("<hr style='margin: 30px 0; border: 1px solid #ddd;'>", unsafe_allow_html=True)