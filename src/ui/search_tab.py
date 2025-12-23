"""
Вкладка поиска
"""

import streamlit as st
import logging
from typing import Dict, Optional, List
from datetime import datetime

from . import helpers
from . import components
from .constants import SEARCH_SCOPE_MAPPING

logger = logging.getLogger(__name__)


class SearchTab:
    """Класс вкладки поиска"""
    
    def __init__(self, auth_manager, rag_agent):
        self.auth_manager = auth_manager
        self.rag_agent = rag_agent
        
        # Инициализация состояния
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
    
    def search_with_filters(self, query: str, user_id: str, scope, limit: int, 
                           tags: Optional[List[str]] = None, author: Optional[str] = None, 
                           date_from: Optional[str] = None, date_to: Optional[str] = None) -> Dict:
        """Поиск с фильтрами - теперь использует rag_agent.search() вместо search_with_filters()"""
        if not self.rag_agent:
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
            
            # ИСПРАВЛЕНИЕ: используем rag_agent.search() вместо rag_agent.search_with_filters()
            search_results = self.rag_agent.search(
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
            grouped = helpers.group_search_results(search_results)
            
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
                search_results = self.rag_agent.search(
                    query, user_id, scope, limit * 3
                )
                grouped = helpers.group_search_results(search_results)
                sorted_articles = sorted(
                    grouped.items(),
                    key=lambda x: x[1]["total_score"],
                    reverse=True
                )[:limit]
                return dict(sorted_articles)
            except Exception as e2:
                logger.error(f"И обычный поиск не работает: {e2}")
                return {}
    
    def display_search_parameters(self):
        """Отображение параметров поиска"""
        with st.container():
            st.subheader("⚙️ Параметры поиска")
            
            # Область поиска и количество результатов
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🌍 Область поиска**")
                
                # Определяем доступные опции в зависимости от роли пользователя
                user_role = self.auth_manager.get_user_role()
                
                if user_role == "guest":
                    # Гости могут искать только в Habr
                    st.info("👥 **Гостевой доступ:** поиск только в Habr")
                    st.session_state.search_scope = "habr"
                    st.markdown("🔍 **Доступно:** Только статьи Habr")
                elif user_role in ["user", "admin"]:
                    # Зарегистрированные пользователи и админы могут искать везде
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
                    else:  # "🌐 Только Habr"
                        st.session_state.search_scope = "habr"
                else:
                    # По умолчанию - только Habr
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
            
            # ФИЛЬТРЫ ПОИСКА (только для авторизованных пользователей)
            if self.auth_manager.is_authenticated() and self.auth_manager.get_user_role() != "guest":
                self.display_search_filters()
            else:
                st.info("🔒 Фильтры доступны только для зарегистрированных пользователей")
            
            return search_limit
    
    def display_search_filters(self):
        """Отображение фильтров поиска для авторизованных пользователей"""
        st.markdown("**🎯 Дополнительные фильтры**")
        
        # Получаем доступные теги и авторов
        all_tags = []
        all_authors = []
        if self.rag_agent:
            try:
                all_tags = self.rag_agent.get_all_tags(self.auth_manager.get_user_id())
                all_authors = self.rag_agent.get_all_authors(self.auth_manager.get_user_id())
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
                        if selected_author and selected_author != "Все авторы":
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
        self.display_active_filters()
        
        st.divider()
    
    def display_active_filters(self):
        """Отображение активных фильтров"""
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
    
    def display_search_input(self, search_limit):
        """Отображение поля поиска"""
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
        
        return search_query, search_button
    
    def display_search_results(self, search_query, search_button, search_limit):
        """Отображение результатов поиска"""
        if search_query and (search_button or st.session_state.get("search_triggered")):
            if search_button:
                st.session_state.search_triggered = True
                st.session_state.last_search_query = search_query
            
            # Определяем область поиска на основе роли пользователя
            user_role = self.auth_manager.get_user_role()
            
            if user_role == "guest":
                # Гости могут искать только в Habr
                from src.core.search_engine import SearchScope
                scope = SearchScope.HABR_ONLY
            else:
                # Зарегистрированные пользователи используют выбранную область
                scope = helpers.get_scope_enum(st.session_state.get("search_scope", "all"))
            
            if scope:
                with st.spinner(f"🔍 Ищу по запросу: '{search_query}'..."):
                    # Получаем ID пользователя для поиска
                    user_id = self.auth_manager.get_user_id()
                    
                    grouped_results = self.search_with_filters(
                        search_query,
                        user_id,
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
                                result = self.rag_agent.generate_answer(
                                    search_query, user_id, scope
                                )
                                st.markdown(result.get("answer", ""))
                                
                                # Показываем источники
                                if result.get("sources"):
                                    st.markdown("**📚 Использованные источники:**")
                                    for i, source in enumerate(result["sources"][:3], 1):
                                        from .constants import SOURCE_TYPES
                                        icon = SOURCE_TYPES.get("user") if source.get("is_user_document") else SOURCE_TYPES.get("habr")
                                        st.markdown(f"{i}. {icon} **{source['title']}** - {source['author']}")
                            except Exception as e:
                                logger.error(f"Ошибка генерации сводки: {e}")
                                st.error(f"Ошибка генерации сводки: {str(e)[:100]}")
                        
                        st.success(f"✅ Найдено {len(grouped_results)} статей")
                        
                        # Отображение результатов
                        components.UIComponents.display_article_results(
                            grouped_results, 
                            user_id, 
                            self.rag_agent
                        )
            
            # Кнопка нового поиска
            if st.session_state.get("search_triggered"):
                if st.button("🔄 Новый поиск", key="new_search", type="secondary"):
                    st.session_state.search_triggered = False
                    st.session_state.last_search_query = ""
                    st.rerun()
        
        elif not st.session_state.get("search_triggered"):
            # Примеры запросов
            user_role = self.auth_manager.get_user_role()
            
            if user_role == "guest":
                st.info("""
                **👥 Гостевой доступ**
                
                **🎯 Примеры запросов (только статьи Habr):**
                - *Машинное обучение в медицине*
                - *Лучшие практики Python*
                - *DevOps инструменты 2024*
                - *Базы данных и оптимизация*
                - *Веб-разработка на React*
                
                **🔒 Для доступа к дополнительным функциям:**
                - 📁 Загрузка документов
                - 🏷️ Расширенные фильтры
                - 📊 Персональная статистика
                
                **Войдите в систему или зарегистрируйтесь!**
                """)
            else:
                st.info("""
                **🎯 Примеры запросов:**
                - *Машинное обучение в медицине*
                - *Лучшие практики Python*
                - *DevOps инструменты 2024*
                - *Базы данных и оптимизация*
                - *Веб-разработка на React*
                """)
    
    def render(self):
        """Рендер вкладки поиска"""
        st.header("🔍 Поиск статей")
        
        # Проверяем, инициализирован ли RAG агент
        if not self.rag_agent:
            st.error("❌ Система поиска не инициализирована")
            st.info("Попробуйте перезагрузить страницу или обратитесь к администратору")
            return
        
        # Проверяем готовность системы
        if not self.rag_agent.is_ready():
            st.warning("⚠️ Система поиска не готова к работе")
            status = self.rag_agent.get_statistics(self.auth_manager.get_user_id())
            
            with st.expander("Техническая информация", expanded=False):
                st.json(status)
            
            if st.button("🔄 Повторить инициализацию", key="retry_init"):
                st.rerun()
            
            return
        
        # Параметры поиска
        search_limit = self.display_search_parameters()
        
        # Поле поиска
        search_query, search_button = self.display_search_input(search_limit)
        
        # Результаты поиска
        self.display_search_results(search_query, search_button, search_limit)