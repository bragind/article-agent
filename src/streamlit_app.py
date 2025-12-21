# src/streamlit_app.py
"""
Streamlit UI с авторизацией и админ-панелью во вкладке
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any, Optional
import hashlib
import traceback
import json

# Подготовка пути к проекту (чтобы импортировать src)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Импорты RAG из вашего проекта
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
    st.warning(f"Модуль аутентификации недоступен: {e}")

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def group_search_results(search_results: List[Dict]) -> Dict[str, Dict]:
    """Группирует результаты поиска по статьям"""
    grouped_results = {}
    
    for result in search_results:
        article = result["article"]
        
        # Используем article_id или создаем уникальный ключ
        article_id = article.get("id", "")
        if not article_id:
            # Создаем уникальный ключ, если нет ID
            title_key = article.get("title", "").replace(" ", "_")[:50]
            author_key = article.get("author", "").replace(" ", "_")[:20]
            article_id = f"{title_key}_{author_key}"
        
        if article_id not in grouped_results:
            # Новая статья
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
                    "total_chunks": 0,
                    "total_characters": 0
                },
                "chunks": [],
                "total_score": 0.0
            }
        
        # Добавляем чанк
        chunk_data = {
            "text": article.get("text", ""),
            "score": result.get("score", 0.0),
            "characters": len(article.get("text", ""))
        }
        
        # Если есть информация о chunk_index, сохраняем для сортировки
        if "chunk_index" in article:
            chunk_data["chunk_index"] = article.get("chunk_index", 0)
        
        grouped_results[article_id]["chunks"].append(chunk_data)
        grouped_results[article_id]["total_score"] += result.get("score", 0.0)
        
        # Обновляем статистику
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

def search_with_grouping(query: str, user_id: str, scope: SearchScope, limit: int = 10) -> Dict[str, Dict]:
    """Поиск с группировкой результатов по статьям"""
    st.session_state.debug_info = f"Поиск: query='{query}', user_id='{user_id}', scope={scope}, limit={limit}"
    
    if not st.session_state.get("rag_agent"):
        st.error("RAG агент не инициализирован")
        return {}
    
    try:
        # Берем больше результатов для лучшей группировки
        search_results = st.session_state.rag_agent.search(
            query, user_id, scope=scope, limit=limit * 3
        )
        
        st.session_state.debug_info += f"\nНайдено результатов: {len(search_results)}"
        
        # Группируем
        grouped = group_search_results(search_results)
        
        # Сортируем статьи по общей релевантности
        sorted_articles = sorted(
            grouped.items(),
            key=lambda x: x[1]["total_score"],
            reverse=True
        )[:limit]
        
        return dict(sorted_articles)
        
    except Exception as e:
        st.error(f"Ошибка поиска: {e}")
        st.session_state.debug_info += f"\nОшибка: {e}"
        return {}

def load_pdf_file(uploaded_file) -> Dict[str, Any]:
    """Обрабатывает загруженный PDF файл (возвращает article_data или None)"""
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

                pages_data.append(
                    {
                        "page_number": page_num,
                        "text": page_text,
                        "char_count": len(page_text),
                        "word_count": len(page_text.split()),
                    }
                )

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

# ==================== ПРОВЕРКА АВТОРИЗАЦИИ ====================
# Показываем форму входа если не аутентифицированы
if AUTH_AVAILABLE:
    if not streamlit_auth.check_auth():
        st.stop()
else:
    # Если аутентификация недоступна, используем демо-пользователя
    if "user_id" not in st.session_state:
        st.session_state.user_id = hashlib.md5("demo_user".encode()).hexdigest()[:8]
        st.session_state.username = "demo"
        st.session_state.user_role = "user"
        st.session_state.authenticated = True

# ==================== НАСТРОЙКИ СТРАНИЦЫ ====================
st.set_page_config(
    page_title="📚 AI RAG-агент",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== ИНИЦИАЛИЗАЦИЯ СЕССИИ ====================
if "rag_agent" not in st.session_state:
    if RAG_AVAILABLE:
        try:
            with st.spinner("🔄 Инициализация RAG агента..."):
                st.session_state.rag_agent = RAGAgent(data_dir=str(PROJECT_ROOT / "data"))
                st.session_state.initialized = True
                st.success("✅ RAG агент инициализирован")
        except Exception as e:
            st.session_state.initialized = False
            st.error(f"Ошибка инициализации RAGAgent: {e}")
            st.write(traceback.format_exc())
    else:
        st.session_state.initialized = False
        st.error(f"RAGAgent не импортирован: {RAG_IMPORT_ERROR}")

if "user_id" not in st.session_state:
    st.session_state.user_id = hashlib.md5("demo_user".encode()).hexdigest()[:8]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "user_documents" not in st.session_state:
    st.session_state.user_documents = []
    
if "stats_loaded" not in st.session_state:
    st.session_state.stats_loaded = False
    
if "search_scope" not in st.session_state:
    st.session_state.search_scope = "all"
    
if "selected_article" not in st.session_state:
    st.session_state.selected_article = None

if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False

if "debug_info" not in st.session_state:
    st.session_state.debug_info = ""

# ==================== САЙДБАР (УПРОЩЕННЫЙ) ====================
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
    
    # Выбор области поиска (только для авторизованных)
    if st.session_state.get("authenticated", True) and st.session_state.get("user_role", "user") != "guest":
        st.subheader("🔍 Область поиска")
        search_scope = st.selectbox(
            "Где искать:",
            options=["🔎 Везде", "📁 Только мои документы", "🌐 Только Habr"],
            index=(
                0
                if st.session_state.search_scope == "all"
                else 1 if st.session_state.search_scope == "user" else 2
            ),
            key="search_scope_select",
            label_visibility="collapsed"
        )
        
        # Сохраняем выбор
        if search_scope == "🔎 Везде":
            st.session_state.search_scope = "all"
        elif search_scope == "📁 Только мои документы":
            st.session_state.search_scope = "user"
        elif search_scope == "🌐 Только Habr":
            st.session_state.search_scope = "habr"
        
        # Информация о выбранной области
        scope_info = {
            "all": "🔍 Поиск по всем источникам (ваши документы + Habr)",
            "user": "📁 Поиск только по вашим документам",
            "habr": "🌐 Поиск только по статьям Habr",
        }
        st.caption(scope_info.get(st.session_state.search_scope, "🔍 Поиск везде"))
    
    st.markdown("---")
    
    # Загрузка документов (только для авторизованных)
    if st.session_state.get("authenticated", True) and st.session_state.get("user_role", "user") in ["admin", "user"]:
        st.subheader("📁 Загрузить документы")
        uploaded_file = st.file_uploader(
            "Выберите PDF файл", 
            type=["pdf"], 
            help="Максимальный размер: 100MB",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Загрузить и обработать", key="sb_upload"):
                with st.spinner("Обработка PDF..."):
                    article_data = load_pdf_file(uploaded_file)
                    
                    if article_data and article_data.get("has_content"):
                        article_id = st.session_state.rag_agent.add_user_article(
                            article_data, st.session_state.user_id
                        )
                        
                        if article_id:
                            st.success(f"✅ Документ '{article_data['title'][:30]}...' добавлен!")
                            st.session_state.user_documents = st.session_state.rag_agent.get_user_articles(
                                st.session_state.user_id
                            )
                            st.rerun()
                        else:
                            st.error("❌ Ошибка при добавлении документа")
                    else:
                        st.error("⚠️ В документе недостаточно текста или ошибка обработки")
    
    st.markdown("---")
    
    # Настройки поиска
    st.subheader("🎯 Настройки поиска")
    search_limit = st.slider("Результатов на странице", 3, 20, 10, key="sb_search_limit")
    
    st.markdown("---")
    
    # Действия
    st.subheader("⚙️ Действия")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Обновить", key="sb_refresh"):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Очистить чат", key="sb_clear"):
            st.session_state.chat_history = []
            st.session_state.selected_article = None
            st.rerun()
    
    # Отладка
    if st.session_state.get("user_role") == "admin":
        st.markdown("---")
        if st.button("🐛 Показать отладку", key="debug_btn"):
            st.session_state.show_debug = not st.session_state.get("show_debug", False)
            st.rerun()
    
    st.markdown("---")
    st.caption("© 2025 DreamTeam AI Agent")

# ==================== ГЛАВНАЯ СТРАНИЦА ====================
st.title("📚 AI RAG-агент для поиска статей")

# Информация о пользователе
if AUTH_AVAILABLE and st.session_state.get("authenticated"):
    user_role_display = {
        "admin": "👑 Администратор",
        "user": "👤 Пользователь",
        "guest": "👥 Гость"
    }
    role = user_role_display.get(st.session_state.user_role, "👤 Пользователь")
    st.markdown(f"**Привет, {st.session_state.username}!** ({role})")

st.markdown(
    """
    *Интеллектуальный поиск и анализ IT-статей с выбором области поиска*

    **📊 Текущая область поиска:**
    - 🔎 Везде - поиск по всем источникам
    - 📁 Только мои документы - только ваши PDF
    - 🌐 Только Habr - только статьи Habr

    **💡 Измените область поиска в боковой панели слева**
"""
)

# Отображение отладочной информации
if st.session_state.get("show_debug", False) and st.session_state.debug_info:
    with st.expander("🐛 Отладочная информация", expanded=True):
        st.text(st.session_state.debug_info)

st.markdown("---")

# ==================== СОЗДАНИЕ ВКЛАДОК ПО РОЛЯМ ====================
# Определяем какие вкладки показывать в зависимости от роли
user_role = st.session_state.get("user_role", "user")

if user_role == "admin":
    tabs = st.tabs(["🔍 Поиск", "💬 Чат", "📚 Мои документы", "📊 Статистика", "👑 Админ"])
    tab1, tab2, tab3, tab4, tab5 = tabs
elif user_role == "user":
    tabs = st.tabs(["🔍 Поиск", "💬 Чат", "📚 Мои документы", "📊 Статистика"])
    tab1, tab2, tab3, tab4 = tabs
    tab5 = None
else:  # guest
    tabs = st.tabs(["🔍 Поиск", "📊 Статистика"])
    tab1, tab4 = tabs
    tab2, tab3, tab5 = None, None, None

# ==================== ВКЛАДКА 1: ПОИСК ====================
with tab1:
    st.header("🔍 Интеллектуальный поиск")

    # Отображение текущей области поиска
    scope_emoji = {"all": "🔎", "user": "📁", "habr": "🌐"}
    
    if st.session_state.get("authenticated", True) and st.session_state.get("user_role", "user") != "guest":
        scope_info = {
            "all": "🔍 Поиск по всем источникам (ваши документы + Habr)",
            "user": "📁 Поиск только по вашим документам",
            "habr": "🌐 Поиск только по статьям Habr",
        }
        st.info(f"{scope_emoji.get(st.session_state.search_scope, '🔎')} **Ищу: {scope_info.get(st.session_state.search_scope, 'везде')}**")
    else:
        st.info("🔍 **Поиск по всем источникам Habr** (только просмотр для гостей)")

    # Поле поиска
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input(
            "Введите запрос для поиска",
            placeholder="Например: Как работает RAG архитектура?",
            key="search_input",
            label_visibility="collapsed"
        )

    with col2:
        search_button = st.button("🔍 Искать", key="search_button")

    if search_query and (search_button or st.session_state.get("search_triggered")):
        if search_button:
            st.session_state.search_triggered = True
            st.session_state.last_search_query = search_query
            
        with st.spinner("🔍 Ищу..."):
            if st.session_state.get("authenticated", True) and st.session_state.get("user_role", "user") != "guest":
                scope = get_scope_enum(st.session_state.search_scope)
            else:
                scope = SearchScope.HABR_ONLY  # Для гостей только Habr

            if scope:
                # Используем группировку результатов
                grouped_results = search_with_grouping(
                    search_query,
                    st.session_state.user_id,
                    scope=scope,
                    limit=search_limit
                )

                if not grouped_results:
                    st.warning(f"📭 По вашему запросу ничего не найдено")
                    # Предложения по улучшению запроса
                    with st.expander("💡 Советы по поиску", expanded=False):
                        st.markdown("""
                        Попробуйте:
                        1. Использовать более простые ключевые слова
                        2. Проверить правильность написания
                        3. Поискать на английском языке
                        4. Изменить область поиска
                        """)
                else:
                    st.success(f"✅ Найдено {len(grouped_results)} уникальных статей")

                    # Сводка от RAG агента
                    with st.expander("📝 Краткая сводка", expanded=True):
                        try:
                            result = st.session_state.rag_agent.generate_answer(
                                search_query, st.session_state.user_id, scope
                            )
                            st.markdown(result.get("answer", ""))
                        except Exception as e:
                            st.error(f"Ошибка генерации сводки: {e}")

                    # Группированные результаты
                    st.subheader("📚 Найденные материалы")

                    for i, (article_id, article_data) in enumerate(grouped_results.items(), 1):
                        metadata = article_data["metadata"]
                        chunks = article_data["chunks"]
                        
                        # Сортируем чанки по релевантности
                        chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
                        
                        source_type = "📁 Ваш документ" if metadata.get("is_user_document") else "🌐 Статья Habr"
                        
                        with st.container():
                            # Заголовок и метаданные
                            col_header1, col_header2 = st.columns([3, 1])
                            
                            with col_header1:
                                st.markdown(f"**{i}. {metadata['title']}**")
                                st.caption(
                                    f"{source_type} • 👤 {metadata['author']} • "
                                    f"📅 {metadata['date'][:10] if metadata.get('date') else 'Нет даты'}"
                                )
                            
                            with col_header2:
                                # Общая статистика
                                st.metric("Символов", f"{metadata['total_characters']:,}")
                                st.caption(f"Фрагментов: {metadata['total_chunks']}")
                            
                            # Теги
                            tags = metadata.get("tags", [])
                            if tags:
                                if isinstance(tags, str):
                                    st.markdown(f"**Теги:** `{tags}`")
                                elif isinstance(tags, list):
                                    tag_text = ", ".join([f"`{tag}`" for tag in tags[:5]])
                                    if len(tags) > 5:
                                        tag_text += f" и еще {len(tags) - 5}"
                                    st.markdown(f"**Теги:** {tag_text}")
                            
                            # Аккордеон с релевантными цитатами
                            with st.expander(f"📖 Показать релевантные фрагменты ({len(chunks)} найдено)", expanded=False):
                                for chunk_idx, chunk in enumerate(chunks, 1):
                                    st.markdown(f"**Фрагмент {chunk_idx}**")
                                    
                                    # Показываем контекст
                                    preview = chunk["text"]
                                    if len(preview) > 300:
                                        preview = preview[:300] + "..."
                                    
                                    st.markdown(f"> *{preview}*")
                                    
                                    # Информация о фрагменте
                                    col_info1, col_info2 = st.columns(2)
                                    with col_info1:
                                        st.caption(f"📄 {chunk['characters']:,} символов")
                                    with col_info2:
                                        if chunk.get("chunk_index") is not None:
                                            st.caption(f"# {chunk['chunk_index'] + 1}")
                                        # Индикатор релевантности
                                        relevance = get_relevance_badge(chunk.get("score", 0))
                                        st.caption(f"Релевантность: {relevance}")
                                    
                                    # Разделитель между фрагментами
                                    if chunk_idx < len(chunks):
                                        st.divider()
                            
                            # Кнопки действий
                            col_btn1, col_btn2, col_btn3 = st.columns(3)
                            with col_btn1:
                                if st.button("📋 Копировать ссылку", key=f"copy_{article_id}"):
                                    st.toast(f"Ссылка скопирована: {metadata['url']}")
                            
                            with col_btn2:
                                if metadata['url'] and not metadata['url'].startswith("file://"):
                                    st.link_button("🔗 Открыть статью", metadata['url'])
                                else:
                                    st.button("📄 Документ", key=f"doc_{article_id}", disabled=True)
                            
                            with col_btn3:
                                if st.button("💬 Спросить об этом", key=f"ask_{article_id}"):
                                    # Сохраняем статью для чата
                                    st.session_state.selected_article = metadata['title']
                                    st.session_state.chat_history.append({
                                        "role": "user",
                                        "content": f"Расскажи подробнее о статье: {metadata['title']}"
                                    })
                                    st.rerun()
                            
                            st.divider()
    
    # Кнопка для нового поиска
    if st.session_state.get("search_triggered"):
        if st.button("🔄 Новый поиск", key="new_search"):
            st.session_state.search_triggered = False
            st.session_state.last_search_query = ""
            st.rerun()

# ==================== ВКЛАДКА 2: ЧАТ (если доступен) ====================
if tab2:
    with tab2:
        st.header("💬 Чат с AI-агентом")

        if st.session_state.get("authenticated", True) and st.session_state.get("user_role", "user") != "guest":
            chat_scope = st.selectbox(
                "Область поиска для чата:",
                options=["🔎 Везде", "📁 Только мои документы", "🌐 Только Habr"],
                key="chat_scope",
            )

            chat_scope_enum = SearchScope.ALL
            if chat_scope == "📁 Только мои документы":
                chat_scope_enum = SearchScope.USER_ONLY
            elif chat_scope == "🌐 Только Habr":
                chat_scope_enum = SearchScope.HABR_ONLY
        else:
            chat_scope_enum = SearchScope.HABR_ONLY
            st.info("💬 Чат доступен только для зарегистрированных пользователей")

        # Показать выбранную статью для контекста
        if st.session_state.selected_article:
            st.info(f"📄 Контекст чата: {st.session_state.selected_article}")

        # История чата
        for message in st.session_state.chat_history:
            with st.chat_message(message.get("role", "user")):
                st.markdown(message.get("content", ""))
                if message.get("sources"):
                    with st.expander("📚 Показать источники"):
                        for source in message.get("sources", []):
                            st.markdown(f"• {source.get('title', '')}")

        chat_input = st.chat_input("Задайте вопрос AI-агенту...")

        if chat_input:
            st.session_state.chat_history.append({"role":"user","content":chat_input,"timestamp":datetime.now().isoformat()})
            with st.chat_message("user"):
                st.markdown(chat_input)

            with st.spinner(f"🤔 AI-агент думает..."):
                try:
                    result = st.session_state.rag_agent.generate_answer(chat_input, st.session_state.user_id, chat_scope_enum)
                except Exception as e:
                    st.error(f"Ошибка генерации ответа: {e}")
                    result = {"answer":"", "sources":[], "questions":[]}

                st.session_state.chat_history.append({"role":"assistant","content":result.get("answer",""),"sources":result.get("sources",[]),"questions":result.get("questions",[]),"timestamp":datetime.now().isoformat()})

                with st.chat_message("assistant"):
                    st.markdown(result.get("answer",""))
                    if result.get("sources"):
                        with st.expander("📚 Показать источники", expanded=False):
                            for i, source in enumerate(result.get("sources", []), 1):
                                icon = "📁" if source.get("is_user_document") else "🌐"
                                st.markdown(f"{i}. {icon} **{source.get('title','')}**")

                if result.get("questions"):
                    with st.expander("❓ Вопросы для самопроверки", expanded=True):
                        for question in result.get("questions", []):
                            st.markdown(f"• {question}")

# ==================== ВКЛАДКА 3: МОИ ДОКУМЕНТЫ (если доступен) ====================
if tab3:
    with tab3:
        st.header("📚 Мои документы")

        if st.session_state.get("authenticated", True) and st.session_state.get("user_role", "user") in ["admin", "user"]:
            user_docs = st.session_state.rag_agent.get_user_articles(st.session_state.user_id)
            st.session_state.user_documents = user_docs

            if not user_docs:
                st.info(
                    """
                📭 **У вас пока нет загруженных документов.**

                **Как добавить документы:**
                1. Перейдите в боковую панель
                2. Выберите PDF файл в разделе "Загрузить документы"
                3. Нажмите "Загрузить и обработать"
                4. Документ появится в этой вкладке
                """
                )
            else:
                total_chars = sum(len(doc.get("text", "")) for doc in user_docs)
                avg_chars = total_chars // len(user_docs) if user_docs else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего документов", len(user_docs))
                with col2:
                    st.metric("Всего символов", f"{total_chars:,}")
                with col3:
                    st.metric("Средний размер", f"{avg_chars:,}")

                st.subheader("🔍 Быстрый поиск по документам")

                quick_search = st.text_input("Поиск в ваших документам", placeholder="Введите ключевые слова...", key="quick_search")

                if quick_search:
                    search_results = st.session_state.rag_agent.search(quick_search, st.session_state.user_id, scope=SearchScope.USER_ONLY, limit=10)
                    if search_results:
                        grouped_results = group_search_results(search_results)
                        if grouped_results:
                            st.success(f"✅ Найдено {len(grouped_results)} документов с совпадениями")
                            for article_id, article_data in grouped_results.items():
                                metadata = article_data["metadata"]
                                with st.expander(f"📄 {metadata['title']}"):
                                    st.markdown(f"**Символов:** {metadata['total_characters']:,}")
                                    st.markdown(f"**Фрагментов с совпадениями:** {metadata['total_chunks']}")
                                    
                                    # Показываем первый найденный фрагмент
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
                                            st.markdown(f"**Контекст:** {context}")
                    else:
                        st.info("📭 Совпадений не найдено")

                st.subheader("📋 Список документов")
                docs_data = []
                for doc in user_docs:
                    docs_data.append(
                        {
                            "Название": doc.get("title", "Без названия"),
                            "Автор": doc.get("author", "Не указан"),
                            "Дата": doc.get("uploaded_at", doc.get("date", ""))[:10],
                            "Символов": doc.get("text_length", 0),
                            "Страниц": len(doc.get("pages", [])),
                            "Теги": ", ".join(doc.get("tags", [])[:3]),
                        }
                    )

                if docs_data:
                    df = pd.DataFrame(docs_data)
                    # Убедимся, что все данные в правильном формате
                    df["Символов"] = df["Символов"].astype(str)
                    
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Название": st.column_config.TextColumn(width="large"),
                            "Символов": st.column_config.TextColumn(width="small"),
                            "Теги": st.column_config.TextColumn(width="medium"),
                        }
                    )
        else:
            st.info("📚 Управление документами доступно только для зарегистрированных пользователей")

# ==================== ВКЛАДКА 4: СТАТИСТИКА ====================
with tab4:
    st.header("📊 Статистика системы")

    if st.session_state.get("initialized", False) and st.session_state.rag_agent:
        try:
            stats = st.session_state.rag_agent.get_statistics(st.session_state.user_id)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Всего документов", stats["total_articles"])
            with col2:
                st.metric("Статей Habr", stats["habr_articles"])
            with col3:
                st.metric("Пользовательских", stats["user_articles"])
            with col4:
                st.metric("Ваших документов", stats["current_user_articles"])

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Распределение по источникам")
                sources_data = {"Habr": stats["habr_articles"], "Пользовательские": stats["user_articles"]}
                fig = px.pie(values=list(sources_data.values()), names=list(sources_data.keys()), title="Источники документов", color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("👤 Ваши документы")
                if stats["current_user_articles"] > 0:
                    fig = px.bar(x=["Ваши документы"], y=[stats["current_user_articles"]], title="Количество ваших документов", labels={"x": "", "y": "Количество"})
                    fig.update_traces(marker_color="green")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("У вас пока нет документов")

            st.subheader("ℹ️ Информация о системе")
            sys_info = {
                "ID пользователя": st.session_state.user_id,
                "Область поиска по умолчанию": scope_info.get(st.session_state.search_scope, "🔎 Везде"),
                "Обновлено": stats.get("last_update", "")[:19],
            }
            for key, value in sys_info.items():
                st.markdown(f"**{key}:** {value}")
        except Exception as e:
            st.error(f"Ошибка получения статистики: {e}")
    else:
        st.error("Система не инициализирована")

# ==================== ВКЛАДКА 5: АДМИН-ПАНЕЛЬ (только для админов) ====================
if tab5 and st.session_state.get("user_role") == "admin" and AUTH_AVAILABLE:
    with tab5:
        st.header("👑 Административная панель")
        
        # Создаем вкладки внутри админ-панели
        admin_tabs = st.tabs(["👥 Пользователи", "⚙️ Система", "📊 Логи"])
        
        # Вкладка 1: Управление пользователями
        with admin_tabs[0]:
            st.subheader("Управление пользователями")
            
            # Статистика пользователей
            users_data = auth_manager.get_users_dataframe()
            total_users = len(users_data)
            active_users = sum(1 for user in users_data if "🟢" in user.get("📊 Статус", ""))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Всего пользователей", total_users)
            with col2:
                st.metric("Активных", active_users)
            
            # Таблица пользователей с четким форматированием
            st.subheader("Список пользователей")
            
            if users_data:
                # Создаем DataFrame
                df_users = pd.DataFrame(users_data)
                
                # Отображаем таблицу без редактирования
                st.dataframe(
                    df_users,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Управление пользователями
                st.subheader("Управление")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Создание нового пользователя
                    with st.expander("👤 Создать пользователя", expanded=False):
                        with st.form("create_user_form"):
                            new_username = st.text_input("Имя пользователя*")
                            new_password = st.text_input("Пароль*", type="password")
                            new_role = st.selectbox("Роль*", ["user", "admin", "guest"])
                            confirm_password = st.text_input("Подтвердите пароль*", type="password")
                            
                            if st.form_submit_button("Создать"):
                                if new_password != confirm_password:
                                    st.error("Пароли не совпадают")
                                elif auth_manager.create_user(new_username, new_password, new_role):
                                    st.success(f"✅ Пользователь {new_username} создан")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Пользователь {new_username} уже существует")
                
                with col2:
                    # Удаление пользователя
                    with st.expander("🗑️ Удалить пользователя", expanded=False):
                        user_to_delete = st.selectbox(
                            "Выберите пользователя",
                            options=[user["👤 Имя пользователя"] for user in users_data if user["👤 Имя пользователя"] != st.session_state.username],
                            key="delete_user_select"
                        )
                        
                        if st.button("Удалить", key="delete_user_btn"):
                            if auth_manager.delete_user(user_to_delete):
                                st.success(f"✅ Пользователь {user_to_delete} удален")
                                st.rerun()
                            else:
                                st.error("❌ Ошибка при удалении")
            else:
                st.info("📭 Нет зарегистрированных пользователей")
        
        # Вкладка 2: Управление системой
        with admin_tabs[1]:
            st.subheader("Управление системой")
            
            if st.session_state.get("initialized", False):
                # Статистика RAG системы
                rag_stats = st.session_state.rag_agent.get_statistics(st.session_state.username)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего статей", rag_stats["total_articles"])
                with col2:
                    st.metric("Чанков в индексе", rag_stats.get("total_chunks", 0))
                with col3:
                    status = "🟢 Активен" if rag_stats.get("vector_search_enabled", False) else "🔴 Ошибка"
                    st.metric("Векторный поиск", status)
                
                # Управление индексом
                st.subheader("Управление векторным индексом")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🔄 Перестроить индекс"):
                        with st.spinner("Перестраиваю векторный индекс..."):
                            success, message = st.session_state.rag_agent.rebuild_index()
                            if success:
                                st.success(f"✅ {message}")
                            else:
                                st.error(f"❌ {message}")
                
                with col2:
                    if st.button("🧹 Очистить кэш"):
                        st.session_state.chat_history = []
                        st.success("✅ Кэш чата очищен")
                
                with col3:
                    if st.button("📊 Обновить статистику"):
                        st.rerun()
                
                # Информация о системе
                st.subheader("Информация о системе")
                
                sys_status = st.session_state.rag_agent.get_status()
                
                # Создаем DataFrame с правильными типами данных
                status_data = [
                    {"Параметр": "Статей загружено", "Значение": str(sys_status["articles_loaded"])},
                    {"Параметр": "Статей Habr", "Значение": str(sys_status["habr_articles"])},
                    {"Параметр": "Пользовательских статей", "Значение": str(sys_status["user_articles"])},
                    {"Параметр": "Векторный поиск", "Значение": "✅ Доступен" if sys_status["vector_search_available"] else "❌ Недоступен"},
                    {"Параметр": "LLM доступен", "Значение": "✅ Доступен" if sys_status["llm_available"] else "❌ Недоступен"},
                    {"Параметр": "Готовность системы", "Значение": "✅ Готова" if sys_status["ready"] else "❌ Ошибка"},
                ]
                
                status_df = pd.DataFrame(status_data)
                
                st.dataframe(
                    status_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Параметр": st.column_config.TextColumn(width="medium"),
                        "Значение": st.column_config.TextColumn(width="medium")
                    }
                )
                
                # Ошибки инициализации
                if sys_status.get("errors"):
                    with st.expander("⚠️ Ошибки инициализации", expanded=False):
                        for error in sys_status["errors"]:
                            st.error(f"• {error}")
            else:
                st.error("RAG система не инициализирована")
        
        # Вкладка 3: Логи системы
        with admin_tabs[2]:
            st.subheader("Логи системы")
            
            # Выбор файла логов
            log_files = []
            for log_file in ["bot.log", "app.log", "auth.log", "rag.log"]:
                if Path(log_file).exists():
                    log_files.append(log_file)
            
            if log_files:
                selected_log = st.selectbox("Выберите файл логов", log_files)
                
                # Параметры отображения
                col1, col2 = st.columns(2)
                with col1:
                    tail_lines = st.number_input("Количество строк (с конца)", min_value=10, max_value=10000, value=100)
                with col2:
                    if st.button("🔄 Обновить логи"):
                        st.rerun()
                
                # Чтение и отображение логов
                try:
                    with open(selected_log, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Берем последние N строк
                    lines_to_show = lines[-tail_lines:] if len(lines) > tail_lines else lines
                    
                    # Создаем красивый вывод
                    st.subheader(f"Последние {len(lines_to_show)} строк из {selected_log}")
                    
                    # Отображаем с подсветкой
                    log_text = "".join(lines_to_show)
                    
                    # Отображаем в текстовом поле с возможностью прокрутки
                    st.text_area(
                        "Содержимое логов",
                        value=log_text,
                        height=400,
                        key=f"log_viewer_{selected_log}"
                    )
                    
                    # Кнопки действий
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            label="📥 Скачать логи",
                            data=log_text,
                            file_name=f"{selected_log}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        if st.button("🧹 Очистить логи"):
                            with open(selected_log, 'w', encoding='utf-8') as f:
                                f.write("")
                            st.success(f"✅ Логи файла {selected_log} очищены")
                            st.rerun()
                    
                    with col3:
                        if st.button("🔍 Поиск в логах"):
                            st.session_state.log_search = True
                            st.rerun()
                    
                    # Поиск по логам
                    if st.session_state.get("log_search", False):
                        search_term = st.text_input("Поиск в логах", placeholder="Введите текст для поиска...")
                        if search_term:
                            matches = [line for line in lines if search_term.lower() in line.lower()]
                            if matches:
                                st.success(f"✅ Найдено {len(matches)} совпадений")
                                with st.expander("📋 Показать совпадения", expanded=True):
                                    for match in matches[-10:]:  # Показываем последние 10 совпадений
                                        st.text(match)
                            else:
                                st.info("📭 Совпадений не найдено")
                        
                        if st.button("✖️ Закрыть поиск"):
                            del st.session_state.log_search
                            st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Ошибка чтения логов: {e}")
            else:
                st.info("📭 Файлы логов не найдены")

# ==================== ФУТЕР ====================
st.markdown("---")
st.markdown(
    """
<div style="text-align: center">
    <p>📚 <strong>AI RAG-агент dreamteam___.ru</strong> | Версия 1.0.0</p>
    <p>🤖 Интеллектуальный поиск и анализ IT-статей</p>
    <p>🔐 Система аутентификации и авторизации</p>
    <p>📧 Поддержка: support@dreamteam____.ru</p>
</div>
""",
    unsafe_allow_html=True,
)