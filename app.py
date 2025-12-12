# app.py
import streamlit as st
import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import hashlib

# Добавляем путь к src
import sys
from pathlib import Path

# Добавляем src в Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Импортируем напрямую
from rag import RAGAgent, SearchScope

# ==================== НАСТРОЙКИ СТРАНИЦЫ ====================
st.set_page_config(
    page_title="📚 AI RAG-агент",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ИНИЦИАЛИЗАЦИЯ СЕССИИ ====================
if 'rag_agent' not in st.session_state:
    if RAGAgent:
        try:
            st.session_state.rag_agent = RAGAgent(data_dir="data")
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Ошибка инициализации RAGAgent: {e}")
            st.session_state.initialized = False
    else:
        st.session_state.initialized = False

    st.session_state.chat_history = []
    st.session_state.user_documents = []
    st.session_state.stats_loaded = False
    st.session_state.search_scope = "all"  # По умолчанию: ищем везде

if 'user_id' not in st.session_state:
    # Для демо используем хеш
    st.session_state.user_id = hashlib.md5("demo_user".encode()).hexdigest()[:8]


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
def load_pdf_file(uploaded_file) -> Dict[str, Any]:
    """Обрабатывает загруженный PDF файл"""
    try:
        import pdfplumber
        from io import BytesIO

        with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
            full_text = ""
            pages_data = []

            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                full_text += f"\n\n--- Страница {page_num} ---\n{page_text}"

                pages_data.append({
                    "page_number": page_num,
                    "text": page_text,
                    "char_count": len(page_text),
                    "word_count": len(page_text.split())
                })

            article_data = {
                "title": uploaded_file.name.replace(".pdf", "").replace("_", " ").title(),
                "author": "",
                "date": datetime.now().isoformat(),
                "text": full_text.strip(),
                "tags": ["pdf", "document", "uploaded"],
                "url": f"file://{uploaded_file.name}",
                "source": "User Upload",
                "views": 0,
                "rating": 0,
                "parsed_at": datetime.now().isoformat(),
                "text_length": len(full_text.strip()),
                "has_content": len(full_text.strip()) > 100,
                "pages": pages_data,
                "uploaded_by": st.session_state.user_id
            }

            return article_data

    except Exception as e:
        st.error(f"Ошибка обработки PDF: {e}")
        return None


def get_scope_enum(scope_str: str):
    """Конвертирует строку в SearchScope"""
    if not SearchScope:
        return None

    if scope_str == "all":
        return SearchScope.ALL
    elif scope_str == "user":
        return SearchScope.USER_ONLY
    elif scope_str == "habr":
        return SearchScope.HABR_ONLY
    return SearchScope.ALL


# ==================== САЙДБАР ====================
with st.sidebar:
    st.title("🤖 AI RAG-агент")
    st.markdown("---")

    if not st.session_state.get('initialized', False):
        st.error("RAG агент не инициализирован")
        if st.button("Попробовать снова"):
            try:
                st.session_state.rag_agent = RAGAgent(data_dir="data")
                st.session_state.initialized = True
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")
        st.stop()

    # Выбор области поиска
    st.subheader("🔍 Область поиска")
    search_scope = st.selectbox(
        "Где искать:",
        options=["🔎 Везде", "📁 Только мои документы", "🌐 Только Habr"],
        index=0 if st.session_state.search_scope == "all" else 1 if st.session_state.search_scope == "user" else 2,
        key="search_scope_select"
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
        "habr": "🌐 Поиск только по статьям Habr"
    }
    st.info(scope_info[st.session_state.search_scope])

    st.markdown("---")

    st.subheader("👤 Профиль")
    st.info(f"ID: `{st.session_state.user_id}`")

    if st.button("📊 Обновить статистику", use_container_width=True):
        st.session_state.stats_loaded = True

    if st.session_state.stats_loaded:
        stats = st.session_state.rag_agent.get_statistics(st.session_state.user_id)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Всего документов", f"{stats['total_articles']:,}")
        with col2:
            st.metric("Ваших документов", f"{stats['current_user_articles']:,}")

    st.markdown("---")

    # Загрузка документов
    st.subheader("📁 Загрузить документы")
    uploaded_file = st.file_uploader(
        "Выберите PDF файл",
        type=['pdf'],
        help="Максимальный размер: 100MB"
    )

    if uploaded_file is not None:
        if st.button("📤 Загрузить и обработать", use_container_width=True):
            with st.spinner("Обработка PDF..."):
                article_data = load_pdf_file(uploaded_file)

                if article_data and article_data.get('has_content'):
                    article_id = st.session_state.rag_agent.add_user_article(
                        article_data,
                        st.session_state.user_id
                    )

                    if article_id:
                        st.success(f"✅ Документ '{article_data['title'][:30]}...' добавлен!")
                        st.session_state.user_documents = st.session_state.rag_agent.get_user_articles(
                            st.session_state.user_id)
                        st.rerun()
                    else:
                        st.error("❌ Ошибка при добавлении документа")
                else:
                    st.error("⚠️ В документе недостаточно текста или ошибка обработки")

    st.markdown("---")

    # Количество результатов
    st.subheader("🎯 Настройки поиска")
    search_limit = st.slider("Результатов на странице", 3, 20, 10)

    st.markdown("---")

    # Действия
    st.subheader("⚙️ Действия")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Обновить", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("🗑️ Очистить", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.caption("© 2024 Cloud.ru AI Agent")

# ==================== ГЛАВНАЯ СТРАНИЦА ====================
st.title("📚 AI RAG-агент для поиска статей")
st.markdown("""
    *Интеллектуальный поиск и анализ IT-статей с выбором области поиска*

    **📊 Текущая область поиска:**
    - 🔎 Везде - поиск по всем источникам
    - 📁 Только мои документы - только ваши PDF
    - 🌐 Только Habr - только статьи Habr

    **💡 Измените область поиска в боковой панели слева**
""")

# Разделитель
st.markdown("---")

# ==================== ВКЛАДКИ ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Поиск",
    "💬 Чат",
    "📚 Мои документы",
    "📊 Статистика"
])

# ==================== ВКЛАДКА 1: ПОИСК ====================
with tab1:
    st.header("🔍 Интеллектуальный поиск")

    # Отображение текущей области поиска
    scope_emoji = {
        "all": "🔎",
        "user": "📁",
        "habr": "🌐"
    }

    st.info(f"{scope_emoji[st.session_state.search_scope]} **Ищу: {scope_info[st.session_state.search_scope]}**")

    # Поле поиска
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input(
            "Введите запрос для поиска",
            placeholder="Например: Как работает RAG архитектура?",
            key="search_input"
        )

    with col2:
        if st.button("🔍 Искать", use_container_width=True):
            pass  # Кнопка для триггера

    if search_query:
        with st.spinner(f"🔍 Ищу {scope_info[st.session_state.search_scope].split()[0]}..."):
            # Получаем область поиска
            scope = get_scope_enum(st.session_state.search_scope)

            if scope:
                # Выполняем поиск
                search_results = st.session_state.rag_agent.search(
                    search_query,
                    st.session_state.user_id,
                    scope=scope,
                    limit=search_limit
                )

                if not search_results:
                    st.warning(
                        f"📭 По вашему запросу ничего не найдено {scope_info[st.session_state.search_scope].split()[0]}")
                else:
                    # Показываем результаты
                    st.success(f"✅ Найдено {len(search_results)} результатов")

                    # Сводка
                    with st.expander("📝 Краткая сводка", expanded=True):
                        # Генерируем ответ
                        result = st.session_state.rag_agent.generate_answer(
                            search_query,
                            st.session_state.user_id,
                            scope
                        )
                        st.markdown(result["answer"])

                    # Детальные результаты
                    st.subheader("📚 Найденные материалы")

                    for i, result in enumerate(search_results, 1):
                        article = result['article']
                        source_type = "📁 Ваш документ" if result['is_user_document'] else "🌐 Habr"

                        with st.container():
                            col_a, col_b = st.columns([4, 1])

                            with col_a:
                                st.markdown(f"**{i}. {article['title']}**")
                                st.caption(f"{source_type} • {article.get('author', 'Не указан')} • "
                                           f"{article.get('date', '')[:10] if article.get('date') else 'Нет даты'}")

                                # Краткий превью текста
                                preview = article['text'][:200] + "..." if len(article['text']) > 200 else article[
                                    'text']
                                st.markdown(f"*{preview}*")

                                # Теги
                                tags = article.get('tags', [])
                                if tags:
                                    tag_cols = st.columns(min(len(tags), 5))
                                    for idx, tag in enumerate(tags[:5]):
                                        with tag_cols[idx]:
                                            st.markdown(f"`{tag}`")

                            with col_b:
                                st.metric("Символов", f"{article.get('text_length', 0):,}")
                                relevance = result['score']
                                st.progress(min(relevance / 10, 1.0), text=f"Релевантность: {relevance:.1f}")

                                # Кнопки действий
                                col_btn1, col_btn2 = st.columns(2)
                                with col_btn1:
                                    if st.button("📋 Копировать", key=f"copy_{i}"):
                                        st.toast(f"Ссылка скопирована: {article['url']}")
                                with col_btn2:
                                    if st.button("🔗 Открыть", key=f"open_{i}"):
                                        if not article['url'].startswith('file://'):
                                            st.markdown(f"[Открыть ссылку]({article['url']})")

                            st.markdown("---")

                    # Рекомендации
                    if len(search_results) > 1:
                        st.subheader("🔗 Рекомендуемые материалы")

                        # Находим похожие статьи
                        similar_articles = []
                        for result in search_results[:3]:
                            article = result['article']
                            tags = article.get('tags', [])
                            if tags:
                                # Ищем статьи с похожими тегами
                                for other in st.session_state.rag_agent.all_articles:
                                    if other['id'] != article['id'] and any(
                                            tag in other.get('tags', []) for tag in tags):
                                        if other not in similar_articles and len(similar_articles) < 5:
                                            similar_articles.append(other)

                        if similar_articles:
                            for article in similar_articles[:3]:
                                with st.container():
                                    st.markdown(f"**{article['title']}**")
                                    st.caption(f"Похоже на ваш запрос • {article.get('source', 'Unknown')}")
                                    if not article['url'].startswith('file://'):
                                        st.markdown(f"[Открыть]({article['url']})")
                        else:
                            st.info("Попробуйте уточнить запрос для получения рекомендаций")

# ==================== ВКЛАДКА 2: ЧАТ ====================
with tab2:
    st.header("💬 Чат с AI-агентом")

    # Область поиска для чата
    chat_scope = st.selectbox(
        "Область поиска для чата:",
        options=["🔎 Везде", "📁 Только мои документы", "🌐 Только Habr"],
        key="chat_scope"
    )

    # Конвертируем в SearchScope
    chat_scope_enum = SearchScope.ALL
    if chat_scope == "📁 Только мои документы":
        chat_scope_enum = SearchScope.USER_ONLY
    elif chat_scope == "🌐 Только Habr":
        chat_scope_enum = SearchScope.HABR_ONLY

    # Отображаем историю чата
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if message.get('sources'):
                with st.expander("📚 Показать источники"):
                    for source in message['sources']:
                        st.markdown(f"• {source['title']}")

    # Поле ввода
    chat_input = st.chat_input("Задайте вопрос AI-агенту...")

    if chat_input:
        # Добавляем сообщение пользователя
        st.session_state.chat_history.append({
            'role': 'user',
            'content': chat_input,
            'timestamp': datetime.now().isoformat()
        })

        with st.chat_message('user'):
            st.markdown(chat_input)

        # Получаем ответ
        with st.spinner(f"🤔 AI-агент думает ({chat_scope})..."):
            result = st.session_state.rag_agent.generate_answer(
                chat_input,
                st.session_state.user_id,
                chat_scope_enum
            )

            # Добавляем ответ в историю
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': result["answer"],
                'sources': result["sources"],
                'questions': result["questions"],
                'timestamp': datetime.now().isoformat()
            })

            # Отображаем ответ
            with st.chat_message('assistant'):
                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("📚 Показать источники", expanded=False):
                        for i, source in enumerate(result["sources"], 1):
                            icon = "📁" if source.get('is_user_document') else "🌐"
                            st.markdown(f"{i}. {icon} **{source['title']}**")

            # Вопросы для самопроверки
            if result.get('questions'):
                with st.expander("❓ Вопросы для самопроверки", expanded=True):
                    for question in result["questions"]:
                        st.markdown(f"• {question}")

# ==================== ВКЛАДКА 3: МОИ ДОКУМЕНТЫ ====================
with tab3:
    st.header("📚 Мои документы")

    # Загрузка пользовательских документов
    user_docs = st.session_state.rag_agent.get_user_articles(st.session_state.user_id)
    st.session_state.user_documents = user_docs

    if not user_docs:
        st.info("""
        📭 **У вас пока нет загруженных документов.**

        **Как добавить документы:**
        1. Перейдите в боковую панель
        2. Выберите PDF файл в разделе "Загрузить документы"
        3. Нажмите "Загрузить и обработать"
        4. Документ появится в этой вкладке

        💡 **После загрузки вы сможете:**
        • Искать только по своим документам
        • Использовать область поиска "📁 Только мои документы"
        • Анализировать содержимое документов
        """)
    else:
        # Статистика документов
        total_chars = sum(len(doc.get('text', '')) for doc in user_docs)
        avg_chars = total_chars // len(user_docs)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Всего документов", len(user_docs))
        with col2:
            st.metric("Всего символов", f"{total_chars:,}")
        with col3:
            st.metric("Средний размер", f"{avg_chars:,}")

        # Быстрый поиск по документам
        st.subheader("🔍 Быстрый поиск по документам")

        quick_search = st.text_input(
            "Поиск в ваших документах",
            placeholder="Введите ключевые слова...",
            key="quick_search"
        )

        if quick_search:
            # Ищем только в документах пользователя
            search_results = st.session_state.rag_agent.search(
                quick_search,
                st.session_state.user_id,
                scope=SearchScope.USER_ONLY,
                limit=10
            )

            if search_results:
                st.success(f"✅ Найдено {len(search_results)} совпадений")
                for result in search_results:
                    article = result['article']
                    with st.expander(f"📄 {article['title']}"):
                        st.markdown(f"**Символов:** {article.get('text_length', 0):,}")
                        # Показываем контекст совпадения
                        text = article.get('text', '')
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

        # Таблица документов
        st.subheader("📋 Список документов")

        # Создаем DataFrame для отображения
        docs_data = []
        for doc in user_docs:
            docs_data.append({
                "Название": doc.get('title', 'Без названия'),
                "Автор": doc.get('author', 'Не указан'),
                "Дата": doc.get('uploaded_at', doc.get('date', ''))[:10],
                "Символов": doc.get('text_length', 0),
                "Страниц": len(doc.get('pages', [])),
                "Теги": ', '.join(doc.get('tags', [])[:3])
            })

        if docs_data:
            df = pd.DataFrame(docs_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Название": st.column_config.TextColumn(width="large"),
                    "Символов": st.column_config.NumberColumn(format="%d"),
                    "Теги": st.column_config.TextColumn(width="medium")
                }
            )

# ==================== ВКЛАДКА 4: СТАТИСТИКА ====================
with tab4:
    st.header("📊 Статистика системы")

    # Получаем статистику
    stats = st.session_state.rag_agent.get_statistics(st.session_state.user_id)

    # Основные метрики
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Всего документов", stats['total_articles'])
    with col2:
        st.metric("Статей Habr", stats['habr_articles'])
    with col3:
        st.metric("Пользовательских", stats['user_articles'])
    with col4:
        st.metric("Ваших документов", stats['current_user_articles'])

    st.markdown("---")

    # Визуализации
    col1, col2 = st.columns(2)

    with col1:
        # Распределение по источникам
        st.subheader("📈 Распределение по источникам")

        sources_data = {
            'Habr': stats['habr_articles'],
            'Пользовательские': stats['user_articles']
        }

        fig = px.pie(
            values=list(sources_data.values()),
            names=list(sources_data.keys()),
            title="Источники документов",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Диаграмма пользовательских документов
        st.subheader("👤 Ваши документы")

        if stats['current_user_articles'] > 0:
            fig = px.bar(
                x=['Ваши документы'],
                y=[stats['current_user_articles']],
                title="Количество ваших документов",
                labels={'x': '', 'y': 'Количество'}
            )
            fig.update_traces(marker_color='green')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("У вас пока нет документов")

    # Информация о системе
    st.subheader("ℹ️ Информация о системе")

    sys_info = {
        "ID пользователя": st.session_state.user_id,
        "Область поиска по умолчанию": scope_info[st.session_state.search_scope],
        "Обновлено": stats['last_update'][:19]
    }

    for key, value in sys_info.items():
        st.markdown(f"**{key}:** {value}")

# ==================== ФУТЕР ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>📚 <strong>AI RAG-агент dreamteam___.ru</strong> | Версия 1.0.0</p>
    <p>🤖 Интеллектуальный поиск и анализ IT-статей</p>
    <p>📧 Поддержка: support@dreamteam____.ru | 📞 +7 (XXX) XXX-XX-XX</p>
</div>
""", unsafe_allow_html=True)

# ==================== СТИЛИ CSS ====================
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }

    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)