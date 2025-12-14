# app.py: Streamlit UI с улучшенным интерфейсом
import streamlit as st
import time
from src.core.rag import RAGSystem

st.set_page_config(
    page_title="AI-агент поиска статей",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация сессии
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem(
        vector_db_path="chroma_full_db",
        llm_model="llama3.2:3b",
        llm_base_url="http://localhost:11434"
    )
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Заголовок и описание
st.title("🔍 AI-агент для интеллектуального поиска статей")
st.markdown("""
    Поиск и анализ технических статей с использованием RAG (Retrieval-Augmented Generation).
    Система использует локальную LLM через Ollama для генерации ответов.
""")

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Проверка статуса LLM
    with st.expander("Статус системы", expanded=True):
        llm_status = st.session_state.rag_system.check_llm_connection()
        
        if llm_status["connected"]:
            st.success(f"✅ LLM подключена")
            st.info(f"**Модель:** {llm_status['current_model']}")
        else:
            st.error("❌ LLM не подключена")
            st.markdown("""
            **Для работы нужен Ollama:**
            1. [Установите Ollama](https://ollama.com/)
            2. Запустите: `ollama serve`
            3. Загрузите модель: `ollama pull llama3.2:3b`
            """)
    
    # Настройки поиска
    st.subheader("Параметры поиска")
    top_k = st.slider("Количество результатов", 1, 10, 5)
    
    # Фильтры
    st.subheader("Фильтры")
    use_filters = st.checkbox("Использовать фильтры")
    
    if use_filters:
        col1, col2 = st.columns(2)
        with col1:
            filter_author = st.text_input("Автор", "")
        with col2:
            filter_source = st.selectbox(
                "Источник",
                ["Все"] + st.session_state.rag_system.get_available_sources()
            )
    
    # Информация о базе
    st.subheader("Информация")
    st.markdown("""
    **Источники статей:**
    - Habr (700+ статей)
    - Загруженные PDF
    - Другие IT-ресурсы
    """)
    
    # Кнопка очистки истории
    if st.button("Очистить историю чата"):
        st.session_state.chat_history = []
        st.rerun()

# Основная область
tab1, tab2 = st.tabs(["💬 Поиск и чат", "📊 История и анализ"])

with tab1:
    # Ввод запроса
    query = st.text_input(
        "Введите ваш запрос",
        placeholder="Например: Как работает RAG? Или: Новости машинного обучения",
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        search_button = st.button("🔍 Искать", type="primary", use_container_width=True)
    with col2:
        clear_query = st.button("Очистить", use_container_width=True)
    
    if clear_query:
        st.rerun()
    
    # Обработка запроса
    if search_button and query:
        with st.spinner("Ищу статьи и генерирую ответ..."):
            try:
                # Применяем фильтры если нужно
                filters = {}
                if use_filters:
                    if filter_author:
                        filters["author"] = filter_author
                    if filter_source and filter_source != "Все":
                        filters["source"] = filter_source
                
                # Получаем ответ от RAG системы
                start_time = time.time()
                result = st.session_state.rag_system.generate_answer(query, top_k=top_k)
                processing_time = time.time() - start_time
                
                # Сохраняем в историю
                st.session_state.chat_history.append({
                    "query": query,
                    "result": result,
                    "timestamp": time.time(),
                    "processing_time": processing_time
                })
                
                # Отображаем ответ
                st.markdown("---")
                
                # Ответ
                st.subheader("📝 Ответ")
                st.markdown(result.get("answer", "Ответ не сгенерирован."))
                
                # Источники
                sources = result.get("sources", [])
                if sources:
                    st.subheader("📚 Источники")
                    
                    for i, src in enumerate(sources, 1):
                        with st.expander(f"{i}. {src.get('title', 'Без названия')}"):
                            st.markdown(f"**Автор:** {src.get('author', 'Неизвестен')}")
                            st.markdown(f"**Источник:** {src.get('source', 'Unknown')}")
                            url = src.get('url', '')
                            if url:
                                st.markdown(f"**Ссылка:** [Открыть статью]({url})")
                
                # Вопросы для самопроверки
                questions = result.get("questions", [])
                if questions:
                    st.subheader("❓ Вопросы для самопроверки")
                    for q in questions:
                        st.markdown(f"- {q}")
                
                # Похожие статьи
                similar = result.get("similar_articles", [])
                if similar:
                    st.subheader("📖 Рекомендуемые статьи")
                    cols = st.columns(min(3, len(similar)))
                    for idx, article in enumerate(similar[:3]):
                        with cols[idx % 3]:
                            st.markdown(f"**{article.get('title', '')}**")
                            url = article.get('url', '')
                            if url:
                                st.markdown(f"[Читать статью]({url})")
                
                # Статистика
                with st.expander("📊 Статистика запроса"):
                    stats = result.get("search_stats", {})
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Время обработки", f"{processing_time:.2f}с")
                    col2.metric("Найдено статей", stats.get('total_found', 0))
                    col3.metric("Использовано в ответе", stats.get('context_used', 0))
                    
            except Exception as e:
                st.error(f"Ошибка при обработке запроса: {str(e)}")
                st.info("Проверьте, что Ollama запущен и модель загружена.")

with tab2:
    if not st.session_state.chat_history:
        st.info("История запросов пуста. Сделайте первый запрос во вкладке 'Поиск и чат'.")
    else:
        st.subheader("📋 История запросов")
        
        # Показываем историю в обратном порядке (последние первые)
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Запрос: {chat['query'][:50]}...", expanded=(idx == 0)):
                st.markdown(f"**Запрос:** {chat['query']}")
                st.markdown(f"**Время:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(chat['timestamp']))}")
                st.markdown(f"**Обработка:** {chat['processing_time']:.2f} секунд")
                
                result = chat['result']
                st.markdown("**Ответ:**")
                st.markdown(result.get('answer', '')[:500] + "...")
                
                # Быстрый доступ к источникам
                sources = result.get('sources', [])
                if sources:
                    st.markdown(f"**Источников:** {len(sources)}")
                
                # Кнопка для повторного использования запроса
                if st.button(f"Повторить этот запрос", key=f"repeat_{idx}"):
                    st.session_state.query_input = chat['query']
                    st.rerun()
        
        # Статистика по истории
        if len(st.session_state.chat_history) > 1:
            st.markdown("---")
            st.subheader("📈 Общая статистика")
            total_queries = len(st.session_state.chat_history)
            avg_time = sum(c['processing_time'] for c in st.session_state.chat_history) / total_queries
            
            col1, col2 = st.columns(2)
            col1.metric("Всего запросов", total_queries)
            col2.metric("Среднее время", f"{avg_time:.2f}с")

# Футер с информацией
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Архитектура:** RAG + локальная LLM")
with col2:
    st.markdown("**Модель:** Llama 3.2 3B через Ollama")
with col3:
    st.markdown("**Векторная БД:** ChromaDB")