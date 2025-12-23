"""
Основной модуль Streamlit приложения
"""

import streamlit as st
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Подготовка пути к проекту
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Импорт модулей UI
from .auth import AuthManager
from .components import UIComponents
from .search_tab import SearchTab
from .documents_tab import DocumentsTab
from .statistics_tab import StatisticsTab
from .admin_tab import AdminTab
from .constants import PAGE_CONFIG

class StreamlitApp:
    """Основной класс Streamlit приложения"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.auth_manager = AuthManager()
        self.rag_agent = None
        self.initialized = False
        
        # Инициализация состояния сессии
        self._init_session_state()
    
    def _init_session_state(self):
        """Инициализация состояния сессии"""
        # Критические состояния
        if "initialized" not in st.session_state:
            st.session_state.initialized = False
        if "rag_agent" not in st.session_state:
            st.session_state.rag_agent = None
        if "user_documents" not in st.session_state:
            st.session_state.user_documents = []
        
        # Состояния поиска
        default_states = {
            "search_triggered": False,
            "search_scope": "all",
            "search_tags": [],
            "search_author": None,
            "search_date_from": None,
            "search_date_to": None,
            "search_limit": 10,
            "last_search_query": ""
        }
        
        for key, default in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default
    
    def initialize_rag(self):
        """Инициализация RAG агента с обработкой ошибок"""
        if not st.session_state.rag_agent:
            try:
                with st.spinner("🤖 Инициализация RAG агента..."):
                    from src.core.rag_orchestrator import RAGOrchestrator
                    
                    st.session_state.rag_agent = RAGOrchestrator(
                        data_dir=str(self.project_root / "data"),
                        vector_db_path=str(self.project_root / "chroma_db"),
                        use_vector_search=True,
                        llm_enabled=True
                    )
                    
                    # Проверяем готовность системы
                    if st.session_state.rag_agent.is_ready():
                        st.session_state.initialized = True
                        logger.info("✅ RAG агент успешно инициализирован")
                        st.success("✅ Система инициализирована!")
                    else:
                        status = st.session_state.rag_agent.get_statistics(
                            self.auth_manager.get_user_id()
                        )
                        logger.warning(f"⚠️ RAG агент не полностью готов: {status}")
                        
                        # Показываем информацию о статусе
                        with st.expander("⚠️ Информация о системе", expanded=False):
                            st.json(status)
                        
                        st.warning("⚠️ Некоторые компоненты системы недоступны")
                        
            except ImportError as e:
                logger.error(f"❌ Ошибка импорта RAGOrchestrator: {e}")
                st.error(f"❌ Ошибка загрузки модулей: {str(e)[:200]}")
                st.info("Проверьте зависимости: pip install sentence-transformers chromadb")
                st.session_state.rag_agent = None
                
            except Exception as e:
                logger.error(f"❌ Ошибка инициализации RAGOrchestrator: {e}")
                st.error(f"❌ Ошибка инициализации системы: {str(e)[:200]}")
                
                # Показать более подробную информацию
                with st.expander("🔍 Технические детали", expanded=False):
                    st.code(str(e))
                
                st.session_state.rag_agent = None
        
        self.rag_agent = st.session_state.get("rag_agent")
        self.initialized = st.session_state.get("initialized", False)
    
    def display_footer(self):
        """Отображение футера"""
        st.markdown("---")
        st.markdown(
            """
<div style="text-align: center; color: #666; font-size: 0.9em; padding: 20px;">
    <p>🤖 <strong>AI RAG-агент для интеллектуального поиска статей</strong></p>
    <p>🔍 Поиск • 📤 Документы • ❓ Самопроверка • 📊 Аналитика</p>
    <p>📚 Версия 2.3 | © 2025 DreamTeam AI Agent</p>
</div>
""",
            unsafe_allow_html=True,
        )
    
    def display_error_page(self, error_message: str):
        """Отображение страницы с ошибкой"""
        st.title("⚠️ Ошибка инициализации")
        
        st.error(f"**Произошла ошибка:** {error_message}")
        
        st.markdown("---")
        st.subheader("🔧 Возможные решения:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **1. Проверьте зависимости:**
            ```bash
            pip install sentence-transformers chromadb
            ```
            
            **2. Перезапустите приложение:**
            ```bash
            streamlit run src/streamlit_app.py
            ```
            """)
        
        with col2:
            st.markdown("""
            **3. Проверьте наличие данных:**
            - Файл `data/articles_batch.jsonl`
            - Директория `chroma_db/`
            
            **4. Проверьте запуск Ollama:**
            ```bash
            ollama serve
            ```
            """)
        
        if st.button("🔄 Попробовать снова"):
            st.session_state.rag_agent = None
            st.rerun()
    
    def run(self):
        """Запуск приложения"""
        # Настройки страницы
        st.set_page_config(**PAGE_CONFIG)
        
        # Проверка аутентификации
        if not self.auth_manager.check_auth():
            st.stop()
        
        # Инициализация RAG
        try:
            self.initialize_rag()
        except Exception as e:
            self.display_error_page(str(e))
            return
        
        # Если система не инициализирована, показываем ошибку
        if not self.rag_agent or not self.initialized:
            st.error("❌ Система не инициализирована")
            
            if st.button("🔄 Попробовать снова"):
                st.session_state.rag_agent = None
                st.rerun()
            
            return
        
        # Сайдбар
        UIComponents.create_sidebar(self.auth_manager, self.rag_agent)
        
        # Заголовок
        UIComponents.display_header(self.auth_manager)
        
        # Создание вкладок
        search_tab, documents_tab, statistics_tab, admin_tab = UIComponents.create_tabs(self.auth_manager)
        
        # Вкладка поиска
        with search_tab:
            if self.rag_agent:
                search_tab_instance = SearchTab(self.auth_manager, self.rag_agent)
                search_tab_instance.render()
            else:
                st.error("❌ Система поиска не инициализирована")
        
        # Вкладка документов
        if documents_tab:
            with documents_tab:
                if self.auth_manager.is_user() and self.rag_agent:
                    documents_tab_instance = DocumentsTab(self.auth_manager, self.rag_agent, self.project_root)
                    documents_tab_instance.render()
                else:
                    st.info("📚 Управление документами доступно только для зарегистрированных пользователей")
        
        # Вкладка статистики
        with statistics_tab:
            if self.rag_agent:
                statistics_tab_instance = StatisticsTab(self.auth_manager, self.rag_agent)
                statistics_tab_instance.render()
            else:
                st.error("❌ Система статистики не инициализирована")
        
        # Админ-панель
        if admin_tab and self.auth_manager.is_admin():
            with admin_tab:
                if self.rag_agent:
                    admin_tab_instance = AdminTab(self.auth_manager, self.rag_agent, self.project_root)
                    admin_tab_instance.render()
                else:
                    st.error("❌ Админ-панель недоступна")
        
        # Футер
        self.display_footer()


# Точка входа
def main():
    """Точка входа приложения"""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()