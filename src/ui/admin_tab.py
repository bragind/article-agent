"""
Вкладка админ-панели
"""

import streamlit as st
import pandas as pd
import logging

from . import helpers

logger = logging.getLogger(__name__)


class AdminTab:
    """Класс админ-панели"""
    
    def __init__(self, auth_manager, rag_agent, project_root):
        self.auth_manager = auth_manager
        self.rag_agent = rag_agent
        self.project_root = project_root
    
    def render_users_tab(self):
        """Рендер вкладки управления пользователями"""
        st.subheader("👥 Управление пользователями")
        
        if hasattr(self.auth_manager, 'auth_manager') and self.auth_manager.auth_manager:
            users_data = self.auth_manager.auth_manager.get_users_dataframe()
            
            if users_data:
                df_users = pd.DataFrame(users_data)
                st.dataframe(df_users, use_container_width=True, hide_index=True)
            else:
                st.info("Нет зарегистрированных пользователей")
        else:
            st.info("Модуль управления пользователями недоступен в демо-режиме")
    
    def render_system_tab(self):
        """Рендер вкладки управления системой"""
        st.subheader("⚙️ Управление системой")
        
        if self.rag_agent:
            rag_stats = self.rag_agent.get_statistics(self.auth_manager.get_user_id())
            
            # Статус системы
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📚 Всего статей", rag_stats["total_articles"])
            with col2:
                status = "🟢 Активен" if rag_stats.get("vector_search_enabled", False) else "🔴 Ошибка"
                st.metric("🔍 Векторный поиск", status)
            
            # Действия администратора
            st.subheader("⚡ Действия")
            
            action_col1, action_col2, action_col3 = st.columns(3)
            with action_col1:
                if st.button("🔄 Перестроить индекс", use_container_width=True):
                    with st.spinner("Перестраиваю векторный индекс..."):
                        success, message = self.rag_agent.rebuild_index()
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
    
    def render(self):
        """Рендер админ-панели"""
        st.header("👑 Административная панель")
        
        admin_tabs = st.tabs(["⚙️ Система"])
        
        with admin_tabs[0]:
            self.render_system_tab()