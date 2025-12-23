"""
Вкладка статистики
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import logging

from . import helpers
from .constants import CHART_COLORS

logger = logging.getLogger(__name__)


class StatisticsTab:
    """Класс вкладки статистики"""
    
    def __init__(self, auth_manager, rag_agent):
        self.auth_manager = auth_manager
        self.rag_agent = rag_agent
    
    def display_enhanced_statistics(self):
        """Отображает расширенную статистику"""
        if not self.rag_agent:
            st.error("RAG агент не инициализирован")
            return
        
        try:
            stats = helpers.create_enhanced_statistics(self.rag_agent, self.auth_manager.get_user_id())
            
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
                
                # Топ теги
                if "top_tags" in stats and stats["top_tags"]:
                    st.subheader("🏷️ Топ-10 теги")
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
                # Топ авторы
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
                st.markdown(f"• ID пользователя: `{self.auth_manager.get_user_id()}`")
                st.markdown(f"• Всего уникальных теги: {len(stats.get('top_tags', {}))}")
            
            # Кнопка обновления статистики
            if st.button("🔄 Обновить статистику", key="refresh_stats"):
                st.rerun()
                
        except Exception as e:
            st.error(f"Ошибка отображения статистики: {e}")
            logger.error(f"Ошибка отображения статистики: {e}")
    
    def render(self):
        """Рендер вкладки статистики"""
        st.header("📊 Расширенная статистика")
        
        if self.rag_agent:
            self.display_enhanced_statistics()
        else:
            st.error("❌ Система не инициализирована")