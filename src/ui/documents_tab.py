"""
Вкладка документов пользователя
"""

import streamlit as st
import pandas as pd
from typing import List, Dict
import logging

from . import helpers
from . import components
from src.core.search_engine import SearchScope

logger = logging.getLogger(__name__)


class DocumentsTab:
    """Класс вкладки документов"""
    
    def __init__(self, auth_manager, rag_agent, project_root):
        self.auth_manager = auth_manager
        self.rag_agent = rag_agent
        self.project_root = project_root
    
    def display_document_upload_section(self):
        """Отображение секции загрузки документов"""
        st.markdown("---")
        st.subheader("📤 Загрузить новые документы")
        
        uploaded_file = st.file_uploader(
            "Выберите PDF файл для загрузки", 
            type=["pdf"], 
            help="Максимальный размер: 100MB",
            key="doc_uploader"
        )
        
        if uploaded_file is not None:
            # Показываем информацию о файле
            file_info_col1, file_info_col2 = st.columns(2)
            with file_info_col1:
                st.info(f"**Файл:** {uploaded_file.name}")
                st.info(f"**Размер:** {uploaded_file.size / 1024:.1f} KB")
            with file_info_col2:
                if st.button("📥 Загрузить и обработать", key="upload_process", use_container_width=True):
                    with st.spinner("📄 Обработка PDF..."):
                        article_data = helpers.load_pdf_file(uploaded_file)
                        
                        if article_data and article_data.get("has_content"):
                            user_id = self.auth_manager.get_user_id()
                            article_id = self.rag_agent.add_user_document(
                                article_data, user_id
                            )
                            
                            if article_id:
                                st.success(f"✅ Документ успешно добавлен!")
                                
                                # Показать детали документа
                                with st.expander("📋 Детали загруженного документа", expanded=True):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown(f"**Название:** {article_data['title']}")
                                        st.markdown(f"**Страниц:** {len(article_data.get('pages', []))}")
                                    with col2:
                                        st.markdown(f"**Символов:** {article_data['text_length']:,}")
                                        st.markdown(f"**ID документа:** `{article_id}`")
                                
                                # Автоматически обновляем список документов
                                if "user_documents" in st.session_state:
                                    del st.session_state.user_documents
                                st.rerun()
                            else:
                                st.error("❌ Ошибка при добавлении документа")
                        else:
                            st.error("❌ В документе недостаточно текста для обработки")
        
        # Подсказки по загрузке
        with st.expander("💡 Советы по загрузке документов", expanded=False):
            st.markdown("""
            **📝 Рекомендации:**
            1. **Формат файла**: поддерживаются только PDF
            2. **Максимальный размер**: 100 MB
            3. **Качество текста**: лучше загружать PDF с распознанным текстом, а не сканы
            4. **Структура**: документы с четкой структурой (заголовки, параграфы) обрабатываются лучше
            5. **Автоматические теги**: система сама генерирует теги для загруженных документов
            
            **🔄 После загрузки:**
            - Документ сразу появится в списке "Ваши документы"
            - Можно искать по его содержимому
            - Доступны вопросы для самопроверки
            - Система предложит похожие статьи
            """)
    
    def display_user_documents(self):
        """Отображение документов пользователя"""
        current_user_id = self.auth_manager.get_user_id()
        
        # Показываем ID пользователя для отладки
        st.caption(f"Ваш ID пользователя: `{current_user_id}`")
        
        # Загружаем документы пользователя
        user_docs = []
        if self.rag_agent:
            try:
                user_docs = self.rag_agent.get_user_documents(current_user_id)
                logger.info(f"Для пользователя {current_user_id} загружено {len(user_docs)} документов")
            except Exception as e:
                st.error(f"Ошибка загрузки документов: {e}")
                logger.error(f"Ошибка в get_user_articles для {current_user_id}: {e}")
        
        # СЕКЦИЯ СУЩЕСТВУЮЩИХ ДОКУМЕНТОВ
        st.markdown("---")
        st.subheader(f"📋 Ваши документы ({len(user_docs)})")
        
        if not user_docs:
            st.info("""
            **📭 У вас пока нет загруженных документов.**
            
            **📥 Как добавить документы:**
            1. Загрузите PDF файл в секции выше
            2. Нажмите "Загрузить и обработать"
            3. Документ появится в этой вкладке
            """)
        else:
            # Проверяем принадлежность каждого документа
            valid_docs = []
            for doc in user_docs:
                uploaded_by = doc.get("uploaded_by", "")
                if uploaded_by == current_user_id:
                    valid_docs.append(doc)
                else:
                    logger.warning(f"Документ '{doc.get('title', 'Без названия')}' принадлежит {uploaded_by}, а не {current_user_id}")
            
            if len(valid_docs) != len(user_docs):
                st.warning(f"⚠️ Найдено {len(user_docs) - len(valid_docs)} документов других пользователей")
            
            if not valid_docs:
                st.warning(f"""
                **⚠️ У вас нет документов или произошла ошибка фильтрации**
                
                ID пользователя: `{current_user_id}`
                Всего найдено документов: {len(user_docs)}
                
                **Возможные причины:**
                1. Документы сохранены под другим ID пользователя
                2. Ошибка в системе хранения
                3. Попробуйте перезагрузить страницу
                """)
            else:
                # Статистика
                total_chars = sum(len(doc.get("text", "")) for doc in valid_docs)
                avg_chars = total_chars // len(valid_docs) if valid_docs else 0
                total_pages = sum(len(doc.get("pages", [])) for doc in valid_docs)
                
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.metric("📄 Всего документов", len(valid_docs))
                with stat_col2:
                    st.metric("📑 Всего страниц", total_pages)
                with stat_col3:
                    st.metric("🔤 Всего символов", f"{total_chars:,}")
                with stat_col4:
                    st.metric("📊 Средний размер", f"{avg_chars:,}")
                
                # Список документов пользователя
                st.subheader("📋 Список ваших документов")
                
                docs_data = []
                for doc in valid_docs:
                    # Дополнительная проверка принадлежности
                    if doc.get("uploaded_by") == current_user_id:
                        tags = doc.get("tags", [])
                        if isinstance(tags, list):
                            tags_display = ", ".join(tags[:3])
                            if len(tags) > 3:
                                tags_display += f" (+{len(tags) - 3})"
                        else:
                            tags_display = str(tags)[:50]
                        
                        docs_data.append({
                            "📄 Название": doc.get("title", "Без названия"),
                            "👤 Автор": doc.get("author", "Не указан"),
                            "📅 Дата загрузки": doc.get("uploaded_at", doc.get("date", ""))[:10],
                            "🔤 Символов": doc.get("text_length", 0),
                            "📑 Страниц": len(doc.get("pages", [])),
                            "🏷️ Теги": tags_display,
                            "📁 ID": doc.get("id", "N/A")[:12]
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
                    
                    # Детальный просмотр и управление документами
                    st.markdown("---")
                    st.subheader("🔍 Управление документами")
                    
                    doc_titles = [d["📄 Название"] for d in docs_data]
                    selected_doc_title = st.selectbox(
                        "Выберите документ для просмотра:",
                        options=doc_titles,
                        index=0,
                        key="doc_viewer_select"
                    )
                    
                    if selected_doc_title:
                        selected_doc = next((d for d in valid_docs if d.get("title") == selected_doc_title), None)
                        if selected_doc:
                            with st.expander("📖 Содержимое документа", expanded=False):
                                st.markdown(f"### {selected_doc['title']}")
                                st.markdown(f"**Автор:** {selected_doc.get('author', 'Не указан')}")
                                st.markdown(f"**Теги:** {', '.join(selected_doc.get('tags', []))}")
                                st.markdown(f"**Загружен:** {selected_doc.get('uploaded_at', '')[:19]}")
                                st.markdown(f"**ID документа:** `{selected_doc.get('id', 'N/A')}`")
                                
                                # Показываем первые 1000 символов
                                text_preview = selected_doc.get("text", "")[:1000]
                                if len(selected_doc.get("text", "")) > 1000:
                                    text_preview += "..."
                                
                                st.markdown("**Текст документа:**")
                                st.markdown(f"> {text_preview}")
                                
                                # Кнопки действий
                                action_col1, action_col2 = st.columns(2)
                                with action_col1:
                                    if st.button("🔍 Искать в этом документе", key=f"search_in_doc_{selected_doc['id']}", use_container_width=True):
                                        st.session_state.search_query = ""
                                        st.session_state.search_scope = "user"
                                        st.session_state.search_triggered = True
                                        # Переключаемся на вкладку поиска
                                        st.success(f"🔍 Поиск в документе: {selected_doc['title']}")
                                        # В Streamlit нет прямого переключения вкладок, обновляем страницу с параметрами
                                        st.rerun()
                                with action_col2:
                                    if st.button("🗑️ Удалить документ", type="secondary", key=f"delete_doc_{selected_doc['id']}", use_container_width=True):
                                        # Подтверждение удаления
                                        with st.container():
                                            st.warning("⚠️ Вы уверены, что хотите удалить этот документ?")
                                            confirm_col1, confirm_col2 = st.columns(2)
                                            with confirm_col1:
                                                if st.button("✅ Да, удалить", key=f"confirm_delete_{selected_doc['id']}"):
                                                    if helpers.delete_user_document(selected_doc['id'], current_user_id, self.project_root):
                                                        st.success("✅ Документ успешно удален")
                                                        # Обновляем список документов
                                                        if "user_documents" in st.session_state:
                                                            del st.session_state.user_documents
                                                        st.rerun()
                                                    else:
                                                        st.error("❌ Ошибка удаления документа")
                                            with confirm_col2:
                                                if st.button("❌ Отмена", key=f"cancel_delete_{selected_doc['id']}"):
                                                    st.rerun()
                else:
                    st.info("""
                    **📭 У вас нет документов**
                    
                    Загрузите PDF файл в секции выше, чтобы добавить документы в вашу коллекцию.
                    """)
    
    def render(self):
        """Рендер вкладки документов"""
        st.header("📚 Мои документы")
        
        if self.auth_manager.is_authenticated() and self.auth_manager.is_user():
            # Секция загрузки документов
            self.display_document_upload_section()
            
            # Секция существующих документов
            self.display_user_documents()
        else:
            st.info("📚 Управление документами доступно только для зарегистрированных пользователей")