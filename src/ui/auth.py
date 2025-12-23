"""
src/ui/auth.py - Тонкая обертка для интеграции аутентификации с UI
"""

import streamlit as st
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Импортируем основной модуль аутентификации
try:
    from src.auth.auth import streamlit_auth, auth_manager as main_auth_manager
    AUTH_MODULE_AVAILABLE = True
    logger.info("Основной модуль аутентификации успешно импортирован")
except ImportError as e:
    AUTH_MODULE_AVAILABLE = False
    logger.warning(f"Основной модуль аутентификации недоступен: {e}")
    # Создаем заглушки
    class StreamlitAuthStub:
        def check_auth(self):
            st.warning("🔒 Модуль аутентификации недоступен. Используется гостевой доступ.")
            st.session_state.authenticated = True
            st.session_state.username = "guest"
            st.session_state.user_role = "guest"
            st.session_state.user_id = "guest_user"
            return True
        
        def show_user_profile(self):
            st.sidebar.info("👤 **Гостевой доступ**")
            st.sidebar.info("🔒 Аутентификация отключена")
    
    streamlit_auth = StreamlitAuthStub()
    main_auth_manager = None


class AuthManager:
    """Менеджер аутентификации для UI"""
    
    def __init__(self):
        self.auth_available = AUTH_MODULE_AVAILABLE
        self.streamlit_auth = streamlit_auth
        self.main_auth_manager = main_auth_manager
        
    def check_auth(self) -> bool:
        """Проверка аутентификации"""
        if self.auth_available:
            return self.streamlit_auth.check_auth()
        else:
            # Если модуль недоступен, используем заглушку
            return self.streamlit_auth.check_auth()
    
    def show_user_profile(self):
        """Отображение профиля пользователя в сайдбаре"""
        if self.auth_available:
            self.streamlit_auth.show_user_profile()
        else:
            # Показываем минимальную информацию для гостя
            with st.sidebar.expander("👤 Профиль", expanded=False):
                st.write("**Пользователь:** Гость")
                st.write("**Роль:** Гостевой доступ")
                st.write("🔒 Аутентификация отключена")
    
    def get_user_role(self) -> str:
        """Получение роли пользователя"""
        return st.session_state.get("user_role", "guest")
    
    def get_user_id(self) -> str:
        """Получение ID пользователя"""
        # Используем username как user_id, если не задан отдельный ID
        username = st.session_state.get("username", "guest")
        if username == "guest":
            return "guest_user"
        
        # Преобразуем username в безопасный ID
        import re
        safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', username.lower())
        return f"user_{safe_id}"
    
    def get_username(self) -> str:
        """Получение имени пользователя"""
        return st.session_state.get("username", "guest")
    
    def is_authenticated(self) -> bool:
        """Проверка аутентификации"""
        return st.session_state.get("authenticated", False)
    
    def is_admin(self) -> bool:
        """Проверка роли администратора"""
        return self.get_user_role() == "admin"
    
    def is_user(self) -> bool:
        """Проверка роли пользователя (админ или обычный пользователь)"""
        return self.get_user_role() in ["admin", "user"]
    
    def require_auth(self):
        """Декоратор для защиты страниц"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.check_auth():
                    st.error("🔒 Требуется аутентификация")
                    st.stop()
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_permission(self, permission: str):
        """Декоратор для проверки разрешений"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.is_authenticated():
                    st.error("🔒 Требуется аутентификация")
                    st.stop()
                
                # Для упрощения проверяем только основные роли
                if permission == "admin" and not self.is_admin():
                    st.error("⛔ Требуются права администратора")
                    st.stop()
                elif permission == "user" and not self.is_user():
                    st.error("⛔ Требуются права пользователя")
                    st.stop()
                
                return func(*args, **kwargs)
            return wrapper
        return decorator