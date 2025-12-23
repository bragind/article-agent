"""
Модуль аутентификации для Streamlit
"""

import streamlit as st
import bcrypt
import hashlib
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import json

class AuthManager:
    """Менеджер аутентификации и авторизации"""
    
    def __init__(self, config_path: str = "config/auth_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.users_file = Path("data/users.json")
        
    def load_config(self) -> Dict:
        """Загрузка конфигурации"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {
            "auth": {
                "enabled": True,
                "session_timeout_minutes": 120,
                "users": {},
                "roles": {
                    "admin": ["view", "search", "upload", "delete", "manage_users"],
                    "user": ["view", "search", "upload"],
                    "guest": ["view"]
                }
            }
        }
    
    def save_config(self):
        """Сохранение конфигурации"""
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def load_users(self) -> Dict:
        """Загрузка пользователей из файла"""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_users(self, users: Dict):
        """Сохранение пользователей в файл"""
        self.users_file.parent.mkdir(exist_ok=True)
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    
    def hash_password(self, password: str) -> str:
        """Хэширование пароля с использованием bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Проверка пароля"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except:
            return False
    
    def create_user(self, username: str, password: str, role: str = "user") -> bool:
        """Создание нового пользователя"""
        users = self.load_users()
        
        if username in users:
            return False
        
        users[username] = {
            "password_hash": self.hash_password(password),
            "role": role,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "permissions": self.config["auth"]["roles"].get(role, [])
        }
        
        self.save_users(users)
        return True
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """Аутентификация пользователя"""
        users = self.load_users()
        
        if username not in users:
            return False, None
        
        user = users[username]
        
        if not self.verify_password(password, user["password_hash"]):
            return False, None
        
        # Обновляем время последнего входа
        user["last_login"] = datetime.now().isoformat()
        users[username] = user
        self.save_users(users)
        
        return True, user
    
    def check_permission(self, username: str, permission: str) -> bool:
        """Проверка наличия разрешения у пользователя"""
        users = self.load_users()
        
        if username not in users:
            return False
        
        user = users[username]
        return permission in user.get("permissions", [])
    
    def get_user_role(self, username: str) -> str:
        """Получение роли пользователя"""
        users = self.load_users()
        return users.get(username, {}).get("role", "guest")
    
    def list_users(self) -> List[Dict]:
        """Список всех пользователей (только для админов)"""
        users = self.load_users()
        return [
            {
                "username": username,
                "role": data.get("role", "user"),
                "created_at": data.get("created_at"),
                "last_login": data.get("last_login")
            }
            for username, data in users.items()
        ]
    
    def delete_user(self, username: str) -> bool:
        """Удаление пользователя"""
        users = self.load_users()
        
        if username not in users:
            return False
        
        del users[username]
        self.save_users(users)
        return True
    
    def get_users_dataframe(self) -> Dict:
        """Получение пользователей в формате для DataFrame"""
        users = self.load_users()
        
        data = []
        
        for username, user_data in users.items():
            data.append({
                "👤 Имя пользователя": username,
                "🎭 Роль": user_data.get("role", "user"),
                "📅 Дата создания": user_data.get("created_at", ""),
                "🕒 Последний вход": user_data.get("last_login", "Никогда"),
                "📊 Статус": "🟢 Активен" if user_data.get("last_login") else "⚪ Неактивен"
            })
        
        return data


class StreamlitAuth:
    """Интеграция аутентификации с Streamlit"""
    
    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager
        
        # Инициализация состояния сессии
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_role = "guest"
            st.session_state.login_time = None
            st.session_state.last_activity = datetime.now()
    
    def check_session_timeout(self) -> bool:
        """Проверка таймаута сессии"""
        if not st.session_state.authenticated:
            return False
        
        if "last_activity" not in st.session_state:
            return False
        
        timeout_minutes = self.auth_manager.config["auth"]["session_timeout_minutes"]
        last_activity = st.session_state.last_activity
        
        if isinstance(last_activity, str):
            last_activity = datetime.fromisoformat(last_activity)
        
        elapsed = datetime.now() - last_activity
        return elapsed < timedelta(minutes=timeout_minutes)
    
    def update_activity(self):
        """Обновление времени последней активности"""
        st.session_state.last_activity = datetime.now()
    
    def login_form(self) -> bool:
        """Отображение формы входа"""
        st.title("🔐 Вход в систему")
        
        with st.form("login_form"):
            username = st.text_input("Имя пользователя", placeholder="Введите ваш username")
            password = st.text_input("Пароль", type="password", placeholder="Введите пароль")
            remember_me = st.checkbox("Запомнить меня", value=True)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                submit = st.form_submit_button("Войти", use_container_width=True)
            with col2:
                if st.form_submit_button("Регистрация", use_container_width=True):
                    st.session_state.show_register = True
                    st.rerun()
            with col3:
                if st.form_submit_button("Гость", use_container_width=True):
                    st.session_state.authenticated = True
                    st.session_state.username = "guest"
                    st.session_state.user_role = "guest"
                    st.session_state.login_time = datetime.now()
                    st.rerun()
        
        if submit and username and password:
            authenticated, user = self.auth_manager.authenticate(username, password)
            
            if authenticated:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_role = user.get("role", "user")
                st.session_state.login_time = datetime.now()
                st.session_state.last_activity = datetime.now()
                
                st.success(f"✅ Успешный вход! Добро пожаловать, {username}!")
                st.rerun()
            else:
                st.error("❌ Неверное имя пользователя или пароль")
        
        return st.session_state.authenticated
    
    def register_form(self) -> bool:
        """Форма регистрации нового пользователя"""
        st.title("📝 Регистрация")
        
        with st.form("register_form"):
            st.subheader("Создание аккаунта")
            
            username = st.text_input("Имя пользователя*", 
                                   help="Только латинские буквы, цифры и нижнее подчеркивание")
            email = st.text_input("Email", help="Необязательно")
            password = st.text_input("Пароль*", type="password", 
                                   help="Минимум 8 символов")
            confirm_password = st.text_input("Подтвердите пароль*", type="password")
            
            # Проверка сложности пароля
            if password:
                col1, col2 = st.columns(2)
                with col1:
                    has_length = len(password) >= 8
                    st.markdown(f"{'✅' if has_length else '❌'} Минимум 8 символов")
                with col2:
                    has_digit = any(c.isdigit() for c in password)
                    st.markdown(f"{'✅' if has_digit else '❌'} Содержит цифры")
            
            agree_tos = st.checkbox("Я согласен с условиями использования*", value=False)
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Зарегистрироваться", use_container_width=True)
            with col2:
                if st.form_submit_button("Назад к входу", use_container_width=True):
                    del st.session_state.show_register
                    st.rerun()
        
        if submit:
            # Валидация
            errors = []
            
            if not username or not username.isalnum():
                errors.append("Имя пользователя должно содержать только латинские буквы и цифры")
            
            if len(password) < 8:
                errors.append("Пароль должен содержать минимум 8 символов")
            
            if password != confirm_password:
                errors.append("Пароли не совпадают")
            
            if not agree_tos:
                errors.append("Необходимо согласие с условиями использования")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                success = self.auth_manager.create_user(username, password, role="user")
                
                if success:
                    st.success("✅ Регистрация успешна! Теперь вы можете войти в систему.")
                    st.session_state.show_register = False
                    st.rerun()
                else:
                    st.error("❌ Пользователь с таким именем уже существует")
        
        return False
    
    def logout(self):
        """Выход из системы"""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = "guest"
        st.session_state.login_time = None
        if "show_register" in st.session_state:
            del st.session_state.show_register
        st.rerun()
    
    def show_user_profile(self):
        """Отображение профиля пользователя"""
        if not st.session_state.authenticated:
            return
        
        with st.sidebar.expander("👤 Профиль", expanded=False):
            st.write(f"**Пользователь:** {st.session_state.username}")
            st.write(f"**Роль:** {st.session_state.user_role}")
            
            if st.session_state.login_time:
                if isinstance(st.session_state.login_time, str):
                    login_time = datetime.fromisoformat(st.session_state.login_time)
                else:
                    login_time = st.session_state.login_time
                
                st.write(f"**Вход:** {login_time.strftime('%d.%m.%Y %H:%M')}")
            
            # Кнопка выхода
            if st.button("🚪 Выйти", use_container_width=True):
                self.logout()
    
    def check_auth(self) -> bool:
        """Проверка аутентификации и отображение соответствующих форм"""
        # Обновляем активность
        self.update_activity()
        
        # Проверяем таймаут сессии
        if st.session_state.authenticated and not self.check_session_timeout():
            st.warning("⏰ Время сессии истекло. Пожалуйста, войдите снова.")
            self.logout()
            return False
        
        # Показываем форму регистрации если нужно
        if st.session_state.get("show_register", False):
            return self.register_form()
        
        # Показываем форму входа если не аутентифицированы
        if not st.session_state.authenticated:
            return self.login_form()
        
        return True
    
    def require_auth(self):
        """Декоратор для защиты страниц"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.check_auth():
                    st.stop()
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_permission(self, permission: str):
        """Декоратор для проверки разрешений"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not st.session_state.authenticated:
                    st.error("❌ Требуется аутентификация")
                    st.stop()
                
                if not self.auth_manager.check_permission(st.session_state.username, permission):
                    st.error("⛔ У вас нет прав для выполнения этого действия")
                    st.stop()
                
                return func(*args, **kwargs)
            return wrapper
        return decorator


# Создаем глобальный экземпляр
auth_manager = AuthManager()
streamlit_auth = StreamlitAuth(auth_manager)