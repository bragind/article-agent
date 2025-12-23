#!/usr/bin/env python3
"""
Инициализация системы аутентификации
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auth.auth import AuthManager
import bcrypt

def main():
    """Инициализация аутентификации"""
    print("Инициализация системы аутентификации")
    
    auth_manager = AuthManager()
    
    # Проверяем существование конфига
    if not auth_manager.config_path.exists():
        print("Создаю конфигурационный файл...")
        auth_manager.save_config()
    
    # Создаем администратора
    print("\nСоздание администратора")
    
    username = input("Имя администратора [admin]: ").strip() or "admin"
    password = input("Пароль администратора [admin123]: ").strip() or "admin123"
    
    # Хэшируем пароль
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Создаем пользователя
    success = auth_manager.create_user(username, password, role="admin")
    
    if success:
        print(f"\nАдминистратор создан:")
        print(f"   Имя: {username}")
        print(f"   Пароль: {password}")
        print(f"   Роль: admin")
        print(f"\nСохраните эти данные!")
    else:
        print(f"\nОшибка: Пользователь {username} уже существует")
    
    # Создаем тестового пользователя
    print("\nСоздание тестового пользователя")
    test_user = "test"
    test_pass = "test123"
    
    if auth_manager.create_user(test_user, test_pass, role="user"):
        print(f"Тестовый пользователь создан:")
        print(f"   Имя: {test_user}")
        print(f"   Пароль: {test_pass}")
        print(f"   Роль: user")
    
    print("\nИнициализация завершена!")
    print("Файлы созданы:")
    print(f"  - {auth_manager.config_path}")
    print(f"  - {auth_manager.users_file}")

if __name__ == "__main__":
    main()