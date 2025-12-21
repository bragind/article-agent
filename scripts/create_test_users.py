#!/usr/bin/env python3
"""
Создание тестовых пользователей
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auth.auth import AuthManager

def main():
    """Создание тестовых пользователей"""
    auth = AuthManager()
    
    # Тестовые пользователи
    test_users = [
        {"username": "admin", "password": "admin123", "role": "admin"},
        {"username": "manager", "password": "manager123", "role": "user"},
        {"username": "developer", "password": "dev123", "role": "user"},
        {"username": "analyst", "password": "analyst123", "role": "user"},
        {"username": "guest_user", "password": "guest123", "role": "guest"},
    ]
    
    print("👥 Создание тестовых пользователей")
    print("=" * 50)
    
    created_count = 0
    for user in test_users:
        if auth.create_user(user["username"], user["password"], user["role"]):
            print(f"✅ {user['username']} ({user['role']}) - пароль: {user['password']}")
            created_count += 1
        else:
            print(f"⚠️  {user['username']} уже существует")
    
    print(f"\n🎉 Создано {created_count} пользователей")
    
    # Показываем список всех пользователей
    users = auth.load_users()
    print(f"\n📋 Всего пользователей в системе: {len(users)}")
    for username, data in users.items():
        print(f"  • {username} - {data.get('role', 'user')}")

if __name__ == "__main__":
    main()