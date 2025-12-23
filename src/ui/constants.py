"""
Константы и настройки приложения
"""

# Настройки страницы
PAGE_CONFIG = {
    "page_title": "AI RAG-агент",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Области поиска
SEARCH_SCOPE_MAPPING = {
    "all": "🔎 Везде",
    "user": "📁 Только мои документы", 
    "habr": "🌐 Только Habr"
}

# Бейджи релевантности
RELEVANCE_BADGES = {
    "high": {"text": "🟢 Высокая", "threshold": 0.8},
    "medium": {"text": "🟡 Средняя", "threshold": 0.5},
    "low": {"text": "🔵 Низкая", "threshold": 0.0}
}

# Типы источников
SOURCE_TYPES = {
    "user": "📁 Ваш документ",
    "habr": "🌐 Статья Habr"
}

# Роли пользователей
USER_ROLES = {
    "admin": "👑 Администратор",
    "user": "👤 Пользователь", 
    "guest": "👥 Гость"
}

# Цвета для графиков
CHART_COLORS = {
    "pie": "Set3",
    "bar_viridis": "viridis",
    "bar_plasma": "plasma",
    "line": "royalblue"
}