"""
Точка входа Streamlit приложения (упрощенная)
"""

import sys
from pathlib import Path

# Подготовка пути к проекту
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Импорт основного модуля UI
from ui.main import main

# Запуск приложения
if __name__ == "__main__":
    main()