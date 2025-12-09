#!/usr/bin/env python3
"""
Скрипт для запуска валидации данных
"""

import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_data import main

if __name__ == "__main__":
    main()