"""
Пакет парсеров для проекта article-agent
"""

from .habr_parser import HabrParser
from .tproger_parser import TprogerParser
from .config import (
    MAX_ARTICLES, ARTICLES_PER_PAGE, PAGES_TO_CHECK,
    ARTICLES_PER_SECTION, BASE_DELAY, MAX_RANDOM_DELAY,
    ERROR_DELAY, SECTIONS
)

__all__ = [
    'HabrParser',
    'TprogerParser',
    'MAX_ARTICLES',
    'ARTICLES_PER_PAGE',
    'PAGES_TO_CHECK',
    'ARTICLES_PER_SECTION',
    'BASE_DELAY',
    'MAX_RANDOM_DELAY',
    'ERROR_DELAY',
    'SECTIONS'
]