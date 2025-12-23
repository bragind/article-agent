"""
Пакет UI модулей
"""

from .main import main, StreamlitApp
from .auth import AuthManager
from .components import UIComponents
from .search_tab import SearchTab
from .documents_tab import DocumentsTab
from .statistics_tab import StatisticsTab
from .admin_tab import AdminTab
from . import helpers
from . import constants

__all__ = [
    'main',
    'StreamlitApp',
    'AuthManager',
    'UIComponents',
    'SearchTab',
    'DocumentsTab',
    'StatisticsTab',
    'AdminTab',
    'helpers',
    'constants'
]