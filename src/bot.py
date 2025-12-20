#!/usr/bin/env python3
"""
Telegram бот для RAG-агента с поддержкой выбора области поиска
"""

import os
import logging
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib

# ==================== НАСТРОЙКИ ПУТЕЙ И ИМПОРТОВ ====================

# Определяем корень проекта
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.core.rag import RAGAgent, SearchScope
    RAG_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"❌ Ошибка импорта RAGAgent: {e}")
    RAG_IMPORT_SUCCESS = False

# Настройка для Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ==================== КОНФИГУРАЦИЯ ====================

# Загружаем переменные окружения
from dotenv import load_dotenv
load_dotenv()

# Получаем токен бота
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    print("❌ ОШИБКА: TELEGRAM_BOT_TOKEN не найден в .env файле")
    print("Создайте файл .env с содержимым:")
    print("TELEGRAM_BOT_TOKEN=ваш_токен_здесь")
    sys.exit(1)

# ==================== НАСТРОЙКА ЛОГИРОВАНИЯ ====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== ИНИЦИАЛИЗАЦИЯ БОТА ====================

try:
    from aiogram import Bot, Dispatcher, F
    from aiogram.filters import Command, CommandStart
    from aiogram.types import (
        Message,
        InlineKeyboardMarkup,
        InlineKeyboardButton,
        CallbackQuery,
        ReplyKeyboardMarkup,
        KeyboardButton,
        Document
    )
    from aiogram.client.default import DefaultBotProperties
    from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
    from aiogram.enums import ParseMode
    from aiogram.exceptions import TelegramNetworkError, TelegramBadRequest
    
    AIOGRAM_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"❌ Ошибка импорта aiogram: {e}")
    print("Установите aiogram: pip install aiogram")
    AIOGRAM_IMPORT_SUCCESS = False
    sys.exit(1)

# Инициализация бота
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
dp = Dispatcher()

# Инициализация RAG агента
try:
    data_dir = PROJECT_ROOT / "data"
    rag_agent = RAGAgent(data_dir=str(data_dir))
    logger.info("✅ RAG агент инициализирован")
except Exception as e:
    logger.error(f"❌ Ошибка инициализации RAG агента: {e}")
    rag_agent = None

# Состояния пользователей
user_states: Dict[str, Dict[str, Any]] = {}

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def clean_text_for_telegram(text: str, max_length: int = 4000, keep_links: bool = True) -> str:
    """
    Очищает текст для отправки в Telegram
    
    Args:
        text: Текст для очистки
        max_length: Максимальная длина текста
        keep_links: Сохранять ли Markdown ссылки
        
    Returns:
        Очищенный текст
    """
    if not text:
        return ""
    
    # Обрезаем до максимальной длины
    if len(text) > max_length:
        text = text[:max_length - 3] + "..."
    
    if keep_links:
        # Убираем только потенциально проблемные символы для Markdown
        problem_chars = ["`", "*", "_", "~"]
        for char in problem_chars:
            text = text.replace(char, "")
    else:
        # Полная очистка
        text = (
            text.replace("`", "'")
            .replace("*", "")
            .replace("_", "")
            .replace("[", "(")
            .replace("]", ")")
            .replace("~", "")
        )
    
    return text.strip()

def get_scope_text(scope: SearchScope) -> str:
    """Текстовое представление области поиска"""
    if scope == SearchScope.ALL:
        return "везде"
    elif scope == SearchScope.USER_ONLY:
        return "в ваших документах"
    elif scope == SearchScope.HABR_ONLY:
        return "в статьях Habr"
    return ""

def get_user_state(user_id: str) -> Dict[str, Any]:
    """Получает состояние пользователя с дефолтными значениями"""
    if user_id not in user_states:
        user_states[user_id] = {
            "mode": "main",
            "scope": SearchScope.ALL,
            "last_query": "",
            "search_results": []
        }
    return user_states[user_id]

def update_user_state(user_id: str, **kwargs):
    """Обновляет состояние пользователя"""
    state = get_user_state(user_id)
    state.update(kwargs)
    user_states[user_id] = state

# ==================== КЛАВИАТУРЫ ====================

def create_main_keyboard() -> ReplyKeyboardMarkup:
    """Создает основную клавиатуру"""
    builder = ReplyKeyboardBuilder()
    
    builder.row(
        KeyboardButton(text="🔍 Поиск"),
        KeyboardButton(text="📁 Мои документы")
    )
    
    builder.row(
        KeyboardButton(text="⚙️ Настройки"),
        KeyboardButton(text="📊 Статистика")
    )
    
    builder.row(
        KeyboardButton(text="❓ Помощь"),
        KeyboardButton(text="🏠 Главное меню")
    )
    
    return builder.as_markup(resize_keyboard=True)

def create_search_keyboard() -> ReplyKeyboardMarkup:
    """Создает клавиатуру для поиска"""
    builder = ReplyKeyboardBuilder()
    
    builder.row(
        KeyboardButton(text="🔎 Везде"),
        KeyboardButton(text="📁 Мои документы"),
        KeyboardButton(text="🌐 Только Habr"),
    )
    
    builder.row(
        KeyboardButton(text="📄 Загрузить PDF"),
        KeyboardButton(text="⬅️ Назад")
    )
    
    return builder.as_markup(resize_keyboard=True)

def create_back_keyboard() -> ReplyKeyboardMarkup:
    """Создает клавиатуру с кнопкой назад"""
    builder = ReplyKeyboardBuilder()
    builder.row(KeyboardButton(text="⬅️ Назад"))
    return builder.as_markup(resize_keyboard=True)

def create_action_keyboard() -> InlineKeyboardMarkup:
    """Создает inline клавиатуру для действий после поиска"""
    builder = InlineKeyboardBuilder()
    
    builder.row(
        InlineKeyboardButton(text="🔍 Новый поиск", callback_data="new_search"),
        InlineKeyboardButton(text="📄 Загрузить PDF", callback_data="upload_pdf"),
    )
    
    builder.row(InlineKeyboardButton(text="🏠 Главное меню", callback_data="main_menu"))
    
    return builder.as_markup()

# ==================== ОБРАБОТЧИКИ КОМАНД ====================

@dp.message(CommandStart())
async def cmd_start(message: Message):
    """Обработчик команды /start"""
    user_id = str(message.from_user.id)
    
    # Сбрасываем состояние пользователя
    update_user_state(user_id, mode="main", scope=SearchScope.ALL)
    
    welcome_text = clean_text_for_telegram(
        "🤖 Добро пожаловать в AI RAG-агент!\n\n"
        "📚 **Возможности:**\n"
        "• Интеллектуальный поиск статей\n"
        "• Анализ PDF документов\n"
        "• Выбор области поиска\n"
        "• Краткие резюме с источниками\n\n"
        "⚡ **Используйте кнопки ниже:**\n"
        "🔍 **Поиск** - начать поиск\n"
        "📁 **Мои документы** - ваши PDF\n"
        "📊 **Статистика** - информация о системе\n\n"
        "👇 Выберите действие:"
    )
    
    await message.answer(welcome_text, reply_markup=create_main_keyboard())
    logger.info(f"Пользователь {user_id} запустил бота")

@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Обработчик команды /help"""
    help_text = clean_text_for_telegram(
        "🛠 **Помощь по использованию бота:**\n\n"
        "🔍 **Поиск информации:**\n"
        "1. Нажмите '🔍 Поиск'\n"
        "2. Выберите область поиска:\n"
        "   • 🔎 **Везде** - искать везде\n"
        "   • 📁 **Мои** - только ваши PDF\n"
        "   • 🌐 **Habr** - только статьи\n"
        "3. Введите запрос\n\n"
        "📄 **Загрузка PDF:**\n"
        "1. Нажмите '📄 Загрузить PDF'\n"
        "2. Отправьте PDF файл\n"
        "3. Документ добавится в поиск\n\n"
        "📊 **Статистика:**\n"
        "• Количество ваших документов\n"
        "• Всего статей в системе\n\n"
        "❓ **Примеры запросов:**\n"
        "• RAG архитектура\n"
        "• Python программирование\n"
        "• Машинное обучение\n\n"
        "📞 **Поддержка:** @support_bot"
    )
    
    await message.answer(help_text, reply_markup=create_main_keyboard())

@dp.message(Command("search"))
async def cmd_search(message: Message):
    """Обработчик команды /search"""
    user_id = str(message.from_user.id)
    update_user_state(user_id, mode="search")
    
    await message.answer(
        clean_text_for_telegram(
            "🔍 **Выберите область поиска:**\n\n"
            "• 🔎 **Везде** - поиск по всем источникам\n"
            "• 📁 **Мои документы** - только ваши PDF\n"
            "• 🌐 **Только Habr** - только статьи Habr\n\n"
            "Или просто введите запрос для поиска везде."
        ),
        reply_markup=create_search_keyboard()
    )

@dp.message(Command("docs"))
async def cmd_docs(message: Message):
    """Обработчик команды /docs"""
    user_id = str(message.from_user.id)
    
    if not rag_agent:
        await message.answer("❌ RAG агент не инициализирован")
        return
    
    user_docs = rag_agent.get_user_articles(user_id)
    
    if not user_docs:
        await message.answer(
            clean_text_for_telegram(
                "📭 **У вас пока нет документов**\n\n"
                "Чтобы добавить документы:\n"
                "1. Нажмите '📄 Загрузить PDF'\n"
                "2. Отправьте PDF файл\n\n"
                "💡 После загрузки вы сможете искать по своим документам!"
            ),
            reply_markup=create_search_keyboard()
        )
        update_user_state(user_id, mode="search")
    else:
        response = f"📚 **Ваши документы ({len(user_docs)}):**\n\n"
        
        for i, doc in enumerate(user_docs[:5], 1):
            title = clean_text_for_telegram(doc.get("title", "Без названия")[:40], keep_links=False)
            chars = doc.get("text_length", 0)
            date = doc.get("uploaded_at", "")[:10]
            
            response += f"{i}. {title}\n"
            response += f"   📄 {chars:,} символов\n"
            if date:
                response += f"   📅 {date}\n"
            response += "\n"
        
        if len(user_docs) > 5:
            response += f"... и еще {len(user_docs) - 5} документов\n\n"
        
        response += "💡 Используйте '📁 Мои документы' для поиска по ним."
        
        await message.answer(clean_text_for_telegram(response), reply_markup=create_search_keyboard())
        update_user_state(user_id, mode="docs")

@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    """Обработчик команды /stats"""
    user_id = str(message.from_user.id)
    
    if not rag_agent:
        await message.answer("❌ RAG агент не инициализирован")
        return
    
    stats = rag_agent.get_statistics(user_id)
    
    response = clean_text_for_telegram(
        "📊 **Статистика системы:**\n\n"
        "👤 **Ваши данные:**\n"
        f"• Документов: {stats['current_user_articles']}\n\n"
        "🌐 **Общая база:**\n"
        f"• Всего статей: {stats['total_articles']:,}\n"
        f"• Статей Habr: {stats['habr_articles']:,}\n"
        f"• Пользовательских: {stats['user_articles']:,}\n\n"
        "🔄 **Обновлено:**\n"
        f"{stats['last_update'][:19]}\n"
    )
    
    await message.answer(response, reply_markup=create_main_keyboard())

# ==================== ОБРАБОТЧИКИ КНОПОК ====================

@dp.message(F.text == "🔍 Поиск")
async def handle_search_button(message: Message):
    await cmd_search(message)

@dp.message(F.text == "📁 Мои документы")
async def handle_docs_button(message: Message):
    await cmd_docs(message)

@dp.message(F.text == "⚙️ Настройки")
async def handle_settings_button(message: Message):
    await message.answer(
        clean_text_for_telegram(
            "⚙️ **Настройки поиска:**\n\n"
            "Используйте кнопки внизу для навигации."
        ),
        reply_markup=create_main_keyboard()
    )

@dp.message(F.text == "📊 Статистика")
async def handle_stats_button(message: Message):
    await cmd_stats(message)

@dp.message(F.text == "❓ Помощь")
async def handle_help_button(message: Message):
    await cmd_help(message)

@dp.message(F.text == "🏠 Главное меню")
async def handle_main_menu_button(message: Message):
    user_id = str(message.from_user.id)
    update_user_state(user_id, mode="main")
    
    await message.answer(
        clean_text_for_telegram("🏠 **Главное меню**\n\nВыберите действие:"),
        reply_markup=create_main_keyboard()
    )

@dp.message(F.text == "⬅️ Назад")
async def handle_back_button(message: Message):
    await cmd_start(message)

@dp.message(F.text == "🔎 Везде")
async def handle_search_everywhere(message: Message):
    user_id = str(message.from_user.id)
    update_user_state(user_id, mode="search_query", scope=SearchScope.ALL)
    
    await message.answer(
        clean_text_for_telegram(
            "🔎 **Поиск везде**\n\n"
            "Я буду искать:\n"
            "• В ваших документах\n"
            "• В статьях Habr\n"
            "• Во всей базе знаний\n\n"
            "📝 Введите ваш запрос:"
        ),
        reply_markup=create_back_keyboard()
    )

@dp.message(F.text == "📁 Мои документы")
async def handle_search_user_only(message: Message):
    user_id = str(message.from_user.id)
    
    if not rag_agent:
        await message.answer("❌ RAG агент не инициализирован")
        return
    
    # Проверяем, есть ли документы
    user_docs = rag_agent.get_user_articles(user_id)
    
    if not user_docs:
        await message.answer(
            clean_text_for_telegram(
                "📭 **У вас пока нет документов**\n\n"
                "Сначала загрузите PDF файлы:\n"
                "1. Нажмите '📄 Загрузить PDF'\n"
                "2. Отправьте файл\n\n"
                "Или выберите другую область поиска."
            ),
            reply_markup=create_search_keyboard()
        )
        update_user_state(user_id, mode="search")
    else:
        update_user_state(user_id, mode="search_query", scope=SearchScope.USER_ONLY)
        
        await message.answer(
            clean_text_for_telegram(
                f"📁 **Поиск в ваших документах**\n\n"
                f"У вас {len(user_docs)} документов.\n"
                f"Я буду искать только в них.\n\n"
                f"📝 Введите ваш запрос:"
            ),
            reply_markup=create_back_keyboard()
        )

@dp.message(F.text == "🌐 Только Habr")
async def handle_search_habr_only(message: Message):
    user_id = str(message.from_user.id)
    update_user_state(user_id, mode="search_query", scope=SearchScope.HABR_ONLY)
    
    if not rag_agent:
        await message.answer("❌ RAG агент не инициализирован")
        return
    
    await message.answer(
        clean_text_for_telegram(
            "🌐 **Поиск в статьях Habr**\n\n"
            "Я буду искать только в статьях с Habr.\n"
            f"Доступно статей: {len(rag_agent.habr_articles):,}\n\n"
            "📝 Введите ваш запрос:"
        ),
        reply_markup=create_back_keyboard()
    )

@dp.message(F.text == "📄 Загрузить PDF")
async def handle_upload_pdf(message: Message):
    user_id = str(message.from_user.id)
    update_user_state(user_id, mode="upload_pdf")
    
    await message.answer(
        clean_text_for_telegram(
            "📄 **Загрузка PDF документа**\n\n"
            "Отправьте мне PDF файл как документ.\n\n"
            "Что произойдет:\n"
            "1. Файл будет обработан\n"
            "2. Текст будет извлечен\n"
            "3. Документ добавится в поиск\n"
            "4. Вы сможете искать по нему\n\n"
            "⚠️ **Ограничения:**\n"
            "• Максимальный размер: 50MB\n"
            "• Только PDF формат\n\n"
            "📎 Отправьте PDF файл:"
        ),
        reply_markup=create_back_keyboard()
    )

# ==================== ОБРАБОТЧИК ТЕКСТОВЫХ СООБЩЕНИЙ ====================

@dp.message(F.text)
async def handle_text_message(message: Message):
    """Главный обработчик всех текстовых сообщений"""
    user_id = str(message.from_user.id)
    user_text = message.text.strip()
    
    if not user_text:
        return
    
    # Игнорируем текст, который является кнопками
    button_texts = [
        "🔍 Поиск", "📁 Мои документы", "⚙️ Настройки", "📊 Статистика",
        "❓ Помощь", "🏠 Главное меню", "⬅️ Назад", "🔎 Везде",
        "🌐 Только Habr", "📄 Загрузить PDF"
    ]
    
    if user_text in button_texts:
        return
    
    # Получаем текущее состояние пользователя
    state = get_user_state(user_id)
    mode = state.get("mode", "main")
    
    if not rag_agent:
        await message.answer("❌ RAG агент не инициализирован. Попробуйте позже.")
        return
    
    # Если пользователь в режиме поиска (уже выбрал область)
    if mode == "search_query":
        scope = state.get("scope", SearchScope.ALL)
        
        try:
            # Уведомляем о начале поиска
            searching_msg = await message.answer(f"🔍 Ищу {get_scope_text(scope)}...")
            
            # Выполняем поиск
            result = rag_agent.generate_answer(user_text, user_id, scope)
            
            # Формируем ответ
            response = clean_text_for_telegram(result["answer"], keep_links=True)
            
            # Добавляем источники
            if result.get("sources"):
                response += "\n\n📚 **Источники:**\n"
                
                for i, source in enumerate(result["sources"][:3], 1):
                    icon = "📁" if source.get("is_user_document") else "🌐"
                    title = clean_text_for_telegram(
                        source["title"][:60] + "..." if len(source["title"]) > 60 else source["title"],
                        keep_links=False
                    )
                    url = source["url"]
                    
                    if url.startswith("file://"):
                        response += f"{i}. {icon} **{title}** (ваш документ)\n"
                    elif url == "#" or not url:
                        response += f"{i}. {icon} **{title}**\n"
                    else:
                        # Экранируем спецсимволы для Markdown
                        safe_url = url.replace(")", "%29").replace("(", "%28")
                        response += f"{i}. {icon} [{title}]({safe_url})\n"
            
            # Обновляем сообщение
            await searching_msg.edit_text(
                response,
                reply_markup=create_action_keyboard(),
                disable_web_page_preview=False
            )
            
            # Сохраняем результаты
            update_user_state(user_id, search_results=result.get("sources", []))
            
        except Exception as e:
            logger.error(f"Ошибка поиска для пользователя {user_id}: {e}")
            error_msg = clean_text_for_telegram(str(e)[:100], keep_links=False)
            await message.answer(
                f"❌ **Ошибка при поиске**\n\n"
                f"Произошла ошибка: {error_msg}\n\n"
                f"Попробуйте еще раз.",
                reply_markup=create_action_keyboard()
            )
    
    # Если пользователь в главном меню и пишет запрос
    elif mode == "main":
        # Сохраняем запрос и переходим в режим поиска
        update_user_state(user_id, mode="search", last_query=user_text)
        
        await message.answer(
            clean_text_for_telegram(
                f"🔍 **Запрос:** {user_text}\n\n"
                "**Выберите область поиска:**\n"
                "• 🔎 Везде - поиск по всем источникам\n"
                "• 📁 Мои документы - только ваши PDF\n"
                "• 🌐 Только Habr - только статьи Habr\n\n"
                "Или введите новый запрос."
            ),
            reply_markup=create_search_keyboard()
        )
    
    # Если пользователь просто ввел запрос без выбора области
    else:
        # По умолчанию ищем везде
        try:
            searching_msg = await message.answer("🔍 Ищу везде...")
            
            result = rag_agent.generate_answer(user_text, user_id, SearchScope.ALL)
            
            # Формируем ответ
            response = clean_text_for_telegram(result["answer"], keep_links=True)
            
            if result.get("sources"):
                response += "\n\n📚 **Источники:**\n"
                for i, source in enumerate(result["sources"][:3], 1):
                    icon = "📁" if source.get("is_user_document") else "🌐"
                    title = clean_text_for_telegram(
                        source["title"][:60] + "..." if len(source["title"]) > 60 else source["title"],
                        keep_links=False
                    )
                    url = source["url"]
                    
                    if url.startswith("file://"):
                        response += f"{i}. {icon} **{title}** (ваш документ)\n"
                    elif url == "#" or not url:
                        response += f"{i}. {icon} **{title}**\n"
                    else:
                        safe_url = url.replace(")", "%29").replace("(", "%28")
                        response += f"{i}. {icon} [{title}]({safe_url})\n"
            
            await searching_msg.edit_text(
                response,
                reply_markup=create_action_keyboard(),
                disable_web_page_preview=False
            )
            
            update_user_state(user_id, mode="search_results")
            
        except Exception as e:
            logger.error(f"Ошибка поиска для пользователя {user_id}: {e}")
            await message.answer(
                f"❌ **Ошибка при поиске**\n\n"
                f"Произошла ошибка: {clean_text_for_telegram(str(e)[:100], keep_links=False)}",
                reply_markup=create_action_keyboard()
            )

# ==================== ОБРАБОТЧИКИ CALLBACK-ЗАПРОСОВ ====================

@dp.callback_query(F.data == "new_search")
async def handle_new_search_callback(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    update_user_state(user_id, mode="search")
    
    await callback.message.edit_text("🔍 Начинаем новый поиск...", reply_markup=None)
    await callback.message.answer(
        "Выберите область поиска:",
        reply_markup=create_search_keyboard()
    )
    await callback.answer()

@dp.callback_query(F.data == "upload_pdf")
async def handle_upload_pdf_callback(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    update_user_state(user_id, mode="upload_pdf")
    
    await callback.message.edit_text("📄 Загрузка PDF", reply_markup=None)
    await callback.message.answer(
        "📎 Отправьте PDF файл:",
        reply_markup=create_back_keyboard()
    )
    await callback.answer()

@dp.callback_query(F.data == "main_menu")
async def handle_main_menu_callback(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    update_user_state(user_id, mode="main")
    
    await callback.message.edit_text(
        "🏠 Возвращаюсь в главное меню...",
        reply_markup=None
    )
    await callback.message.answer(
        "🏠 Главное меню\n\nВыберите действие:",
        reply_markup=create_main_keyboard()
    )
    await callback.answer()

# ==================== ОБРАБОТЧИК ДОКУМЕНТОВ ====================

@dp.message(F.document)
async def handle_document(message: Message):
    """Обработка загруженных PDF документов"""
    user_id = str(message.from_user.id)
    document = message.document
    
    # Проверяем, что это PDF
    if not document.file_name or not document.file_name.lower().endswith(".pdf"):
        await message.answer(
            clean_text_for_telegram(
                "❌ **Неверный формат файла**\n\n"
                "Поддерживаются только PDF файлы."
            ),
            reply_markup=create_search_keyboard()
        )
        return
    
    # Проверяем размер (50MB)
    if document.file_size > 50 * 1024 * 1024:
        await message.answer(
            clean_text_for_telegram(
                "❌ **Файл слишком большой**\n\n"
                "Максимальный размер: 50MB"
            ),
            reply_markup=create_search_keyboard()
        )
        return
    
    processing_msg = await message.answer("📥 Обрабатываю PDF...")
    
    try:
        # Скачиваем файл
        file = await bot.get_file(document.file_id)
        file_path = file.file_path
        
        # Создаем временный файл
        temp_file = f"temp_{user_id}_{document.file_name}"
        
        # Скачиваем файл
        await bot.download_file(file_path, temp_file)
        
        # Читаем файл
        with open(temp_file, 'rb') as f:
            pdf_bytes = f.read()
        
        # Удаляем временный файл
        os.remove(temp_file)
        
        # Извлекаем текст из PDF
        import pdfplumber
        from io import BytesIO
        
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            full_text = ""
            pages_data = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                full_text += f"\n\n--- Страница {page_num} ---\n{page_text}"
                
                pages_data.append({
                    "page_number": page_num,
                    "text": page_text,
                    "char_count": len(page_text),
                })
        
        # Создаем статью
        article_data = {
            "title": clean_text_for_telegram(
                document.file_name.replace(".pdf", "").replace("_", " ").title(),
                keep_links=False
            ),
            "author": "",
            "date": datetime.now().isoformat(),
            "text": full_text.strip(),
            "tags": [],
            "url": f"file://{document.file_name}",
            "source": "User Upload",
            "views": 0,
            "rating": 0,
            "parsed_at": datetime.now().isoformat(),
            "text_length": len(full_text.strip()),
            "has_content": len(full_text.strip()) > 100,
            "pages": pages_data,
            "uploaded_by": user_id,
        }
        
        # Добавляем в RAG агент
        if not rag_agent:
            await processing_msg.edit_text("❌ RAG агент не инициализирован")
            return
        
        if article_data["has_content"]:
            article_id = rag_agent.add_user_article(article_data, user_id)
            
            if article_id:
                response = clean_text_for_telegram(
                    f"✅ **PDF успешно обработан!**\n\n"
                    f"📄 **Документ:** {article_data['title']}\n"
                    f"📊 **Страниц:** {len(pages_data)}\n"
                    f"🔤 **Символов:** {article_data['text_length']:,}\n\n"
                    f"💡 Теперь вы можете искать по этому документу."
                )

                # Добавляем теги если есть
                if article_data.get("tags"):
                    tags_text = ", ".join([f"`{tag}`" for tag in article_data["tags"][:5]])
                    response += f"\n🏷️ **Теги:** {tags_text}"
            else:
                response = "❌ **Ошибка при добавлении документа**"
        else:
            response = clean_text_for_telegram(
                "⚠️ **В документе недостаточно текста**\n\n"
                "Попробуйте другой документ."
            )
        
        await processing_msg.edit_text(response, reply_markup=create_action_keyboard())
        update_user_state(user_id, mode="search_results")
        
    except Exception as e:
        logger.error(f"Ошибка обработки PDF для пользователя {user_id}: {e}")
        error_msg = clean_text_for_telegram(str(e)[:150], keep_links=False)
        await processing_msg.edit_text(
            clean_text_for_telegram(
                f"❌ **Ошибка обработки PDF**\n\n"
                f"Произошла ошибка: {error_msg}"
            ),
            reply_markup=create_action_keyboard()
        )

# ==================== ЗАПУСК БОТА ====================

async def main():
    """Основная функция запуска бота"""
    logger.info("🤖 Запуск Telegram бота...")
    
    try:
        # Получаем информацию о боте
        bot_info = await bot.get_me()
        print(f"\n{'='*50}")
        print(f"🤖 Бот: @{bot_info.username}")
        print(f"📛 Имя: {bot_info.first_name}")
        print(f"🆔 ID: {bot_info.id}")
        print(f"🔗 Ссылка: https://t.me/{bot_info.username}")
        print(f"{'='*50}\n")
        
        if not rag_agent:
            print("⚠️ ВНИМАНИЕ: RAG агент не инициализирован!")
            print("Бот будет работать, но поиск недоступен.")
        
        print("✅ Бот запущен и слушает сообщения...")
        print("⏳ Для остановки нажмите Ctrl+C\n")
        
        # Запускаем бота
        await dp.start_polling(bot)
        
    except Exception as e:
        logger.error(f"❌ Ошибка запуска бота: {e}")
        print(f"\n❌ Критическая ошибка: {e}")
    finally:
        # Закрываем сессию бота
        await bot.session.close()
        logger.info("👋 Бот остановлен")

if __name__ == "__main__":
    # Проверяем импорты
    if not RAG_IMPORT_SUCCESS:
        print("❌ Не удалось импортировать RAG модули")
        sys.exit(1)
    
    if not AIOGRAM_IMPORT_SUCCESS:
        print("❌ Не удалось импортировать aiogram")
        sys.exit(1)
    
    # Запускаем бота
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен пользователем")
    except Exception as e:
        print(f"\n💥 Непредвиденная ошибка: {e}")