# src/bot.py
import os
import logging
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.exceptions import TelegramRetryAfter
from dotenv import load_dotenv

# Импортируем нашу новую RAG систему
from src.core.rag import RAGSystem

# === Настройка ===
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Инициализация ===
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Инициализируем RAG систему с локальной LLM
rag_system = RAGSystem(
    vector_db_path="chroma_full_db",
    llm_model="llama3.2:3b",  # Модель по умолчанию
    llm_base_url="http://localhost:11434"
)

# Проверяем подключение к LLM
llm_status = rag_system.check_llm_connection()
if llm_status["connected"]:
    logger.info(f"LLM подключена. Модель: {llm_status['current_model']}")
else:
    logger.warning("LLM не подключена. Ответы будут ограничены простым поиском.")

# === Обработчики ===
@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "🔍 *Привет! Я AI-агент для интеллектуального поиска технических статей.*\n\n"
        "Я могу:\n"
        "• Искать статьи по IT и техническим темам\n"
        "• Генерировать краткие ответы на основе статей\n"
        "• Предоставлять источники с ссылками\n"
        "• Создавать вопросы для самопроверки\n\n"
        "*Просто отправьте мне любой вопрос, например:*\n"
        "• Как работает RAG?\n"
        "• Что такое квантование в LLM?\n"
        "• Новости про машинное обучение\n"
        "• Объясни архитектуру трансформера\n\n"
        "Используйте /status для проверки состояния системы."
    )

@dp.message(Command("status"))
async def cmd_status(message: Message):
    """Проверка статуса системы"""
    llm_status = rag_system.check_llm_connection()
    
    status_text = "*Статус системы:*\n"
    status_text += f"• LLM подключена: {'✅ Да' if llm_status['connected'] else '❌ Нет'}\n"
    status_text += f"• Модель: {llm_status['current_model']}\n"
    
    if llm_status["available_models"]:
        status_text += f"• Доступные модели: {', '.join(llm_status['available_models'][:3])}\n"
    
    # Можно добавить проверку векторной базы
    try:
        from src.ingest.vector_store import VectorStore
        vector_store = VectorStore(path="chroma_db")
        count = vector_store.count()
        status_text += f"• Статей в базе: {count}\n"
    except:
        status_text += "• Векторная база: требуется проверка\n"
    
    if not llm_status["connected"]:
        status_text += "\n⚠️ *Внимание:* LLM не подключена. Убедитесь, что:\n"
        status_text += "1. Ollama запущен (`ollama serve`)\n"
        status_text += "2. Модель загружена (`ollama pull llama3.2:3b`)\n"
        status_text += "Без LLM ответы будут ограничены простым поиском."
    
    await message.answer(status_text, parse_mode="Markdown")

@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "*Доступные команды:*\n\n"
        "/start - Начало работы\n"
        "/status - Проверка состояния системы\n"
        "/help - Эта справка\n\n"
        "*Примеры запросов:*\n"
        "• Поиск статей по теме\n"
        "• Технические объяснения\n"
        "• IT новости и тренды\n\n"
        "Бот использует локальную LLM через Ollama для генерации умных ответов."
    )

@dp.message()
async def handle_query(message: Message):
    user = message.from_user
    query = message.text.strip()

    if not query:
        await message.answer("Пожалуйста, введите текстовый запрос.")
        return

    logger.info(f"Пользователь {user.id} (@{user.username}): {query[:60]}...")

    try:
        # Отправляем сообщение о начале обработки
        thinking_msg = await message.answer("⏳ Ищу статьи и анализирую...")

        # Используем RAG систему для генерации ответа
        result = rag_system.generate_answer(query, top_k=5)

        # Форматируем ответ
        response = self._format_response(query, result)
        
        # Редактируем исходное сообщение с результатом
        await thinking_msg.edit_text(
            response, 
            parse_mode="Markdown",
            disable_web_page_preview=True
        )

    except TelegramRetryAfter as e:
        await asyncio.sleep(e.retry_after)
        await message.answer("⚠️ Слишком много запросов. Повторите через несколько секунд.")
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса от {user.id}: {e}", exc_info=True)
        await message.answer(
            "❌ Произошла ошибка при обработке запроса. Попробуйте позже или проверьте /status."
        )

def _format_response(self, query: str, result: dict) -> str:
    """Форматирование ответа для Telegram"""
    response_parts = []
    
    # Заголовок с запросом
    response_parts.append(f"*🔍 Запрос:* {query}\n")
    
    # Ответ
    answer = result.get("answer", "Ответ не сгенерирован.")
    response_parts.append(f"*📝 Ответ:*\n{answer}\n")
    
    # Источники
    sources = result.get("sources", [])
    if sources:
        response_parts.append("*📚 Источники:*")
        for i, src in enumerate(sources[:5], 1):  # Ограничиваем 5 источниками
            title = src.get("title", "Без названия")
            url = src.get("url", "")
            author = src.get("author", "")
            
            source_line = f"{i}. {title}"
            if author:
                source_line += f" ({author})"
            if url:
                source_line += f" - [Ссылка]({url})"
            
            response_parts.append(source_line)
    
    # Вопросы для самопроверки
    questions = result.get("questions", [])
    if questions:
        response_parts.append("\n*❓ Вопросы для самопроверки:*")
        for q in questions[:3]:  # Ограничиваем 3 вопросами
            response_parts.append(f"• {q}")
    
    # Похожие статьи
    similar = result.get("similar_articles", [])
    if similar:
        response_parts.append("\n*📖 Похожие статьи:*")
        for article in similar[:3]:
            title = article.get("title", "")
            url = article.get("url", "")
            if title and url:
                response_parts.append(f"• [{title}]({url})")
    
    # Статистика
    stats = result.get("search_stats", {})
    if stats:
        response_parts.append(f"\n_Найдено статей: {stats.get('total_found', 0)}_")
    
    return "\n".join(response_parts)

# Привязываем метод форматирования к обработчику
handle_query._format_response = _format_response.__get__(handle_query)

# === Запуск ===
if __name__ == "__main__":
    logger.info("Запуск Telegram-бота с RAG системой...")
    
    # Дополнительная проверка при запуске
    llm_status = rag_system.check_llm_connection()
    if not llm_status["connected"]:
        logger.warning("""
        ⚠️ ВНИМАНИЕ: LLM не подключена!
        Для полноценной работы выполните:
        1. Установите Ollama: https://ollama.com/
        2. Запустите сервер: ollama serve
        3. Загрузите модель: ollama pull llama3.2:3b
        Без LLM бот будет использовать только простой поиск.
        """)
    
    dp.run_polling(bot)