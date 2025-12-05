# src/bot.py
import os
import logging
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.exceptions import TelegramRetryAfter
from dotenv import load_dotenv

from rag import RAGAgent

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
rag_agent = RAGAgent()

# === Обработчики ===
@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "🔍 Привет! Я AI-агент для поиска статей по IT и техническим темам.\n\n"
        "Отправьте любой запрос, например:\n"
        "• Как работает RAG?\n"
        "• Что такое квантование LLM?\n"
        "• Объясни архитектуру трансформера\n\n"
        "Я найду релевантные материалы и дам краткий ответ с источниками."
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
        thinking = await message.answer("⏳ Ищу и анализирую статьи...")
        result = rag_agent.generate_answer(query)

        # Формирование ответа
        answer = result.get("answer", "Ответ не сгенерирован.")
        sources = "\n".join(
            f"• [{src['title']}]({src['url']})" for src in result.get("sources", [])
        ) or "Источники не найдены."
        questions = "\n".join(
            f"• {q.strip()}" for q in result.get("questions", []) if q.strip()
        ) or "Нет вопросов."

        response = (
            f"**Ответ:**\n{answer}\n\n"
            f"**Источники:**\n{sources}\n\n"
            f"**Вопросы для самопроверки:**\n{questions}"
        )

        await thinking.edit_text(response, parse_mode="Markdown", disable_web_page_preview=True)

    except TelegramRetryAfter as e:
        await asyncio.sleep(e.retry_after)
        await message.answer("⚠️ Слишком много запросов. Повторите через несколько секунд.")
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса от {user.id}: {e}")
        await message.answer("❌ Произошла ошибка. Попробуйте позже.")

# === Запуск ===
if __name__ == "__main__":
    logger.info("Запуск Telegram-бота...")
    dp.run_polling(bot)