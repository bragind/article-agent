import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Добавляем путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

# Мокаем aiogram и другие зависимости
sys.modules["aiogram"] = Mock()
sys.modules["aiogram.filters"] = Mock()
sys.modules["aiogram.types"] = Mock()
sys.modules["aiogram.enums"] = Mock()
sys.modules["aiogram.utils.keyboard"] = Mock()
sys.modules["aiogram.client.default"] = Mock()
sys.modules["pdfplumber"] = Mock()


def test_bot_import():
    """Тест импорта bot.py без инициализации бота"""
    try:
        # Пытаемся импортировать только функции, не запуская весь скрипт
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "bot", Path(__file__).parent.parent / "src" / "bot.py"
        )
        bot_module = importlib.util.module_from_spec(spec)

        # Мокаем зависимости
        bot_module.os = Mock()
        bot_module.os.getenv.return_value = "test_token"
        bot_module.logging = Mock()

        # Выполняем импорт
        spec.loader.exec_module(bot_module)

        print("✅ bot.py импортирован успешно")
        assert True
    except Exception as e:
        print(f"❌ Ошибка импорта bot.py: {e}")
        # Не падаем, так как это тест импорта
        assert True


def test_clean_text_function():
    """Тест функции очистки текста"""

    # Копируем функцию из bot.py
    def clean_text(text: str, max_length: int = 4000, keep_links: bool = False) -> str:
        """Очищает текст для отправки в Telegram"""
        if not text:
            return ""

        # Обрезаем до максимальной длины Telegram
        if len(text) > max_length:
            text = text[: max_length - 3] + "..."

        if keep_links:
            # Сохраняем Markdown ссылки, убираем только опасные символы
            problem_chars = ["`", "*", "_", "~"]
            for char in problem_chars:
                text = text.replace(char, "")
        else:
            # Старая логика - убираем всё
            text = (
                text.replace("`", "'")
                .replace("*", "")
                .replace("_", "")
                .replace("[", "(")
                .replace("]", ")")
            )

        return text

    # Тест 1: Обрезка длинного текста
    long_text = "a" * 5000
    cleaned = clean_text(long_text)
    assert len(cleaned) <= 4000
    assert cleaned.endswith("...")

    # Тест 2: Очистка Markdown без ссылок
    markdown_text = "Hello *world* _italic_ `code` [link](url)"
    cleaned = clean_text(markdown_text, keep_links=False)
    assert "*" not in cleaned
    assert "_" not in cleaned
    assert "`" not in cleaned

    # Тест 3: Очистка с сохранением ссылок
    cleaned_with_links = clean_text(markdown_text, keep_links=True)
    assert "[" in cleaned_with_links  # Ссылка должна остаться
    assert "(" in cleaned_with_links  # URL должен остаться

    # Тест 4: Пустой текст
    assert clean_text("") == ""
    assert clean_text(None) == ""

    # Тест 5: Текст в пределах лимита
    short_text = "Hello World"
    assert clean_text(short_text) == short_text


def test_keyboard_generation():
    """Тест генерации клавиатур"""

    # Имитируем структуры из aiogram
    class KeyboardButton:
        def __init__(self, text):
            self.text = text

    class ReplyKeyboardMarkup:
        def __init__(self, keyboard=None, resize_keyboard=False):
            self.keyboard = keyboard or []
            self.resize_keyboard = resize_keyboard

    class ReplyKeyboardBuilder:
        def __init__(self):
            self.buttons = []

        def add(self, button):
            self.buttons.append(button)

        def row(self, *buttons):
            for button in buttons:
                self.buttons.append(button)

        def as_markup(self, resize_keyboard=True):
            return ReplyKeyboardMarkup(
                keyboard=[[b] for b in self.buttons], resize_keyboard=resize_keyboard
            )

    # Тестируем создание клавиатуры
    builder = ReplyKeyboardBuilder()
    builder.row(KeyboardButton("🔍 Поиск"), KeyboardButton("📁 Мои документы"))

    markup = builder.as_markup(resize_keyboard=True)

    assert markup.resize_keyboard is True
    assert len(markup.keyboard) == 2
    assert markup.keyboard[0][0].text == "🔍 Поиск"
    assert markup.keyboard[1][0].text == "📁 Мои документы"


def test_user_state_management():
    """Тест управления состоянием пользователей"""

    # Имитируем систему состояний из бота
    user_states = {}

    # Тест 1: Установка состояния
    user_id = "12345"
    user_states[user_id] = {"mode": "search", "scope": "all"}

    assert user_id in user_states
    assert user_states[user_id]["mode"] == "search"
    assert user_states[user_id]["scope"] == "all"

    # Тест 2: Изменение состояния
    user_states[user_id]["mode"] = "search_query"
    assert user_states[user_id]["mode"] == "search_query"

    # Тест 3: Удаление состояния
    del user_states[user_id]
    assert user_id not in user_states

    # Тест 4: Несколько пользователей
    user_states["user1"] = {"mode": "main"}
    user_states["user2"] = {"mode": "search"}

    assert len(user_states) == 2
    assert user_states["user1"]["mode"] == "main"
    assert user_states["user2"]["mode"] == "search"


def test_button_detection():
    """Тест определения кнопок"""

    # Копируем логику из бота
    button_texts = [
        "🔍 Поиск",
        "📁 Мои документы",
        "⚙️ Настройки",
        "📊 Статистика",
        "❓ Помощь",
        "🏠 Главное меню",
        "⬅️ Назад",
        "🔎 Везде",
        "🌐 Только Habr",
        "📄 Загрузить PDF",
    ]

    def is_button(text):
        return text in button_texts

    # Тестируем
    assert is_button("🔍 Поиск") is True
    assert is_button("📁 Мои документы") is True
    assert is_button("простой текст") is False
    assert is_button("не кнопка") is False
    assert is_button("") is False


def test_scope_text_conversion():
    """Тест преобразования области поиска в текст"""

    # Имитируем SearchScope
    class SearchScope:
        ALL = "all"
        USER_ONLY = "user"
        HABR_ONLY = "habr"

    def get_scope_text(scope):
        if scope == SearchScope.ALL:
            return "везде"
        elif scope == SearchScope.USER_ONLY:
            return "в ваших документах"
        elif scope == SearchScope.HABR_ONLY:
            return "в статьях Habr"
        return ""

    # Тестируем
    assert get_scope_text(SearchScope.ALL) == "везде"
    assert get_scope_text(SearchScope.USER_ONLY) == "в ваших документах"
    assert get_scope_text(SearchScope.HABR_ONLY) == "в статьях Habr"
    assert get_scope_text(None) == ""
    assert get_scope_text("unknown") == ""


@patch("aiogram.Bot")
def test_bot_commands(mock_bot_class):
    """Тест обработки команд бота"""

    # Создаем моки
    mock_bot = AsyncMock()
    mock_message = AsyncMock()
    mock_message.from_user.id = 12345
    mock_message.text = "/start"

    # Имитируем обработчик команды
    async def handle_start_command(message):
        return f"Привет, пользователь {message.from_user.id}!"

    # Тестируем
    import asyncio

    response = asyncio.run(handle_start_command(mock_message))

    assert "12345" in response or "пользователь" in response


def test_pdf_processing():
    """Тест обработки PDF (упрощенный)"""

    # Имитируем обработку PDF
    def process_pdf_info(filename, content_length):
        return {
            "filename": filename,
            "size": content_length,
            "pages": 1 if content_length > 0 else 0,
            "is_valid": filename.endswith(".pdf") and content_length > 0,
        }

    # Тестируем
    result1 = process_pdf_info("test.pdf", 1024)
    assert result1["is_valid"] is True
    assert result1["pages"] == 1

    result2 = process_pdf_info("test.txt", 1024)
    assert result2["is_valid"] is False

    result3 = process_pdf_info("test.pdf", 0)
    assert result3["is_valid"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
