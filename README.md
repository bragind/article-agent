# AI-агент для интеллектуального поиска статей

> **MVP на основе RAG** для поиска, анализа и аннотирования IT-статей из открытых источников (Habr, CNews, Rusbase и др.).

Основной интерфейс — **Telegram-бот**.  
Дополнительно: веб-демо через Streamlit.

---

## 🎯 Возможности

- Поиск статей по техническим запросам
- Генерация краткого аннотированного ответа
- Указание источников с ссылками на оригиналы
- Генерация вопросов для самопроверки
- Поддержка русского и английского языков
- Этичность: точность, прозрачность, защита данных

---

## 🧠 Архитектура

- Пользователь → Telegram-бот → RAG-ядро → ChromaDB + LLM → Ответ с источниками

- **Data Ingestion**: сбор статей → `data/articles.jsonl`
- **Indexing**: эмбеддинги (`intfloat/multilingual-e5-large`) → ChromaDB
- **RAG Pipeline**: retrieval + промптинг → генерация ответа
- **Serving**: 
  - Основное: `src/bot.py` (Telegram, aiogram)
  - Демо: `src/app.py` (Streamlit)

---

## 🚀 Быстрый старт

### 1. Клонируйте репозиторий
```bash
git clone https://github.com/bragind/article-agent.git
cd article-agent
git lfs install
git lfs pull

# 2. Создайте окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📁 Данные

- `data/articles_batch.jsonl` - 700 статей (Git LFS, 15MB)
- Формат: JSONL с полями `title`, `text`, `author`, `url`, `date`, `source`

## 🧠 Архитектура

```
Пользователь → Telegram/Web → RAG → Векторный поиск (ChromaDB) → LLM → Ответ с источниками
```

## 🔧 Парсинг статей

```bash
# Тест (5 статей)
python parser_articles/main.py --limit 5

# Полный сбор (700 статей)
python parser_articles/main.py --all
```

## ⚙️ Конфигурация

Настройки в `parser_articles/parsers/config.py`:
- `MAX_ARTICLES = 700` - сколько статей собирать
- `BASE_DELAY = 3.0` - задержка между запросами
- `SECTIONS` - источники для парсинга

## 📞 Контакты

Issues: [GitHub](https://github.com/bragind/article-agent/issues)  
Telegram: @bragind

*Проект использует Git LFS для хранения больших файлов*