
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

### 2. Настройте окружение
'''bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# или
venv\Scripts\activate        # Windows

### 3. Установите зависимости
'''bash
pip install -r requirements.

### 4. Подготовьте данные и индекс
'''bash
# Добавьте статьи в data/articles.jsonl (формат JSONL)
python scripts/build_index.py

### 5. Настройте Telegram-бота
- Создайте бота в @BotFather
- Получите токен
- Создайте файл .env
TELEGRAM_BOT_TOKEN=ваш_токен_здесь

### 6. Запустите
'''bash
# Основной режим — Telegram-бот
python src/bot.py

# Или демо — Streamlit
streamlit run src/app.py

### Docker
'''bash
# Сборка
docker build -t article-agent .

# Запуск бота
docker run --env-file .env article-agent

# Запуск Streamlit
docker run -p 8501:8501 article-agent streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0

### MLOps и CI/CD
- Версионирование кода (Git)
- Тестирование (pytest)
- Автоматическая сборка и публикация Docker-образа (GitHub Actions)
- Логирование (logs/bot.log)
- Конфигурация через config/config.yaml

### Структура проекта

article-agent/
├── src/                  # Исходный код
│   ├── bot.py            # Telegram-бот (основной)
│   ├── app.py            # Streamlit (демо)
│   ├── rag.py            # RAG-ядро
│   └── ingest.py         # Индексация
├── data/                 # Корпус статей (articles.jsonl)
├── chroma_db/            # Векторное хранилище (не в репо)
├── models/               # LLM (не в репо)
├── config/               # Конфигурация
├── scripts/              # Вспомогательные скрипты
├── tests/                # Тесты
├── .github/workflows/    # CI/CD
├── Dockerfile
├── Makefile
├── requirements.txt
├── .env.example
└── README.md

### Документация и материалы

- Отчёт о работе — ход разработки, метрики, выводы
- Презентация — финальная защита
- Репозиторий: https://github.com/bragind/article-agent

### Контакт
MLOps, Backend: Дмитрий Брагин
Email: dimanb1982@gmail.com
