markdown
123456789101112131415161718192021222324252627282930313233343536373839
# AI-агент для интеллектуального поиска статей

> **MVP на основе RAG** для поиска, анализа и аннотирования IT-статей из открытых источников (Habr, CNews, Rusbase и др.).

Основной интерфейс — **Telegram-бот**.  
Дополнительно: веб-демо через Streamlit.

---

## 🎯 Возможности

2. Настройте окружение
bash
1234
python -m venv venv
source venv/bin/activate      # Linux/macOS
# или
venv\Scripts\activate        # Windows
3. Установите зависимости
bash
1
4. Подготовьте данные и индекс
bash
12
5. Настройте Telegram-бота
Создайте бота в @BotFather
Получите токен
Создайте файл .env:
env
1
6. Запустите
bash
12345
# Основной режим — Telegram-бот
python src/bot.py

# Или демо — Streamlit
streamlit run src/app.py
🐳 Docker
bash
12345678
# Сборка
docker build -t article-agent .

# Запуск бота
docker run --env-file .env article-agent

# Запуск Streamlit
docker run -p 8501:8501 article-agent streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0
🛠️ MLOps и CI/CD
Версионирование кода (Git)
Тестирование (pytest)
Автоматическая сборка и публикация Docker-образа (GitHub Actions)
Логирование (logs/bot.log)
Конфигурация через config/config.yaml
📂 Структура проекта
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

📄 Документация и материалы
Отчёт о работе — ход разработки, метрики, выводы
Презентация — финальная защита
Репозиторий: https://github.com/bragind/article-agent
📬 Контакты
MLOps, Backend: Дмитрий Брагин
Email: dimanb1982@gmail.com
