FROM python:3.11-slim

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы Python
COPY app.py ./
COPY src/ ./src/
COPY data/ ./data/

# Создаем необходимые директории
RUN mkdir -p /app/data/habr_articles /app/data/user_uploads /app/data/vector_db /app/logs

# Порт для Streamlit
EXPOSE 8501

# Команда по умолчанию
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]