# AI-агент поиска статей 

MVP на основе RAG для интеллектуального поиска и аннотирования статей.

## Установка
\`\`\`bash
make install
\`\`\`

## Индексация
\`\`\`bash
make ingest
\`\`\`

## Запуск Streamlit
\`\`\`bash
make streamlit
\`\`\`

## Запуск Telegram-бота
Создайте \`.env\` с \`TELEGRAM_BOT_TOKEN\` и выполните:
\`\`\`bash
make bot
\`\`\`

## Docker
\`\`\`bash
make docker-build
make docker-run
\`\`\`
