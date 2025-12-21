.PHONY: install test ingest bot streamlit docker-build docker-run clean lint

install:
	pip install -r requirements.txt

parse:
	python parser_articles/main.py

ingest:
	python scripts/build_index.py

bot:
	python src/bot.py

auth-init:
	python scripts/init_auth.py

auth-reset:
	rm -f config/auth_config.yaml data/users.json

test-users:
	python scripts/create_test_users.py

reset-auth:
	rm -f data/users.json
	rm -f config/auth_config.yaml
	@echo "Система аутентификации сброшена"

streamlit:
	streamlit run src/streamlit_app.py

docker-build:
	docker build -t article-agent .

docker-run:
	docker run -p 8501:8501 article-agent

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

lint:
	flake8 src/ --max-line-length=88
	black src/ --check
	mypy src/

format:
	black src/
	isort src/