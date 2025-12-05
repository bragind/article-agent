.PHONY: install test ingest bot streamlit docker-build

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

ingest:
	python scripts/build_index.py

bot:
	python src/bot.py

streamlit:
	streamlit run src/app.py

docker-build:
	docker build -t article-agent .