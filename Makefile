.PHONY: install test ingest streamlit bot docker-build docker-run

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

ingest:
	python scripts/build_index.py

streamlit:
	streamlit run src/app.py

bot:
	python src/bot.py

docker-build:
	docker build -t article-agent .

docker-run:
	docker-compose up
