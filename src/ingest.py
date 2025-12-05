# ingest.py: индексация статей в ChromaDB
import json
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "articles"

def load_articles(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    client = PersistentClient(path="../chroma_db")
    collection = client.get_or_create_collection(COLLECTION_NAME)

    model = SentenceTransformer(MODEL_NAME)
    articles = list(load_articles("../data/articles.jsonl"))

    ids = [a["id"] for a in articles]
    texts = [a["text"] for a in articles]
    metadatas = [
        {"title": a["title"], "url": a["url"], "date": a["date"], "author": a["author"], "source": a["source"]}
        for a in articles
    ]

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print("Adding to ChromaDB...")
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
    print(f"Indexed {len(ids)} articles.")

if __name__ == "__main__":
    main()
