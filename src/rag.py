# rag.py: RAG-ядро
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "articles"

class RAGAgent:
    def __init__(self):
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.client = PersistentClient(path="../chroma_db")
        self.collection = self.client.get_collection(COLLECTION_NAME)

    def retrieve(self, query: str, k: int = 3):
        emb = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=emb, n_results=k)
        return results

    def generate_answer(self, query: str, k: int = 3):
        results = self.retrieve(query, k)
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        # В реальном проекте — подключить LLM
        answer = f"Пример ответа по запросу: {query}"
        sources = [{"title": m["title"], "url": m["url"]} for m in metas]
        questions = ["Пример вопроса 1?", "Пример вопроса 2?"]

        return {
            "answer": answer,
            "sources": sources,
            "questions": questions
        }
