"""
Векторное хранилище с поддержкой батчинга
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import numpy as np


class VectorStore:
    """Работа с векторной базой данных ChromaDB с поддержкой батчинга"""
    
    def __init__(self, path: str = "chroma_db", batch_size: int = 4000):
        """
        Инициализация векторного хранилища
        
        Args:
            path: Путь к директории с базой данных
            batch_size: Размер батча для добавления (ограничение ChromaDB: 5461)
        """
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Создаем или получаем коллекцию
        self.collection = self.client.get_or_create_collection(
            name="articles",
            metadata={"description": "Чанки статей для RAG поиска"}
        )
        
        self.batch_size = min(batch_size, 5461)  # Ограничение ChromaDB
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """
        Добавление чанков в векторную базу данных с разбиением на батчи
        
        Args:
            chunks: Список чанков с текстом и метаданными
            embeddings: Матрица эмбеддингов чанков
        """
        total_chunks = len(chunks)
        print(f"Добавление {total_chunks} чанков батчами по {self.batch_size}")
        
        # Подготавливаем данные для батчей
        ids = [f"chunk_{i}" for i in range(total_chunks)]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Разбиваем на батчи
        for start_idx in range(0, total_chunks, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_chunks)
            
            batch_ids = ids[start_idx:end_idx]
            batch_documents = documents[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx]
            batch_embeddings = embeddings[start_idx:end_idx].tolist()
            
            # Добавляем батч
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            
            batch_num = (start_idx // self.batch_size) + 1
            total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
            print(f"Батч {batch_num}/{total_batches}: чанки {start_idx}-{end_idx-1}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Dict[str, List]:
        """
        Поиск похожих чанков по эмбеддингу запроса
        
        Args:
            query_embedding: Эмбеддинг запроса
            top_k: Количество возвращаемых результатов
            
        Returns:
            Результаты поиска
        """
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
    
    def count(self) -> int:
        """
        Возвращает количество чанков в коллекции
        """
        return self.collection.count()