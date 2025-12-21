import hashlib
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Работа с векторной базой данных ChromaDB с поддержкой фильтрации"""
    
    def __init__(self, path: str = "chroma_db", batch_size: int = 4000):
        """
        Инициализация векторного хранилища
        
        Args:
            path: Путь к директории с базой данных
            batch_size: Размер батча для добавления
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
        Гарантирует отсутствие дубликатов
        """
        if not chunks:
            logger.warning("Пустой список чанков для добавления")
            return
        
        total_chunks = len(chunks)
        logger.info(f"Начинаю добавление {total_chunks} чанков...")
        
        # Детерминированные, уникальные ID на основе контента
        ids = []
        cleaned_chunks = []
        
        for chunk in chunks:
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            
            cleaned_metadata = {}
            for key, value in metadata.items():
                if key == "tags":
                    # Убедимся, что теги - строка
                    if isinstance(value, list):
                        cleaned_metadata[key] = ", ".join([str(v).strip() for v in value if v])[:200]
                    elif value is None:
                        cleaned_metadata[key] = ""
                    else:
                        cleaned_metadata[key] = str(value)[:200]
                elif isinstance(value, (list, dict, set)):
                    cleaned_metadata[key] = str(value)[:500]
                elif value is None:
                    cleaned_metadata[key] = ""
                else:
                    cleaned_metadata[key] = value

            # Создаем уникальный ключ из текста и ключевых метаданных
            content_key = f"{text[:200]}_{metadata.get('article_id', '')}_{metadata.get('chunk_index', 0)}"
            content_hash = hashlib.md5(content_key.encode()).hexdigest()[:16]
            
            # Формируем ID: префикс + хеш
            source_type = metadata.get("source_type", "unknown")
            chunk_id = f"{source_type}_{content_hash}"
            ids.append(chunk_id)

            cleaned_chunks.append({
                "text": text,
                "metadata": cleaned_metadata
            })

        chunks = cleaned_chunks
        
        # Проверка существующих ID перед добавлением
        existing_ids = set()
        try:
            existing = self.collection.get()
            if existing and existing.get("ids"):
                existing_ids = set(existing["ids"])
                logger.info(f"В коллекции уже есть {len(existing_ids)} записей")
        except Exception as e:
            logger.warning(f"Не удалось получить существующие ID: {e}")
        
        # Фильтрация дубликатов
        new_chunks = []
        new_embeddings = []
        new_ids = []
        
        for i, (chunk_id, chunk, embedding) in enumerate(zip(ids, chunks, embeddings)):
            if chunk_id in existing_ids:
                # Пропускаем уже существующий чанк
                if i % 1000 == 0:
                    logger.debug(f"Пропущен дубликат {chunk_id[:20]}...")
                continue
            
            new_ids.append(chunk_id)
            new_chunks.append(chunk)
            new_embeddings.append(embedding)
        
        if not new_ids:
            logger.info("Все чанки уже существуют в базе, ничего не добавляю")
            return
        
        logger.info(f"Будет добавлено {len(new_ids)} новых чанков (дубликатов: {total_chunks - len(new_ids)})")
        
        # Подготавливаем данные
        documents = [chunk["text"] for chunk in new_chunks]
        metadatas = [chunk["metadata"] for chunk in new_chunks]
        
        # Безопасное добавление батчей с транзакциями
        added_count = 0
        for start_idx in range(0, len(new_ids), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(new_ids))
            
            batch_ids = new_ids[start_idx:end_idx]
            batch_documents = documents[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx]
            batch_embeddings = new_embeddings[start_idx:end_idx]
            
            # Преобразуем embeddings в список (если это numpy array)
            if hasattr(batch_embeddings[0], 'tolist'):
                batch_embeddings_list = [emb.tolist() for emb in batch_embeddings]
            else:
                batch_embeddings_list = batch_embeddings
            
            try:
                # Используем upsert
                self.collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings_list,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                added_count += len(batch_ids)
                logger.debug(f"Добавлен батч {start_idx//self.batch_size + 1}: "
                            f"{len(batch_ids)} чанков (всего: {added_count})")
                
            except Exception as e:
                logger.error(f"Ошибка добавления батча {start_idx}-{end_idx}: {e}")
                
                # Безопасное поштучное добавление при ошибке
                for i, chunk_id in enumerate(batch_ids):
                    try:
                        # Проверяем, не добавился ли уже чанк при частичном успехе
                        self.collection.upsert(
                            ids=[chunk_id],
                            embeddings=[batch_embeddings_list[i]],
                            documents=[batch_documents[i]],
                            metadatas=[batch_metadatas[i]]
                        )
                        added_count += 1
                    except Exception as e2:
                        # Если чанк уже существует, это нормально для upsert
                        if "already exists" in str(e2).lower():
                            logger.debug(f"Чанк {chunk_id[:20]}... уже существует, пропускаю")
                        else:
                            logger.error(f"Ошибка добавления чанк {chunk_id[:20]}...: {e2}")
        
        logger.info(f"Успешно добавлено {added_count} чанков")

    def search(self, 
           query_embedding: np.ndarray, 
           top_k: int = 5, 
           where: Optional[Dict] = None) -> Dict[str, List]:
        """
        Поиск похожих чанков с поддержкой фильтрации
        
        Args:
            query_embedding: Эмбеддинг запроса
            top_k: Количество возвращаемых результатов
            where: Условия фильтрации (например {"source_type": "habr"})
            
        Returns:
            Результаты поиска
        """
        # Всегда преобразуем в список списков
        query_embedding_list = query_embedding
        if hasattr(query_embedding, 'tolist'):
            query_embedding_list = query_embedding.tolist()
        
        try:
            if where is None:
                return self.collection.query(
                    query_embeddings=[query_embedding_list],
                    n_results=top_k
                )
            else:
                # Проверяем, что where не пустой
                if where == {}:
                    return self.collection.query(
                        query_embeddings=[query_embedding_list],
                        n_results=top_k
                    )
                return self.collection.query(
                    query_embeddings=[query_embedding_list],
                    n_results=top_k,
                    where=where
                )
        except Exception as e:
            logger.error(f"Ошибка при поиске с where={where}: {e}")
            # Пробуем самый простой запрос без where
            try:
                return self.collection.query(
                    query_embeddings=[query_embedding_list],
                    n_results=top_k
                )
            except Exception as e2:
                logger.error(f"Критическая ошибка при поиске: {e2}")
                return {"documents": [], "metadatas": [], "ids": []}
    
    def count(self) -> int:
        """Количество чанков в коллекции"""
        return self.collection.count()