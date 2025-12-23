"""
Менеджер для работы с пользовательскими документами
"""

import json
import hashlib
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os

logger = logging.getLogger(__name__)


class DocumentManager:
    """Менеджер для работы с документами пользователей"""
    
    def __init__(self, data_dir: Path, chunker, embedder, vector_store, llm_client):
        """
        Args:
            data_dir: Директория с данными
            chunker: Разбиватель текста на чанки
            embedder: Генератор эмбеддингов
            vector_store: Векторное хранилище
            llm_client: Клиент для LLM (для генерации тегов)
        """
        self.data_dir = data_dir
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_client = llm_client
        
        # Создаем необходимые директории
        self.user_uploads_dir = data_dir / "user_uploads"
        self.user_uploads_dir.mkdir(exist_ok=True)
    
    def add_user_document(self, 
                         document_data: Dict, 
                         user_id: str, 
                         generate_tags: bool = True) -> Tuple[bool, str, Optional[str]]:
        """
        Добавление пользовательского документа
        
        Args:
            document_data: Данные документа
            user_id: ID пользователя
            generate_tags: Генерировать теги автоматически
            
        Returns:
            Tuple[успех, сообщение, ID документа]
        """
        try:
            # Проверка обязательных полей
            if not document_data.get("text") or len(document_data["text"]) < 50:
                return False, "Документ слишком короткий", None
            
            # Генерация ID документа
            document_id = self._generate_document_id(user_id, document_data)
            
            # Обогащение метаданных
            enriched_data = self._enrich_document_data(document_data, document_id, user_id)
            
            # Генерация тегов если требуется
            if generate_tags and self.llm_client:
                tags = self._generate_document_tags(enriched_data)
                enriched_data["tags"] = tags
            
            # Сохранение документа
            save_path = self._save_document_to_disk(enriched_data, user_id)
            if not save_path:
                return False, "Ошибка сохранения документа", None
            
            # Добавление в векторный индекс если доступно
            index_success = False
            if self.vector_store and self.embedder and self.chunker:
                index_success = self._add_to_vector_index(enriched_data, user_id)
            
            return True, f"Документ добавлен{' и проиндексирован' if index_success else ''}", document_id
            
        except Exception as e:
            logger.error(f"Ошибка добавления документа: {e}")
            return False, f"Ошибка: {str(e)}", None
    
    def get_user_documents(self, user_id: str) -> List[Dict]:
        """Получение всех документов пользователя"""
        user_dir = self.user_uploads_dir / f"user_{user_id}"
        
        if not user_dir.exists():
            return []
        
        documents = []
        for filepath in user_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    
                    # Гарантируем обязательные поля
                    doc["uploaded_by"] = user_id
                    doc["is_user_document"] = True
                    doc["source_type"] = "user"
                    
                    documents.append(doc)
                    
            except Exception as e:
                logger.error(f"Ошибка загрузки документа {filepath}: {e}")
        
        # Сортировка по дате загрузки
        documents.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
        return documents
    
    def delete_user_document(self, document_id: str, user_id: str) -> Tuple[bool, str]:
        """Удаление документа пользователя"""
        try:
            user_dir = self.user_uploads_dir / f"user_{user_id}"
            
            if not user_dir.exists():
                return False, "Директория пользователя не найдена"
            
            # Ищем файл документа
            for filepath in user_dir.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                        if doc.get("id") == document_id:
                            # Удаляем файл
                            os.remove(filepath)
                            
                            # Пытаемся удалить из векторного индекса
                            self._remove_from_vector_index(document_id)
                            
                            return True, "Документ удален"
                except:
                    continue
            
            return False, "Документ не найден"
            
        except Exception as e:
            logger.error(f"Ошибка удаления документа: {e}")
            return False, f"Ошибка: {str(e)}"
    
    def get_document_stats(self, user_id: str) -> Dict:
        """Статистика документов пользователя"""
        documents = self.get_user_documents(user_id)
        
        if not documents:
            return {
                "count": 0,
                "total_size": 0,
                "total_pages": 0,
                "avg_length": 0,
                "recent_uploads": []
            }
        
        total_chars = sum(len(d.get("text", "")) for d in documents)
        total_pages = sum(len(d.get("pages", [])) for d in documents)
        
        # Последние загрузки
        recent = []
        for doc in documents[:5]:
            recent.append({
                "title": doc.get("title", "Без названия"),
                "date": doc.get("uploaded_at", "")[:10],
                "length": len(doc.get("text", "")),
                "pages": len(doc.get("pages", []))
            })
        
        return {
            "count": len(documents),
            "total_size": total_chars,
            "total_pages": total_pages,
            "avg_length": total_chars // len(documents) if documents else 0,
            "recent_uploads": recent
        }
    
    def _generate_document_id(self, user_id: str, document_data: Dict) -> str:
        """Генерация уникального ID документа"""
        timestamp = datetime.now().isoformat()
        content_hash = hashlib.md5(
            f"{user_id}_{timestamp}_{document_data.get('title', '')[:50]}".encode()
        ).hexdigest()[:12]
        
        return f"user_{content_hash}"
    
    def _enrich_document_data(self, 
                             document_data: Dict, 
                             document_id: str, 
                             user_id: str) -> Dict:
        """Обогащение документа метаданными"""
        enriched = document_data.copy()
        
        # Обязательные поля
        enriched["id"] = document_id
        enriched["uploaded_at"] = datetime.now().isoformat()
        enriched["uploaded_by"] = user_id
        enriched["is_user_document"] = True
        enriched["source"] = "User Upload"
        enriched["source_type"] = "user"
        
        # Дополнительные поля
        if "text_length" not in enriched:
            enriched["text_length"] = len(document_data.get("text", ""))
        
        if "has_content" not in enriched:
            enriched["has_content"] = enriched["text_length"] > 100
        
        # Нормализация тегов
        if "tags" in enriched:
            enriched["tags"] = self._normalize_tags(enriched["tags"])
        
        return enriched
    
    def _generate_document_tags(self, document_data: Dict) -> List[str]:
        """Генерация тегов для документа"""
        try:
            text = document_data.get("text", "")[:2000]
            title = document_data.get("title", "")
            
            if self.llm_client:
                tags = self.llm_client.generate_tags_for_document(text, title, max_tags=5)
                return self._normalize_tags(tags)
        except Exception as e:
            logger.warning(f"Ошибка генерации тегов: {e}")
        
        # Fallback теги
        return ["документ", "загружено", "пользовательский"]
    
    def _save_document_to_disk(self, document_data: Dict, user_id: str) -> Optional[Path]:
        """Сохранение документа на диск"""
        try:
            user_dir = self.user_uploads_dir / f"user_{user_id}"
            user_dir.mkdir(exist_ok=True)
            
            filename = f"{document_data['id']}.json"
            filepath = user_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Документ сохранен: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Ошибка сохранения документа: {e}")
            return None
    
    def _add_to_vector_index(self, document_data: Dict, user_id: str) -> bool:
        """Добавление документа в векторный индекс"""
        if not all([self.chunker, self.embedder, self.vector_store]):
            return False
        
        try:
            text = document_data.get("text", "")
            if not text or len(text) < 50:
                return False
            
            # Разбиваем на чанки
            chunks = self.chunker.split_text(text)
            if not chunks:
                return False
            
            # Подготавливаем чанки
            chunk_objects = []
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"chunk_{uuid.uuid4().hex[:12]}_{document_data['id'][:8]}"
                
                metadata = {
                    "article_id": document_data["id"],
                    "title": document_data.get("title", "Без названия")[:200],
                    "author": document_data.get("author", "Неизвестен")[:100],
                    "url": document_data.get("url", "")[:200],
                    "source": document_data.get("source", "User Upload")[:100],
                    "date": str(document_data.get("date", ""))[:50],
                    "tags": ", ".join(document_data.get("tags", []))[:200],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "is_user_document": True,
                    "source_type": "user",
                    "uploaded_by": user_id[:50]
                }
                
                chunk_objects.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": metadata
                })
            
            # Генерируем эмбеддинги
            texts = [chunk["text"] for chunk in chunk_objects]
            embeddings = self.embedder.embed(texts)
            
            if len(embeddings) != len(chunk_objects):
                logger.error("Количество эмбеддингов не совпадает с количеством чанков")
                return False
            
            # Добавляем в векторную БД
            self.vector_store.add_chunks(chunk_objects, embeddings)
            
            logger.info(f"Документ добавлен в векторный индекс: {len(chunks)} чанков")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления в векторный индекс: {e}")
            return False
    
    def _remove_from_vector_index(self, document_id: str) -> bool:
        """Удаление документа из векторного индекса"""
        try:
            if not self.vector_store:
                return False
            
            # Находим и удаляем все чанки документа
            results = self.vector_store.collection.get(
                where={"article_id": {"$eq": document_id}}
            )
            
            if results and results.get("ids"):
                self.vector_store.collection.delete(ids=results["ids"])
                logger.info(f"Удалено {len(results['ids'])} чанков документа {document_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка удаления из векторного индекса: {e}")
            return False
    
    def _normalize_tags(self, tags) -> List[str]:
        """Нормализация тегов"""
        if not tags:
            return []
        
        if isinstance(tags, str):
            import re
            tags = [tag.strip() for tag in re.split(r'[,;]', tags) if tag.strip()]
        elif isinstance(tags, list):
            tags = [str(tag).strip() for tag in tags if tag]
        
        # Убираем дубликаты
        unique_tags = []
        seen = set()
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower not in seen:
                seen.add(tag_lower)
                unique_tags.append(tag)
        
        return unique_tags[:10]