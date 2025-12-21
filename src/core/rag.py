"""
src/core/rag.py - Основной RAG агент с поддержкой пользовательских документов и областей поиска
"""

import uuid
import logging
import json
import os
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
import sys
from pathlib import Path

# Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.ingest.embedder import Embedder
    from src.ingest.vector_store import VectorStore
    from src.ingest.chunker import TextChunker
    HAS_INGEST = True
except ImportError as e:
    HAS_INGEST = False
    logging.warning(f"Модули ingest не найдены: {e}")

try:
    from src.generation.llm_client import LLMClient
    HAS_LLM = True
except ImportError as e:
    HAS_LLM = False
    logging.warning(f"LLMClient не найден: {e}")

logger = logging.getLogger(__name__)


class SearchScope(Enum):
    """Области поиска"""
    ALL = "all"
    USER_ONLY = "user"
    HABR_ONLY = "habr"


class RAGAgent:
    """Основной RAG агент с поддержкой пользовательских документов и векторного поиска"""
    
    def __init__(self, 
                 data_dir: str = "data",
                 vector_db_path: str = "chroma_db",
                 use_vector_search: bool = True,
                 llm_enabled: bool = True):
        """
        Инициализация RAG агента
        
        Args:
            data_dir: Директория с данными
            vector_db_path: Путь к векторной базе данных
            use_vector_search: Использовать векторный поиск (иначе текстовый)
            llm_enabled: Использовать LLM для генерации ответов
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.vector_db_path = vector_db_path
        self.use_vector_search = use_vector_search
        self.llm_enabled = llm_enabled
        
        # Проверка доступности компонентов
        if self.use_vector_search and not HAS_INGEST:
            logger.error("Векторный поиск отключен: модули ingest не найдены")
            self.use_vector_search = False
        
        if self.llm_enabled and not HAS_LLM:
            logger.error("LLM отключен: модуль не найден")
            self.llm_enabled = False
        
        # Создаем необходимые директории
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "user_uploads").mkdir(exist_ok=True)
        (self.data_dir / "vector_db").mkdir(exist_ok=True)
        
        # Инициализация компонентов
        self.embedder = None
        self.vector_store = None
        self.chunker = None
        self.llm_client = None
        self._init_components()
        
        # Загрузка статей
        self.habr_articles = []
        self.user_articles = []
        self.all_articles = []
        self._load_articles()
        
        # Статистика
        self.total_chunks = 0
        self.initialization_errors = []
        
        # Проверяем готовность
        self._check_readiness()
        
        logger.info(f"RAGAgent инициализирован: {len(self.all_articles)} статей, "
                   f"векторный поиск: {self.use_vector_search}, LLM: {self.llm_enabled}")
    
    def _init_components(self):
        """Инициализация компонентов системы"""
        # Инициализация векторного поиска
        if self.use_vector_search and HAS_INGEST:
            try:
                self.embedder = Embedder()
                self.vector_store = VectorStore(path=self.vector_db_path)
                self.chunker = TextChunker(chunk_size=800, chunk_overlap=150)
                logger.info("Компоненты векторного поиска инициализированы")
            except Exception as e:
                logger.error(f"Ошибка инициализации векторного поиска: {e}")
                self.initialization_errors.append(f"Vector search init: {str(e)}")
                self.use_vector_search = False
        
        # Инициализация LLM
        if self.llm_enabled and HAS_LLM:
            try:
                self.llm_client = LLMClient()
                if hasattr(self.llm_client, 'test_connection'):
                    if not self.llm_client.test_connection():
                        logger.warning("LLM недоступен, отключаем генерацию")
                        self.llm_enabled = False
                    else:
                        logger.info("LLM клиент инициализирован")
                else:
                    logger.info("LLM клиент инициализирован (без проверки соединения)")
            except Exception as e:
                logger.error(f"Ошибка инициализации LLM: {e}")
                self.initialization_errors.append(f"LLM init: {str(e)}")
                self.llm_enabled = False
    
    def _check_readiness(self):
        """Проверка готовности системы"""
        if not self.all_articles:
            self.initialization_errors.append("Нет загруженных статей")
        
        if self.use_vector_search and not self.vector_store:
            self.initialization_errors.append("Векторное хранилище не инициализировано")
        
        if self.llm_enabled and not self.llm_client:
            self.initialization_errors.append("LLM клиент не инициализирован")
    
    def _load_articles(self):
        """Загрузка статей из всех источников"""
        try:
            # Загрузка статей Habr
            habr_path = self.data_dir / "articles_batch.jsonl"
            if habr_path.exists():
                count = 0
                with open(habr_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                article = json.loads(line)
                                article["is_user_document"] = False
                                article["source_type"] = "habr"
                                article["uploaded_by"] = "system"
                                self.habr_articles.append(article)
                                self.all_articles.append(article)
                                count += 1
                            except json.JSONDecodeError as e:
                                logger.warning(f"Ошибка парсинга JSON в строке: {e}")
                logger.info(f"Загружено {count} статей Habr")
            else:
                logger.warning(f"Файл {habr_path} не найден")
            
            # Загрузка пользовательских статей
            user_dir = self.data_dir / "user_uploads"
            if user_dir.exists():
                user_count = 0
                for filepath in user_dir.glob("*.json"):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            article = json.load(f)
                            article["is_user_document"] = True
                            article["source_type"] = "user"
                            # Гарантируем наличие uploaded_by
                            if "uploaded_by" not in article:
                                article["uploaded_by"] = "unknown"
                            self.user_articles.append(article)
                            self.all_articles.append(article)
                            user_count += 1
                    except Exception as e:
                        logger.error(f"Ошибка загрузки файла {filepath.name}: {e}")
                logger.info(f"Загружено {user_count} пользовательских статей")
            else:
                logger.info(f"Директория {user_dir} не существует, пользовательских статей нет")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки статей: {e}")
            self.initialization_errors.append(f"Load articles: {str(e)}")
    
    def _add_to_vector_index(self, article: Dict, user_id: str = None) -> bool:
        """Добавление статьи в векторный индекс"""
        # Проверяем доступность компонентов
        if not self.use_vector_search:
            logger.warning("Векторный поиск отключен")
            return False
        
        if not self.vector_store:
            logger.error("Векторное хранилище не инициализировано")
            return False
        
        if not self.embedder:
            logger.error("Embedder не инициализирован")
            return False
        
        if not self.chunker:
            logger.error("Chunker не инициализирован")
            return False
        
        try:
            text = article.get("text", "")
            if not text or len(text) < 50:
                logger.warning(f"Статья {article.get('id')} слишком короткая для индексации ({len(text)} символов)")
                return False
            
            # Разбиваем на чанки
            chunks = self.chunker.split_text(text)
            if not chunks:
                logger.warning(f"Не удалось разбить статью {article.get('id')} на чанки")
                return False
            
            # Подготавливаем чанки с метаданными
            chunk_objects = []
            for i, chunk_text in enumerate(chunks):
                # Используем UUID для уникальных ID
                chunk_id = f"chunk_{uuid.uuid4().hex[:12]}_{article.get('id', '')[:8]}"
                
                # КОНСИСТЕНТНОСТЬ: Всегда сохраняем теги как строку
                tags = article.get("tags", [])
                tags_str = ""
                
                if isinstance(tags, list):
                    # Фильтруем пустые теги, убираем дубликаты
                    unique_tags = []
                    seen = set()
                    for tag in tags:
                        if tag and str(tag).strip():
                            tag_str = str(tag).strip()
                            if tag_str not in seen:
                                seen.add(tag_str)
                                unique_tags.append(tag_str)
                    tags_str = ", ".join(unique_tags[:10])  # Ограничиваем 10 тегами
                elif isinstance(tags, str):
                    # Очищаем строку: удаляем лишние разделители
                    tags = tags.replace(";", ",").replace("|", ",")
                    tags = re.sub(r',\s*,', ',', tags)  # Убираем двойные запятые
                    tags = re.sub(r'\s+', ' ', tags)    # Убираем лишние пробелы
                    tags_str = tags.strip(" ,")[:200]   # Ограничиваем длину
                elif tags:  # Любой другой тип
                    tags_str = str(tags)[:200]
                
                # Определяем uploaded_by
                uploaded_by = user_id if article.get("is_user_document") else "system"
                if not uploaded_by and article.get("is_user_document"):
                    uploaded_by = article.get("uploaded_by", "unknown")
                
                metadata = {
                    "article_id": article.get("id", ""),
                    "title": article.get("title", "Без названия")[:200],
                    "author": article.get("author", "Неизвестен")[:100],
                    "url": article.get("url", "")[:200],
                    "source": article.get("source", "Unknown")[:100],
                    "date": str(article.get("date", ""))[:50],
                    "tags": tags_str,  # Всегда строка!
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "is_user_document": article.get("is_user_document", False),
                    "source_type": article.get("source_type", "unknown"),
                    "uploaded_by": uploaded_by[:50]
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
                logger.error(f"Количество эмбеддингов ({len(embeddings)}) не совпадает с количеством чанков ({len(chunk_objects)})")
                return False
            
            # Добавляем в векторную БД
            self.vector_store.add_chunks(chunk_objects, embeddings)
            self.total_chunks += len(chunks)
            
            logger.info(f"Статья добавлена в векторный индекс: {len(chunks)} чанков")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления в векторный индекс: {e}")
            self.initialization_errors.append(f"Vector index add: {str(e)}")
            return False
    
    def search(self, 
               query: str, 
               user_id: str, 
               scope: SearchScope = SearchScope.ALL, 
               limit: int = 10) -> List[Dict]:
        """
        Поиск статей по запросу с учетом области поиска
        
        Args:
            query: Текст запроса
            user_id: ID пользователя
            scope: Область поиска
            limit: Максимальное количество результатов
            
        Returns:
            Список результатов поиска
        """
        # Векторный поиск
        if self.use_vector_search and self.vector_store:
            try:
                return self._vector_search(query, user_id, scope, limit)
            except Exception as e:
                logger.error(f"Ошибка векторного поиска: {e}")
                # Fallback на текстовый поиск
                return self._text_search(query, user_id, scope, limit)
        # Текстовый поиск (fallback)
        else:
            logger.info("Использую текстовый поиск (векторный недоступен)")
            return self._text_search(query, user_id, scope, limit)
    
    def _vector_search(self, query: str, user_id: str, scope: SearchScope, limit: int) -> List[Dict]:
        """Векторный поиск с фильтрацией по области"""
        # Проверяем, что векторный поиск доступен
        if not self.use_vector_search or not self.vector_store or not self.embedder:
            logger.warning("Векторный поиск недоступен, возвращаю пустой список")
            return []
        
        try:
            # embed() возвращает numpy array формы (1, embedding_dim) даже для одного текста
            query_embedding_result = self.embedder.embed([query])
            
            if query_embedding_result is None or len(query_embedding_result) == 0:
                logger.error("Не удалось создать эмбеддинг для запроса")
                return []
            
            # Берём первый (и единственный) эмбеддинг
            #query_embedding = query_embedding_result[0]  # Одномерный numpy array
            
            # Фильтрация по области поиска
            where_filter = None
            if scope == SearchScope.HABR_ONLY:
                where_filter = {"source_type": {"$eq": "habr"}}
                logger.debug(f"Поиск только в Habr, фильтр: {where_filter}")
            elif scope == SearchScope.USER_ONLY:
                where_filter = {
                    "$and": [
                        {"source_type": {"$eq": "user"}},
                        {"uploaded_by": {"$eq": user_id}}
                    ]
                }
                logger.debug(f"Поиск только в пользовательских документах, фильтр: {where_filter}")
            elif scope == SearchScope.ALL:
                # ВАЖНО: Используем $or для поиска в обоих источниках
                where_filter = {
                    "$or": [
                        {"source_type": {"$eq": "habr"}},
                        {
                            "$and": [
                                {"source_type": {"$eq": "user"}},
                                {"uploaded_by": {"$eq": user_id}}
                            ]
                        }
                    ]
                }
                logger.debug(f"Поиск везде, фильтр: {where_filter}")
            
            # Выполняем поиск с увеличенным limit для лучшей фильтрации
            search_limit = limit * 2
            results = self.vector_store.search(
                query_embedding_result, 
                top_k=search_limit,
                where=where_filter
            )
            
            if not results or not results.get("documents"):
                logger.info("Векторный поиск не дал результатов")
                return []
            
            # ChromaDB возвращает список списков: documents[0] - результаты для первого запроса
            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results["metadatas"] else []
            
            if not docs or not metas:
                return []
            
            # Форматируем результаты
            formatted_results = []
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                # Дополнительная проверка для пользовательских документов
                if scope == SearchScope.USER_ONLY:
                    if meta.get("uploaded_by") != user_id:
                        continue
                
                # Обрабатываем теги из строки обратно в список
                tags_data = meta.get("tags", "")
                if isinstance(tags_data, str):
                    # Разделяем строку тегов по запятым, очищаем
                    tags_list = [tag.strip() for tag in tags_data.split(',') if tag.strip()]
                elif isinstance(tags_data, list):
                    tags_list = tags_data
                else:
                    tags_list = []
                
                formatted_results.append({
                    "article": {
                        "id": meta.get("article_id", f"unknown_{i}"),
                        "title": meta.get("title", "Без названия"),
                        "text": doc,
                        "author": meta.get("author", "Неизвестен"),
                        "url": meta.get("url", ""),
                        "source": meta.get("source", "Unknown"),
                        "date": meta.get("date", ""),
                        "tags": tags_list,  # Возвращаем как список
                        "text_length": len(doc),
                        "is_user_document": meta.get("is_user_document", False),
                        "uploaded_by": meta.get("uploaded_by", "")
                    },
                    "score": 1.0 - (i * 0.05),  # Имитация релевантности (первые результаты лучше)
                    "is_user_document": meta.get("is_user_document", False)
                })
            
            # Ограничиваем количество результатов
            return formatted_results[:limit]
            
        except Exception as e:
            logger.error(f"Ошибка векторного поиска: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []  # Возвращаем пустой список вместо выброса исключения
        
    def _text_search(self, query: str, user_id: str, scope: SearchScope, limit: int) -> List[Dict]:
        """Текстовый поиск (fallback)"""
        try:
            results = []
            query_lower = query.lower()
            
            # Определяем какие статьи искать
            articles_to_search = []
            if scope == SearchScope.HABR_ONLY:
                articles_to_search = self.habr_articles
                logger.debug(f"Текстовый поиск в Habr: {len(articles_to_search)} статей")
            elif scope == SearchScope.USER_ONLY:
                articles_to_search = [a for a in self.user_articles if a.get("uploaded_by") == user_id]
                logger.debug(f"Текстовый поиск в пользовательских документах: {len(articles_to_search)} статей")
            else:  # ALL
                articles_to_search = self.habr_articles + \
                                   [a for a in self.user_articles if a.get("uploaded_by") == user_id]
                logger.debug(f"Текстовый поиск везде: {len(articles_to_search)} статей")
            
            # Простой поиск по тексту
            for article in articles_to_search:
                score = 0
                
                # Поиск в заголовке
                title = article.get("title", "").lower()
                if query_lower in title:
                    score += 10
                
                # Поиск в тексте
                text = article.get("text", "").lower()
                if query_lower in text:
                    occurrences = text.count(query_lower)
                    score += occurrences * 3
                
                # Поиск в тегах
                tags = article.get("tags", "")
                if tags and isinstance(tags, str):
                    # Ищем вхождение запроса в строке тегов
                    if query_lower in tags.lower():
                        score += 5
                    # Также проверяем отдельные теги (разделенные запятыми)
                    tag_list = [t.strip().lower() for t in tags.split(',') if t.strip()]
                    for tag in tag_list:
                        if query_lower in tag:
                            score += 3  # Меньший вес для поиска в отдельных тегах
                
                if score > 0:
                    results.append({
                        "article": article,
                        "score": score,
                        "is_user_document": article.get("is_user_document", False)
                    })
            
            # Сортировка по релевантности
            results.sort(key=lambda x: x["score"], reverse=True)
            logger.debug(f"Текстовый поиск нашел {len(results)} результатов")
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Ошибка текстового поиска: {e}")
            return []
    
    def generate_answer(self, 
                        query: str, 
                        user_id: str, 
                        scope: SearchScope = SearchScope.ALL) -> Dict:
        """
        Генерация ответа на основе найденных статей
        
        Args:
            query: Вопрос пользователя
            user_id: ID пользователя
            scope: Область поиска
            
        Returns:
            Словарь с ответом, источниками и вопросами
        """
        # Проверяем, есть ли статьи
        if not self.all_articles:
            return {
                "answer": "В системе пока нет статей для поиска.\n\n"
                          "Загрузите статьи или проверьте настройки данных.",
                "sources": [],
                "questions": []
            }
        
        # Поиск релевантных статей
        search_results = self.search(query, user_id, scope, limit=5)
        
        if not search_results:
            return {
                "answer": f"По вашему запросу '{query}' ничего не найдено.\n\n"
                         f"Попробуйте:\n"
                         f"- Изменить формулировку запроса\n"
                         f"- Использовать ключевые слова\n"
                         f"- Изменить область поиска",
                "sources": [],
                "questions": []
            }
        
        # Используем LLM для генерации ответа
        if self.llm_enabled and self.llm_client:
            return self._generate_with_llm(query, search_results)
        # Простой ответ без LLM
        else:
            return self._generate_simple_answer(query, search_results)
    
    def _generate_with_llm(self, query: str, search_results: List[Dict]) -> Dict:
        """Генерация ответа с использованием LLM"""
        try:
            # Подготавливаем контекст для LLM
            context_chunks = []
            for result in search_results[:3]:  # Используем топ-3 результата
                article = result["article"]
                source_type = "ваш документ" if result["is_user_document"] else "статья Habr"
                context_chunks.append({
                    "text": article.get("text", "")[:1000],  # Ограничиваем длину
                    "metadata": {
                        "title": article.get("title", ""),
                        "author": article.get("author", ""),
                        "source": source_type,
                        "is_user_document": result["is_user_document"]
                    }
                })
            
            # Генерируем ответ через LLM
            llm_result = self.llm_client.generate_answer_with_context(
                query=query,
                context=context_chunks,
                top_k=len(context_chunks)
            )
            
            # Формируем источники
            sources = []
            for result in search_results[:3]:
                article = result["article"]
                sources.append({
                    "title": article.get("title", "Без названия"),
                    "url": article.get("url", "#"),
                    "author": article.get("author", "Неизвестен"),
                    "is_user_document": result["is_user_document"]
                })
            
            # Генерация вопросов для самопроверки
            questions = []
            if search_results:
                top_article_text = search_results[0]["article"].get("text", "")[:500]
                questions = self.llm_client.generate_questions(
                    article_text=top_article_text,
                    num_questions=3
                )
            
            return {
                "answer": llm_result["answer"],
                "sources": sources,
                "questions": questions
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации через LLM: {e}")
            return self._generate_simple_answer(query, search_results)
    
    def _generate_simple_answer(self, query: str, search_results: List[Dict]) -> Dict:
        """Простая генерация ответа без LLM"""
        try:
            answer_lines = [
                f"🔍 **По вашему запросу '{query}' найдено {len(search_results)} материалов:**\n"
            ]
            
            for i, result in enumerate(search_results[:3], 1):
                article = result["article"]
                source_type = "📁 Ваш документ" if result["is_user_document"] else "🌐 Habr"
                
                # Краткое содержание (первые 150 символов)
                preview = article.get("text", "")
                if len(preview) > 150:
                    preview = preview[:150] + "..."
                
                answer_lines.append(f"\n**{i}. {article['title']}**")
                answer_lines.append(f"   {source_type} • {article.get('author', 'Не указан')}")
                answer_lines.append(f"   {preview}")
            
            answer_lines.append("\n💡 *Для более подробной информации перейдите по ссылкам в источниках.*")
            
            # Формируем источники
            sources = []
            for result in search_results[:3]:
                article = result["article"]
                sources.append({
                    "title": article.get("title", "Без названия"),
                    "url": article.get("url", "#"),
                    "author": article.get("author", "Неизвестен"),
                    "is_user_document": result["is_user_document"]
                })
            
            # Вопросы для самопроверки
            questions = [
                f"Какие ключевые идеи по теме '{query}' вы нашли?",
                f"Какую информацию из найденных материалов вы считаете наиболее полезной?",
                f"Что нового вы узнали о теме '{query}'?"
            ]
            
            return {
                "answer": "\n".join(answer_lines),
                "sources": sources,
                "questions": questions
            }
        except Exception as e:
            logger.error(f"Ошибка в simple answer: {e}")
            return {
                "answer": f"🔍 Найдено {len(search_results)} материалов по вашему запросу.\n\n"
                         f"Произошла ошибка при формировании ответа. Попробуйте еще раз.",
                "sources": [],
                "questions": []
            }
    
    def add_user_article(self, article_data: Dict, user_id: str) -> str:
        """
        Добавление пользовательской статьи
        
        Args:
            article_data: Данные статьи
            user_id: ID пользователя
            
        Returns:
            ID добавленной статьи или None при ошибке
        """
        try:
            # Генерируем уникальный ID
            timestamp = datetime.now().isoformat()
            article_id = f"user_{hashlib.md5(f'{user_id}_{timestamp}'.encode()).hexdigest()[:12]}"
            
            # Обогащаем данные статьи
            article_data["id"] = article_id
            article_data["uploaded_at"] = timestamp
            article_data["uploaded_by"] = user_id
            article_data["is_user_document"] = True
            article_data["source"] = "User Upload"
            article_data["source_type"] = "user"

            # Генерируем теги автоматически
            text = article_data.get("text", "")
            title = article_data.get("title", "")

            # Используем LLM для генерации тегов (можно вынести в конфиг)
            use_llm_for_tags = True  # Можно сделать настройкой
            tags = self.generate_document_tags(text, title, use_llm=use_llm_for_tags)
            
            # Убедимся что теги - список
            if isinstance(tags, str):
                # Если это строка, разделяем по запятым
                tags = [t.strip() for t in tags.split(',') if t.strip()]
            elif not isinstance(tags, list):
                tags = []
            
            article_data["tags"] = tags
            
            # Обрабатываем теги
            tags = article_data.get("tags", [])
            if isinstance(tags, str):
                article_data["tags"] = [t.strip() for t in tags.split(',') if t.strip()]
            elif not isinstance(tags, list):
                article_data["tags"] = []
            
            # Сохраняем в файл
            user_dir = self.data_dir / "user_uploads"
            user_dir.mkdir(exist_ok=True)
            
            filename = f"{article_id}.json"
            filepath = user_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, ensure_ascii=False, indent=2)
            
            # Добавляем в память
            self.user_articles.append(article_data)
            self.all_articles.append(article_data)
            
            # Добавляем в векторный индекс
            if self.use_vector_search:
                success = self._add_to_vector_index(article_data, user_id)
                if success:
                    logger.info(f"Статья {article_id} успешно добавлена в векторный индекс")
                else:
                    logger.warning(f"Статья {article_id} добавлена, но не проиндексирована в векторной БД")
            else:
                logger.info(f"Векторный поиск отключен, статья {article_id} добавлена только в текстовый поиск")
            
            logger.info(f"Добавлена пользовательская статья: {article_id}")
            return article_id
            
        except Exception as e:
            logger.error(f"Ошибка добавления статьи: {e}")
            return None
    
    def get_user_articles(self, user_id: str) -> List[Dict]:
        """Получение статей пользователя"""
        return [article for article in self.user_articles if article.get("uploaded_by") == user_id]
    
    def get_statistics(self, user_id: str = "") -> Dict:
        """Получение статистики системы"""
        if user_id:
            user_articles = [a for a in self.user_articles if a.get("uploaded_by") == user_id]
            current_user_articles = len(user_articles)
        else:
            current_user_articles = 0
        
        return {
            "total_articles": len(self.all_articles),
            "habr_articles": len(self.habr_articles),
            "user_articles": len(self.user_articles),
            "current_user_articles": current_user_articles,
            "vector_search_enabled": self.use_vector_search and self.vector_store is not None,
            "llm_enabled": self.llm_enabled and self.llm_client is not None,
            "total_chunks": self.total_chunks,
            "initialization_errors": self.initialization_errors,
            "last_update": datetime.now().isoformat()
        }
    
    def is_ready(self) -> bool:
        """Проверка готовности агента к работе"""
        return len(self.all_articles) > 0 and (
            not self.use_vector_search or self.vector_store is not None
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Получение подробного статуса системы"""
        return {
            "articles_loaded": len(self.all_articles),
            "habr_articles": len(self.habr_articles),
            "user_articles": len(self.user_articles),
            "vector_search_available": self.use_vector_search,
            "vector_store_initialized": self.vector_store is not None,
            "llm_available": self.llm_enabled,
            "llm_client_initialized": self.llm_client is not None,
            "embedder_initialized": self.embedder is not None,
            "chunker_initialized": self.chunker is not None,
            "ready": self.is_ready(),
            "errors": self.initialization_errors,
            "data_directory": str(self.data_dir),
            "vector_db_path": self.vector_db_path
        }
    
    def rebuild_index(self) -> Tuple[bool, str]:
        """Перестроение векторного индекса"""
        if not self.use_vector_search:
            return False, "Векторный поиск отключен"
        
        try:
            logger.info("Начинаю перестроение векторного индекса...")
            
            # Очищаем существующий индекс
            old_path = Path(self.vector_db_path)
            if old_path.exists():
                import shutil
                shutil.rmtree(old_path)
            
            # Пересоздаем векторное хранилище
            self.vector_store = VectorStore(path=self.vector_db_path)
            
            # Сбрасываем счетчик чанков
            self.total_chunks = 0
            
            # Индексируем все статьи Habr
            habr_count = 0
            for article in self.habr_articles:
                if self._add_to_vector_index(article):
                    habr_count += 1
            
            # Индексируем пользовательские статьи
            user_count = 0
            for article in self.user_articles:
                user_id = article.get("uploaded_by", "unknown")
                if self._add_to_vector_index(article, user_id):
                    user_count += 1
            
            message = f"Индекс перестроен: {habr_count} статей Habr, {user_count} пользовательских статей, {self.total_chunks} чанков"
            logger.info(message)
            return True, message
            
        except Exception as e:
            error_msg = f"Ошибка перестроения индекса: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def search_user_documents(self, query: str, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Поиск только в пользовательских документах (специальная функция для отладки)
        
        Args:
            query: Текст запроса
            user_id: ID пользователя
            limit: Максимальное количество результатов
            
        Returns:
            Список результатов поиска
        """
        logger.info(f"Поиск в пользовательских документах для пользователя {user_id}")
        
        # Получаем все документы пользователя
        user_docs = self.get_user_articles(user_id)
        logger.info(f"У пользователя {user_id} найдено {len(user_docs)} документов")
        
        # Выполняем поиск
        results = self.search(query, user_id, SearchScope.USER_ONLY, limit)
        
        logger.info(f"Поиск в пользовательских документах вернул {len(results)} результатов")
        
        # Дополнительная отладочная информация
        for i, result in enumerate(results[:3]):
            article = result["article"]
            logger.debug(f"Результат {i+1}: {article.get('title', 'Без названия')} (длина: {article.get('text_length', 0)})")
        
        return results

    def generate_document_tags(self, text: str, title: str = "", use_llm: bool = True) -> List[str]:
        """
        Генерация тегов для документа
        
        Args:
            text: Текст документа
            title: Заголовок документа
            use_llm: Использовать LLM для генерации (иначе простой метод)
            
        Returns:
            Список тегов
        """
        try:
            # Пробуем LLM если доступен и разрешено
            if use_llm and self.llm_enabled and self.llm_client and text:
                logger.info(f"Генерация тегов через LLM для документа: {title[:50]}...")
                
                # Вызываем LLMClient
                tags = self.llm_client.generate_tags_for_document(
                    text=text[:2000],  # Первые 2000 символов достаточно
                    title=title,
                    max_tags=5
                )
                
                if tags and len(tags) >= 2:
                    logger.info(f"Сгенерировано тегов через LLM: {tags}")
                    return tags
            
            # Fallback: простой метод извлечения ключевых слов
            logger.info(f"Использую простой метод генерации тегов для: {title[:50]}...")
            simple_tags = self._extract_tags_simple(text, title)
            
            return simple_tags[:5]
            
        except Exception as e:
            logger.error(f"Ошибка генерации тегов: {e}")
            return ["документ", "загружено"]

    def _extract_tags_simple(self, text: str, title: str = "") -> List[str]:
        """
        Простой метод извлечения тегов без LLM
        """
        try:
            import re
            from collections import Counter
            
            # Комбинируем заголовок и текст для анализа
            full_text = f"{title} {text}".lower()
            
            # Русские стоп-слова
            russian_stopwords = {
                'это', 'как', 'так', 'и', 'в', 'над', 'к', 'до', 'не', 'на', 'но', 'за', 'то', 'с', 
                'ли', 'а', 'во', 'от', 'со', 'для', 'о', 'же', 'ну', 'вы', 'бы', 'что', 'кто', 
                'он', 'она', 'оно', 'они', 'где', 'куда', 'когда', 'зачем', 'почему', 'сколько',
                'их', 'им', 'него', 'нее', 'них', 'мне', 'нам', 'мной', 'вами', 'вас', 'ваш',
                'твой', 'свой', 'наш', 'его', 'ее', 'их', 'мой', 'твой', 'свой', 'свои', 'своих',
                'который', 'которая', 'которые', 'которых', 'каждый', 'каждая', 'каждое', 'каждые',
                'весь', 'вся', 'все', 'всё', 'этот', 'эта', 'это', 'эти', 'тут', 'там', 'здесь',
                'вот', 'туда', 'сюда', 'оттуда', 'отсюда', 'сегодня', 'вчера', 'завтра', 'сейчас',
                'потом', 'опять', 'опять', 'уже', 'еще', 'уж', 'даже', 'ни', 'либо', 'или', 'ибо',
                'чтобы', 'чтоб', 'будто', 'как', 'как', 'так', 'такой', 'такая', 'такое', 'такие',
                'таких', 'таком', 'таким', 'такими', 'таков', 'такова', 'таково', 'таковы',
            }
            
            # Технические стоп-слова
            tech_stopwords = {
                'статья', 'документ', 'файл', 'текст', 'материал', 'раздел', 'глава',
                'страница', 'пример', 'использование', 'работа', 'метод', 'способ',
                'система', 'технология', 'информация', 'данные', 'процесс', 'решение'
            }
            
            all_stopwords = russian_stopwords.union(tech_stopwords)
            
            # Извлекаем слова (русские и английские, от 3 букв)
            words = re.findall(r'[а-яёa-z]{3,}', full_text)
            
            # Фильтруем стоп-слова
            filtered_words = [w for w in words if w not in all_stopwords]
            
            if not filtered_words:
                return ["технический", "документ"]
            
            # Частотный анализ
            word_freq = Counter(filtered_words)
            
            # Берем самые частые слова (но не слишком частые)
            common_words = []
            for word, freq in word_freq.most_common(20):
                # Пропускаем слишком частые слова (более 10% от общего числа)
                if freq < len(filtered_words) * 0.1:
                    common_words.append(word)
                if len(common_words) >= 8:
                    break
            
            # Категории на основе ключевых слов
            categories = []
            
            # Проверяем наличие ключевых слов для категорий
            tech_keywords = {
                'программирование': ['код', 'программа', 'алгоритм', 'функция', 'класс', 'объект'],
                'машинное обучение': ['модель', 'обучение', 'данные', 'нейрон', 'алгоритм', 'тренировка'],
                'базы данных': ['запрос', 'таблица', 'данные', 'sql', 'nosql', 'индекс'],
                'веб разработка': ['сайт', 'сервер', 'клиент', 'браузер', 'http', 'api'],
                'devops': ['контейнер', 'docker', 'kubernetes', 'deploy', 'ci/cd', 'мониторинг'],
                'безопасность': ['безопасность', 'шифрование', 'аутентификация', 'авторизация'],
                'cloud': ['облако', 'aws', 'azure', 'google cloud', 'инфраструктура'],
            }
            
            for category, keywords in tech_keywords.items():
                if any(keyword in full_text for keyword in keywords):
                    categories.append(category)
            
            # Комбинируем слова и категории
            tags = categories[:2] + common_words[:3]
            
            # Уникальные теги
            unique_tags = []
            for tag in tags:
                if tag not in unique_tags:
                    unique_tags.append(tag)
            
            return unique_tags[:5]
            
        except Exception as e:
            logger.warning(f"Ошибка в простом методе извлечения тегов: {e}")
            return ["технический", "документ"]
    
__all__ = ["RAGAgent", "SearchScope"]