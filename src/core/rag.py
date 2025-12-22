"""
src/core/rag.py - Основной RAG агент с поддержкой пользовательских документов, областей поиска и рекомендаций
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

# Настройка логирования
logger = logging.getLogger(__name__)

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


class SearchScope(Enum):
    """Области поиска"""
    ALL = "all"
    USER_ONLY = "user"
    HABR_ONLY = "habr"


class RAGAgent:
    """Основной RAG агент с поддержкой пользовательских документов, векторного поиска и рекомендаций"""
    
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
        
        # Проверка качества данных (для отладки)
        self._check_data_quality()
        
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
    
    def _check_data_quality(self):
        """Проверка качества данных для отладки"""
        print("\n" + "="*50)
        print("ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
        print("="*50)
        
        # Проверяем первые 5 статей Habr
        print("\n📊 Первые 5 статей Habr:")
        for i, article in enumerate(self.habr_articles[:5]):
            print(f"\n{i+1}. {article.get('title', 'Без названия')[:50]}...")
            print(f"   Автор: '{article.get('author', 'НЕТ')}'")
            print(f"   Теги: {article.get('tags', [])}")
            print(f"   Тип тегов: {type(article.get('tags'))}")
        
        # Проверяем общую статистику
        print(f"\n📈 Общая статистика:")
        print(f"   Всего статей: {len(self.all_articles)}")
        print(f"   Статей Habr: {len(self.habr_articles)}")
        print(f"   Пользовательских: {len(self.user_articles)}")
        
        # Проверяем авторов
        authors_set = set()
        for article in self.all_articles:
            author = article.get('author', '')
            if author and str(author).strip():
                authors_set.add(str(author).strip())
        
        print(f"\n👥 Авторы:")
        print(f"   Уникальных авторов: {len(authors_set)}")
        if authors_set:
            print(f"   Примеры авторов: {list(authors_set)[:5]}")
        
        # Проверяем теги
        tags_set = set()
        for article in self.all_articles:
            tags = article.get('tags', [])
            if isinstance(tags, list):
                for tag in tags:
                    if tag and str(tag).strip():
                        tags_set.add(str(tag).strip())
            elif isinstance(tags, str):
                if tags.strip():
                    tags_set.add(tags.strip())
        
        print(f"\n🏷️ Теги:")
        print(f"   Уникальных тегов: {len(tags_set)}")
        if tags_set:
            print(f"   Примеры тегов: {list(tags_set)[:10]}")
    
    def _check_readiness(self):
        """Проверка готовности системы"""
        if not self.all_articles:
            self.initialization_errors.append("Нет загруженных статей")
        
        if self.use_vector_search and not self.vector_store:
            self.initialization_errors.append("Векторное хранилище не инициализировано")
        
        if self.llm_enabled and not self.llm_client:
            self.initialization_errors.append("LLM клиент не инициализирован")
    
    def _load_articles(self, max_articles: int = 2000):
        """Загрузка статей из всех источников"""
        try:
            # Загрузка статей Habr
            habr_path = self.data_dir / "articles_batch.jsonl"
            if habr_path.exists():
                count = 0
                import itertools
                with open(habr_path, 'r', encoding='utf-8') as f:
                    for line in itertools.islice(f, max_articles):
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
            # Генерируем эмбеддинг запроса
            query_embedding_result = self.embedder.embed([query])
            
            if query_embedding_result is None or len(query_embedding_result) == 0:
                logger.error("Не удалось создать эмбеддинг для запроса")
                return []
            
            # Извлекаем одномерный массив эмбеддинга
            query_embedding = query_embedding_result[0]  # Одномерный numpy array
            
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
                query_embedding, 
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
                                scope: SearchScope = SearchScope.ALL,
                                tags: Optional[List[str]] = None,
                                author: Optional[str] = None,
                                date_from: Optional[str] = None,
                                date_to: Optional[str] = None) -> Dict:
        """
        Генерация ответа с применением фильтров
        """
        # Поиск с фильтрами
        search_results = self.search_with_filters(
            query, user_id, scope, limit=5,
            tags=tags, author=author, date_from=date_from, date_to=date_to
        )
        
        # Остальной код такой же как в generate_answer
        if not search_results:
            return {
                "answer": f"По вашему запросу '{query}' с указанными фильтрами ничего не найдено.",
                "sources": [],
                "questions": []
            }
        
        # Используем LLM для генерации ответа
        if self.llm_enabled and self.llm_client:
            return self._generate_with_llm(query, search_results)
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

            # Используем LLM для генерации теги
            use_llm_for_tags = True
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
                logger.info(f"Генерация теги через LLM для документа: {title[:50]}...")
                
                # Вызываем LLMClient
                tags = self.llm_client.generate_tags_for_document(
                    text=text[:2000],  # Первые 2000 символов достаточно
                    title=title,
                    max_tags=5
                )
                
                if tags and len(tags) >= 2:
                    logger.info(f"Сгенерировано теги через LLM: {tags}")
                    return tags
            
            # Fallback: простой метод извлечения ключевых слов
            logger.info(f"Использую простой метод генерации теги для: {title[:50]}...")
            simple_tags = self._extract_tags_simple(text, title)
            
            return simple_tags[:5]
            
        except Exception as e:
            logger.error(f"Ошибка генерации теги: {e}")
            return ["документ", "загружено"]

    def _extract_tags_simple(self, text: str, title: str = "") -> List[str]:
        """
        Простой метод извлечения теги без LLM
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
            logger.warning(f"Ошибка в простом методе извлечения теги: {e}")
            return ["технический", "документ"]
        
    def search_with_filters(self, 
                        query: str, 
                        user_id: str, 
                        scope: SearchScope = SearchScope.ALL, 
                        limit: int = 10,
                        tags: Optional[List[str]] = None,
                        author: Optional[str] = None,
                        date_from: Optional[str] = None,
                        date_to: Optional[str] = None) -> List[Dict]:
        """
        Поиск с фильтрами по тегам, автору и дате
        """
        logger.info("=== ПОИСК С ФИЛЬТРАМИ ===")
        logger.info(f"Запрос: {query}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Scope: {scope}")
        logger.info(f"Теги: {tags}")
        logger.info(f"Автор: {author}")
        logger.info(f"Дата от: {date_from}")
        logger.info(f"Дата до: {date_to}")
        
        # Сначала обычный поиск
        logger.info("Выполняю обычный поиск...")
        search_results = self.search(query, user_id, scope, limit * 10)  # Увеличиваем в 10 раз для фильтрации
        logger.info(f"Найдено результатов до фильтрации: {len(search_results)}")
        
        if not search_results:
            logger.info("Нет результатов для фильтрации")
            return []
        
        # Применяем фильтры
        filtered_results = []
        
        for result in search_results:
            article = result["article"]
            include = True
            
            # Фильтр по тегам
            if tags and tags != [] and tags != [""]:
                article_tags = article.get("tags", [])
                
                # Нормализуем теги статьи
                if isinstance(article_tags, str):
                    article_tags = [t.strip().lower() for t in article_tags.split(',') if t.strip()]
                elif isinstance(article_tags, list):
                    article_tags = [str(t).strip().lower() for t in article_tags if t]
                else:
                    article_tags = []
                
                # Нормализуем запрошенные теги
                norm_tags = [t.strip().lower() for t in tags if t and str(t).strip()]
                
                # Проверяем, есть ли хотя бы один из запрошенных тегов
                if norm_tags and not any(tag in article_tags for tag in norm_tags):
                    include = False
                    logger.debug(f"Статья '{article.get('title')}' исключена по тегам")
            
            # Фильтр по автору
            if author and author.strip():
                article_author = str(article.get("author", "")).lower()
                if author.lower() not in article_author:
                    include = False
                    logger.debug(f"Статья '{article.get('title')}' исключена по автору")
            
            # Фильтр по дате
            article_date = article.get("date", "")
            if article_date and (date_from or date_to):
                try:
                    date_str = str(article_date)
                    date_obj = None
                    
                    # Пробуем разные форматы
                    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                        try:
                            date_obj = datetime.strptime(date_str[:10], "%Y-%m-%d")
                            break
                        except:
                            continue
                    
                    if date_obj:
                        if date_from:
                            try:
                                filter_from = datetime.strptime(date_from, "%Y-%m-%d")
                                if date_obj.date() < filter_from.date():
                                    include = False
                                    logger.debug(f"Статья '{article.get('title')}' исключена по дате (слишком старая)")
                            except:
                                pass
                        
                        if date_to:
                            try:
                                filter_to = datetime.strptime(date_to, "%Y-%m-%d")
                                if date_obj.date() > filter_to.date():
                                    include = False
                                    logger.debug(f"Статья '{article.get('title')}' исключена по дате (слишком новая)")
                            except:
                                pass
                                
                except Exception as e:
                    logger.warning(f"Ошибка фильтрации даты {article_date}: {e}")
            
            if include:
                filtered_results.append(result)
                
                if len(filtered_results) >= limit:
                    break
        
        logger.info(f"После фильтров: {len(filtered_results)} из {len(search_results)} результатов")
        
        # Вывести информацию о первых 3 результатах
        for i, result in enumerate(filtered_results[:3]):
            article = result["article"]
            logger.info(f"Результат {i+1}: {article.get('title')}")
            logger.info(f"  Автор: {article.get('author')}")
            logger.info(f"  Теги: {article.get('tags')}")
            logger.info(f"  Дата: {article.get('date')}")
        
        return filtered_results[:limit]

    def get_all_tags(self, user_id: str = None) -> List[str]:
        """
        Получение всех уникальных тегов из статей
        
        Args:
            user_id: Если указан, возвращает теги только для пользователя
            
        Returns:
            Список уникальных тегов
        """
        all_tags = set()
        
        # Используем все статьи для глобальных тегов
        articles_to_check = self.all_articles
        
        for article in articles_to_check:
            # Пропускаем статьи другого пользователя, если user_id указан
            if user_id and article.get("is_user_document"):
                if article.get("uploaded_by") != user_id:
                    continue
            
            tags = article.get("tags", [])
            
            # ВАЖНО: В статьях Habr теги хранятся как список строк
            # В пользовательских документах могут быть как список, так и строка
            
            if isinstance(tags, str):
                # Разделяем строку тегов
                tag_list = [t.strip() for t in tags.split(',') if t.strip()]
                for tag in tag_list:
                    if tag and len(tag) < 50 and tag.lower() not in ['', 'none', 'nan']:
                        all_tags.add(tag)
            elif isinstance(tags, list):
                for tag in tags:
                    tag_str = str(tag).strip()
                    if tag_str and len(tag_str) < 50 and tag_str.lower() not in ['', 'none', 'nan']:
                        all_tags.add(tag_str)
            elif tags:  # Любой другой тип
                tag_str = str(tags).strip()[:50]
                if tag_str:
                    all_tags.add(tag_str)
        
        # Сортируем и возвращаем
        return sorted(list(all_tags))

    def get_all_authors(self, user_id: str = None) -> List[str]:
        """
        Получение всех уникальных авторов
        
        Args:
            user_id: Если указан, возвращает авторов только для пользователя
            
        Returns:
            Список уникальных авторов
        """
        authors = set()
        
        # Используем все статьи для глобальных авторов
        articles_to_check = self.all_articles
        
        for article in articles_to_check:
            # Пропускаем статьи другого пользователя, если user_id указан
            if user_id and article.get("is_user_document"):
                if article.get("uploaded_by") != user_id:
                    continue
            
            author = article.get("author", "")
            
            if author:
                # Очищаем авторское имя
                author_clean = str(author).strip()
                
                # Убираем пустые значения и дефолтные
                if author_clean and len(author_clean) > 1:
                    # Убираем префиксы типа @
                    if author_clean.startswith('@'):
                        author_clean = author_clean[1:].strip()
                    
                    # Убираем "неизвестен", "unknown" и т.д.
                    lower_author = author_clean.lower()
                    if lower_author not in ['', 'неизвестен', 'неизвестный', 'unknown', 'n/a', 'none', 'автор не указан']:
                        authors.add(author_clean)
        
        # Сортируем и возвращаем
        return sorted(list(authors))
    
    # ==================== МЕТОДЫ ДЛЯ РЕКОМЕНДАЦИЙ ====================
    
    def get_similar_articles(self, 
                            article_id: str, 
                            user_id: str = None, 
                            limit: int = 5,
                            similarity_threshold: float = 0.6) -> List[Dict]:
        """
        Получение похожих статей на основе контента и метаданных
        
        Args:
            article_id: ID целевой статьи
            user_id: ID пользователя (для фильтрации)
            limit: Количество рекомендаций
            similarity_threshold: Порог схожести (0-1)
            
        Returns:
            Список похожих статей с оценкой схожести
        """
        try:
            logger.info(f"Поиск похожих статей для {article_id}, пользователь: {user_id}")
            
            # Находим целевую статью
            target_article = None
            for article in self.all_articles:
                if article.get("id") == article_id:
                    target_article = article
                    break
            
            if not target_article:
                logger.warning(f"Статья {article_id} не найдена")
                return []
            
            # Если векторный поиск недоступен, используем метаданные
            if not self.use_vector_search or not self.vector_store or not self.embedder:
                logger.info("Векторный поиск недоступен, использую поиск по метаданным")
                return self._get_similar_by_metadata(target_article, user_id, limit)
            
            # Подготавливаем текст для эмбеддинга
            target_text = self._prepare_text_for_embedding(target_article)
            logger.debug(f"Подготовленный текст для эмбеддинга: {len(target_text)} символов")
            
            # Генерируем эмбеддинг
            try:
                # embed() возвращает numpy array с формой (1, embedding_dim) для одного текста
                query_embedding_result = self.embedder.embed([target_text])
                
                if query_embedding_result is None or len(query_embedding_result) == 0:
                    logger.error("Не удалось создать эмбеддинг для статьи")
                    return self._get_similar_by_metadata(target_article, user_id, limit)
                
                # Извлекаем эмбеддинг - это одномерный массив
                query_embedding = query_embedding_result[0]  # Это одномерный numpy array
                
                logger.debug(f"Размер эмбеддинга: {query_embedding.shape}")
                
            except Exception as e:
                logger.error(f"Ошибка создания эмбеддинга: {e}")
                return self._get_similar_by_metadata(target_article, user_id, limit)
            
            # Выполняем поиск похожих статей
            try:
                # Ищем больше результатов, чтобы отфильтровать
                search_results = self.vector_store.search(
                    query_embedding, 
                    top_k=limit * 3  # Ищем больше для фильтрации
                )
                
                if not search_results or not search_results.get("documents"):
                    logger.info("Векторный поиск не дал результатов")
                    return self._get_similar_by_metadata(target_article, user_id, limit)
                
                # Обрабатываем результаты
                similar_articles = []
                
                # ChromaDB возвращает списки в списке: documents[0] - результаты для первого запроса
                docs = search_results["documents"][0] if search_results.get("documents") else []
                metas = search_results["metadatas"][0] if search_results.get("metadatas") else []
                distances = search_results.get("distances", [[]])[0] if search_results.get("distances") else []
                
                logger.debug(f"Найдено {len(docs)} потенциально похожих чанков")
                
                seen_articles = set()  # Для отслеживания уже добавленных статей
                
                for i, (doc, meta) in enumerate(zip(docs, metas)):
                    # Пропускаем чанки из той же статьи
                    if meta.get("article_id") == article_id:
                        continue
                    
                    article_id_from_meta = meta.get("article_id")
                    if not article_id_from_meta:
                        continue
                    
                    # Проверяем схожесть (если есть расстояния)
                    if distances and i < len(distances):
                        similarity = 1 - distances[i]  # Преобразуем расстояние в схожесть
                        if similarity < similarity_threshold:
                            continue
                    else:
                        similarity = 0.7  # Значение по умолчанию
                    
                    # Проверяем доступность для пользователя
                    if user_id and meta.get("source_type") == "user":
                        if meta.get("uploaded_by") != user_id:
                            continue
                    
                    # Добавляем статью, если еще не добавляли
                    if article_id_from_meta not in seen_articles:
                        # Ищем статью в памяти
                        similar_article = self._get_article_by_id(article_id_from_meta)
                        if similar_article:
                            seen_articles.add(article_id_from_meta)
                            
                            # Вычисляем причину схожести
                            reason = self._get_similarity_reason(target_article, similar_article)
                            
                            similar_articles.append({
                                "article": similar_article,
                                "similarity_score": similarity,
                                "reason": reason,
                                "source": "vector_search"
                            })
                            
                            if len(similar_articles) >= limit:
                                break
                
                # Если векторный поиск дал мало результатов, дополняем метаданными
                if len(similar_articles) < limit:
                    logger.info(f"Векторный поиск дал {len(similar_articles)} результатов, дополняю метаданными")
                    metadata_results = self._get_similar_by_metadata(target_article, user_id, limit - len(similar_articles))
                    
                    # Добавляем только уникальные статьи
                    for meta_result in metadata_results:
                        meta_article_id = meta_result["article"].get("id")
                        if meta_article_id not in seen_articles and meta_article_id != article_id:
                            similar_articles.append(meta_result)
                            seen_articles.add(meta_article_id)
                            
                            if len(similar_articles) >= limit:
                                break
                
                # Сортируем по схожести
                similar_articles.sort(key=lambda x: x["similarity_score"], reverse=True)
                
                logger.info(f"Найдено {len(similar_articles)} похожих статей для {article_id}")
                return similar_articles[:limit]
                
            except Exception as e:
                logger.error(f"Ошибка векторного поиска похожих статей: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return self._get_similar_by_metadata(target_article, user_id, limit)
        
        except Exception as e:
            logger.error(f"Критическая ошибка в get_similar_articles: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _prepare_text_for_embedding(self, article: Dict) -> str:
        """Подготовка текста статьи для эмбеддинга"""
        try:
            title = article.get("title", "")
            text = article.get("text", "")
            tags = article.get("tags", [])
            
            # Ограничиваем длину текста для эмбеддинга (оптимально 500-1000 символов)
            text_preview = text[:800] if len(text) > 800 else text
            
            # Обрабатываем теги
            if isinstance(tags, list):
                tags_text = " ".join([str(tag) for tag in tags[:10] if tag])
            elif isinstance(tags, str):
                tags_text = tags
            else:
                tags_text = ""
            
            # Комбинируем все компоненты
            combined = f"{title} {tags_text} {text_preview}"
            
            # Очищаем и нормализуем текст
            import re
            combined = re.sub(r'\s+', ' ', combined)  # Убираем лишние пробелы
            combined = combined.strip()
            
            logger.debug(f"Подготовлен текст для эмбеддинга: {len(combined)} символов")
            return combined
            
        except Exception as e:
            logger.error(f"Ошибка подготовки текста для эмбеддинга: {e}")
            return article.get("title", "") + " " + (article.get("text", "")[:500] or "")
    
    def _get_similar_by_metadata(self, target_article: Dict, user_id: str = None, limit: int = 5) -> List[Dict]:
        """Поиск похожих статей по метаданным (fallback)"""
        try:
            logger.info(f"Поиск похожих по метаданным для статьи: {target_article.get('title', '')[:50]}")
            
            target_tags = set()
            if isinstance(target_article.get("tags"), list):
                target_tags = {str(tag).lower().strip() for tag in target_article.get("tags", []) if tag}
            elif isinstance(target_article.get("tags"), str):
                target_tags = {tag.strip().lower() for tag in target_article.get("tags", "").split(",") if tag.strip()}
            
            target_author = str(target_article.get("author", "")).lower().strip()
            target_source = target_article.get("source", "")
            
            candidates = []
            
            for article in self.all_articles:
                # Пропускаем ту же статью
                if article.get("id") == target_article.get("id"):
                    continue
                
                # Проверяем доступность для пользователя
                if user_id and article.get("is_user_document"):
                    if article.get("uploaded_by") != user_id:
                        continue
                
                # Вычисляем схожесть по метаданным
                similarity_score = 0
                reasons = []
                
                # По тегам (самый важный фактор)
                article_tags = set()
                if isinstance(article.get("tags"), list):
                    article_tags = {str(tag).lower().strip() for tag in article.get("tags", []) if tag}
                elif isinstance(article.get("tags"), str):
                    article_tags = {tag.strip().lower() for tag in article.get("tags", "").split(",") if tag.strip()}
                
                common_tags = target_tags.intersection(article_tags)
                if common_tags:
                    tag_similarity = len(common_tags) / max(len(target_tags), 1) * 0.5
                    similarity_score += min(tag_similarity, 0.5)
                    if common_tags:
                        reasons.append(f"общие теги: {', '.join(list(common_tags)[:2])}")
                
                # По автору
                article_author = str(article.get("author", "")).lower().strip()
                if target_author and article_author and target_author == article_author:
                    similarity_score += 0.3
                    reasons.append(f"один автор: {target_author}")
                
                # По источнику
                if target_source and article.get("source") == target_source:
                    similarity_score += 0.1
                    reasons.append(f"один источник: {target_source}")
                
                # По дате (близкие даты) - только если обе даты есть
                try:
                    target_date = target_article.get("date", "")
                    article_date = article.get("date", "")
                    if target_date and article_date and len(target_date) >= 10 and len(article_date) >= 10:
                        # Проверяем, что даты в одном году
                        if target_date[:4] == article_date[:4]:
                            similarity_score += 0.05
                            if not reasons:
                                reasons.append("публикация в одном году")
                except:
                    pass
                
                if similarity_score > 0.2:  # Минимальный порог
                    candidates.append({
                        "article": article,
                        "similarity_score": min(similarity_score, 1.0),
                        "reason": "; ".join(reasons) if reasons else "тематическая схожесть",
                        "source": "metadata"
                    })
            
            # Сортируем по схожести
            candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            logger.info(f"Найдено {len(candidates)} кандидатов по метаданным")
            return candidates[:limit]
            
        except Exception as e:
            logger.error(f"Ошибка поиска по метаданным: {e}")
            return []
    
    def _get_article_by_id(self, article_id: str) -> Optional[Dict]:
        """Поиск статьи по ID"""
        for article in self.all_articles:
            if article.get("id") == article_id:
                return article
        return None
    
    def _get_similarity_reason(self, article1: Dict, article2: Dict) -> str:
        """Определение причины схожести для пользователя"""
        reasons = []
        
        # Проверка тегов
        tags1 = set()
        tags2 = set()
        
        if isinstance(article1.get("tags"), list):
            tags1 = {str(tag).lower().strip() for tag in article1.get("tags", []) if tag}
        elif isinstance(article1.get("tags"), str):
            tags1 = {tag.strip().lower() for tag in article1.get("tags", "").split(",") if tag.strip()}
        
        if isinstance(article2.get("tags"), list):
            tags2 = {str(tag).lower().strip() for tag in article2.get("tags", []) if tag}
        elif isinstance(article2.get("tags"), str):
            tags2 = {tag.strip().lower() for tag in article2.get("tags", "").split(",") if tag.strip()}
        
        common_tags = tags1.intersection(tags2)
        if common_tags:
            tags_list = list(common_tags)[:3]
            reasons.append(f"общие теги: {', '.join(tags_list)}")
        
        # Проверка автора
        author1 = str(article1.get("author", "")).strip()
        author2 = str(article2.get("author", "")).strip()
        if author1 and author2 and author1.lower() == author2.lower():
            reasons.append(f"один автор: {author1}")
        
        # Проверка источника
        source1 = article1.get("source", "")
        source2 = article2.get("source", "")
        if source1 and source2 and source1 == source2:
            reasons.append(f"один источник: {source1}")
        
        # Проверка даты (если есть)
        try:
            date1 = article1.get("date", "")[:10]
            date2 = article2.get("date", "")[:10]
            if date1 and date2 and len(date1) == 10 and len(date2) == 10:
                if date1[:7] == date2[:7]:  # Год-месяц
                    reasons.append("публикация в одном месяце")
                elif date1[:4] == date2[:4]:  # Год
                    reasons.append("публикация в одном году")
        except:
            pass
        
        return "; ".join(reasons) if reasons else "тематическая схожесть"


__all__ = ["RAGAgent", "SearchScope"]