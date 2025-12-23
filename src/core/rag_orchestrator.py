"""
Главный оркестратор RAG системы
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
import itertools

from .search_engine import SearchEngine, SearchScope
from .recommender import Recommender
from .document_manager import DocumentManager

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Главный оркестратор RAG системы"""
    
    def __init__(self, 
                 data_dir: str,
                 vector_db_path: str,
                 use_vector_search: bool = True,
                 llm_enabled: bool = True):
        """
        Args:
            data_dir: Директория с данными
            vector_db_path: Путь к векторной БД
            use_vector_search: Использовать векторный поиск
            llm_enabled: Использовать LLM
        """
        self.data_dir = Path(data_dir)
        self.vector_db_path = vector_db_path
        self.use_vector_search = use_vector_search
        self.llm_enabled = llm_enabled
        
        # Компоненты
        self.search_engine = None
        self.recommender = None
        self.document_manager = None
        self.llm_client = None
        
        # Данные
        self.habr_articles = []
        self.user_articles = []
        self.all_articles = []
        
        # Инициализация
        try:
            self._init_components()
            self._load_articles()
            logger.info(f"RAGOrchestrator успешно инициализирован: {len(self.all_articles)} статей")
        except Exception as e:
            logger.error(f"Ошибка инициализации RAGOrchestrator: {e}")
            raise
    
    def _init_components(self):
        """Инициализация компонентов системы"""
        logger.info("Инициализация компонентов RAG системы...")
        
        # Инициализация LLM клиента
        if self.llm_enabled:
            try:
                from src.generation.llm_client import LLMClient
                self.llm_client = LLMClient()
                logger.info("LLM клиент инициализирован")
            except ImportError as e:
                logger.error(f"Ошибка импорта LLMClient: {e}")
                self.llm_enabled = False
            except Exception as e:
                logger.error(f"Ошибка инициализации LLM клиента: {e}")
                self.llm_enabled = False
        
        # Инициализация компонентов для векторного поиска
        vector_store = None
        embedder = None
        chunker = None
        
        if self.use_vector_search:
            try:
                from src.ingest.embedder import Embedder
                from src.ingest.vector_store import VectorStore
                from src.ingest.chunker import TextChunker
                
                embedder = Embedder()
                vector_store = VectorStore(path=self.vector_db_path)
                chunker = TextChunker(chunk_size=800, chunk_overlap=150)
                logger.info("Компоненты векторного поиска инициализированы")
            except ImportError as e:
                logger.error(f"Ошибка импорта компонентов векторного поиска: {e}")
                self.use_vector_search = False
            except Exception as e:
                logger.error(f"Ошибка инициализации векторного поиска: {e}")
                self.use_vector_search = False
        
        # Инициализация менеджера документов
        try:
            self.document_manager = DocumentManager(
                data_dir=self.data_dir,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
                llm_client=self.llm_client
            )
            logger.info("Менеджер документов инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации менеджера документов: {e}")
            # Создаем заглушку
            self.document_manager = None
        
        # Инициализация поисковой системы
        try:
            self.search_engine = SearchEngine(
                vector_store=vector_store,
                embedder=embedder,
                all_articles=self.all_articles
            )
            logger.info("Поисковая система инициализирована")
        except Exception as e:
            logger.error(f"Ошибка инициализации поисковой системы: {e}")
            self.search_engine = None
        
        # Инициализация рекомендательной системы
        try:
            self.recommender = Recommender(
                vector_store=vector_store,
                embedder=embedder,
                all_articles=self.all_articles
            )
            logger.info("Рекомендательная система инициализирована")
        except Exception as e:
            logger.error(f"Ошибка инициализации рекомендательной системы: {e}")
            self.recommender = None
    
    def _load_articles(self, max_articles: int = 2000):
        """Загрузка статей из всех источников"""
        # Загрузка статей Habr
        habr_path = self.data_dir / "articles_batch.jsonl"
        if habr_path.exists():
            count = 0
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
                            logger.warning(f"Ошибка парсинга JSON: {e}")
                            continue
            
            logger.info(f"Загружено {count} статей Habr")
        else:
            logger.warning(f"Файл {habr_path} не найден")
        
        # Загрузка пользовательских статей (пока пусто, будут загружены по требованию)
        logger.info(f"Всего загружено статей: {len(self.all_articles)}")
    
    def _load_user_articles(self, user_id: str):
        """Загрузка документов пользователя в общий пул"""
        if not self.document_manager:
            return
        
        try:
            # Получаем документы пользователя
            user_docs = self.document_manager.get_user_documents(user_id)
            
            # Удаляем старые документы этого пользователя
            self.user_articles = [a for a in self.user_articles 
                                 if a.get("uploaded_by") != user_id]
            
            # Обновляем all_articles: удаляем старые документы этого пользователя
            self.all_articles = [a for a in self.all_articles 
                                if not a.get("is_user_document") or 
                                a.get("uploaded_by") != user_id]
            
            # Добавляем новые документы
            self.user_articles.extend(user_docs)
            self.all_articles.extend(user_docs)
            
            # Обновляем компоненты
            if self.search_engine:
                self.search_engine.all_articles = self.all_articles
            
            if self.recommender:
                self.recommender.all_articles = self.all_articles
                self.recommender.article_index = {a["id"]: a for a in self.all_articles}
            
            logger.info(f"Загружено {len(user_docs)} документов пользователя {user_id}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки документов пользователя {user_id}: {e}")
    
    def search(self, 
               query: str, 
               user_id: str, 
               scope: SearchScope = SearchScope.ALL, 
               limit: int = 10,
               tags: Optional[List[str]] = None,
               author: Optional[str] = None,
               date_from: Optional[str] = None,
               date_to: Optional[str] = None) -> List[Dict]:
        """Умный поиск статей"""
        # Загружаем документы пользователя
        self._load_user_articles(user_id)
        
        if not self.search_engine:
            logger.error("Поисковая система не инициализирована")
            return []
        
        return self.search_engine.search(
            query, user_id, scope, limit, 
            tags, author, date_from, date_to
        )
    
    def get_similar_articles(self, 
                            article_id: str, 
                            user_id: str = None, 
                            limit: int = 5) -> List[Dict]:
        """Получение похожих статей"""
        if not self.recommender:
            logger.error("Рекомендательная система не инициализирована")
            return []
        
        # Загружаем документы пользователя
        if user_id:
            self._load_user_articles(user_id)
        
        return self.recommender.get_similar_articles(article_id, user_id, limit)
    
    def add_user_document(self, 
                         document_data: Dict, 
                         user_id: str) -> Tuple[bool, str, Optional[str]]:
        """Добавление пользовательского документа"""
        if not self.document_manager:
            return False, "Менеджер документов не инициализирован", None
        
        success, message, doc_id = self.document_manager.add_user_document(
            document_data, user_id, generate_tags=True
        )
        
        # Обновляем списки статей
        if success:
            self._load_user_articles(user_id)
        
        return success, message, doc_id
    
    def get_user_documents(self, user_id: str) -> List[Dict]:
        """Получение документов пользователя"""
        if not self.document_manager:
            return []
        
        return self.document_manager.get_user_documents(user_id)
    
    def get_user_document_stats(self, user_id: str) -> Dict:
        """Статистика документов пользователя"""
        if not self.document_manager:
            return {"count": 0, "total_size": 0}
        
        return self.document_manager.get_document_stats(user_id)
    
    def delete_user_document(self, document_id: str, user_id: str) -> Tuple[bool, str]:
        """Удаление документа пользователя"""
        if not self.document_manager:
            return False, "Менеджер документов не инициализирован"
        
        success, message = self.document_manager.delete_user_document(document_id, user_id)
        
        # Обновляем списки статей
        if success:
            self._load_user_articles(user_id)
        
        return success, message
    
    def generate_answer(self, 
                       query: str, 
                       user_id: str, 
                       scope: SearchScope = SearchScope.ALL) -> Dict:
        """Генерация ответа на запрос"""
        # Поиск релевантных статей
        search_results = self.search(query, user_id, scope, limit=5)
        
        if not search_results:
            return {
                "answer": f"По вашему запросу '{query}' ничего не найдено.",
                "sources": [],
                "questions": []
            }
        
        # Используем LLM если доступен
        if self.llm_enabled and self.llm_client:
            return self._generate_with_llm(query, search_results)
        else:
            return self._generate_simple_answer(query, search_results)
    
    def _generate_with_llm(self, query: str, search_results: List[Dict]) -> Dict:
        """Генерация ответа через LLM"""
        try:
            # Подготавливаем контекст
            context_chunks = []
            for result in search_results[:3]:
                article = result["article"]
                context_chunks.append({
                    "text": article.get("text", "")[:1000],
                    "metadata": {
                        "title": article.get("title", ""),
                        "author": article.get("author", ""),
                        "source": "ваш документ" if result.get("is_user_document") else "статья Habr"
                    }
                })
            
            # Генерируем ответ
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
                    "is_user_document": result.get("is_user_document", False)
                })
            
            # Генерация вопросов
            questions = []
            if search_results and self.llm_client:
                top_article_text = search_results[0]["article"].get("text", "")[:500]
                questions = self.llm_client.generate_questions(
                    article_text=top_article_text,
                    num_questions=5
                )
            
            return {
                "answer": llm_result.get("answer", "Не удалось сгенерировать ответ"),
                "sources": sources,
                "questions": questions
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации через LLM: {e}")
            return self._generate_simple_answer(query, search_results)
    
    def _generate_simple_answer(self, query: str, search_results: List[Dict]) -> Dict:
        """Простая генерация ответа без LLM"""
        answer_lines = [
            f"🔍 **По вашему запросу '{query}' найдено {len(search_results)} материалов:**\n"
        ]
        
        for i, result in enumerate(search_results[:3], 1):
            article = result["article"]
            source_type = "📁 Ваш документ" if result.get("is_user_document") else "🌐 Habr"
            
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
                "is_user_document": result.get("is_user_document", False)
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
    
    def get_all_tags(self, user_id: str = None) -> List[str]:
        """Получение всех уникальных тегов"""
        all_tags = set()
        
        for article in self.all_articles:
            # Фильтрация по пользователю
            if user_id and article.get("is_user_document"):
                if article.get("uploaded_by") != user_id:
                    continue
            
            tags = article.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    if tag and str(tag).strip():
                        all_tags.add(str(tag).strip())
            elif isinstance(tags, str) and tags.strip():
                all_tags.add(tags.strip())
        
        return sorted(list(all_tags))
    
    def get_all_authors(self, user_id: str = None) -> List[str]:
        """Получение всех уникальных авторов"""
        authors = set()
        
        for article in self.all_articles:
            # Фильтрация по пользователю
            if user_id and article.get("is_user_document"):
                if article.get("uploaded_by") != user_id:
                    continue
            
            author = article.get("author", "")
            if author and author.strip():
                author_clean = str(author).strip()
                if author_clean.lower() not in ['неизвестен', 'unknown', 'n/a', '']:
                    authors.add(author_clean)
        
        return sorted(list(authors))
    
    def get_statistics(self, user_id: str = "") -> Dict:
        """Получение статистики системы"""
        # Загружаем документы пользователя для точной статистики
        user_docs_count = 0
        if user_id and self.document_manager:
            user_docs = self.document_manager.get_user_documents(user_id)
            user_docs_count = len(user_docs)
        
        return {
            "total_articles": len(self.all_articles),
            "habr_articles": len(self.habr_articles),
            "user_articles": len(self.user_articles),
            "current_user_articles": user_docs_count,
            "vector_search_enabled": self.use_vector_search and self.search_engine is not None,
            "llm_enabled": self.llm_enabled and self.llm_client is not None,
            "search_engine_ready": self.search_engine is not None,
            "recommender_ready": self.recommender is not None,
            "document_manager_ready": self.document_manager is not None,
            "last_update": datetime.now().isoformat()
        }
    
    def is_ready(self) -> bool:
        """Проверка готовности системы"""
        return len(self.all_articles) > 0 and self.search_engine is not None
    
    def rebuild_index(self) -> Tuple[bool, str]:
        """Перестроение векторного индекса"""
        if not self.use_vector_search:
            return False, "Векторный поиск отключен"
        
        try:
            logger.info("Начинаю перестроение векторного индекса...")
            
            # Очищаем старый индекс
            import shutil
            old_path = Path(self.vector_db_path)
            if old_path.exists():
                shutil.rmtree(old_path)
            
            # Инициализируем заново
            self._init_components()
            
            return True, "Векторный индекс перестроен"
            
        except Exception as e:
            logger.error(f"Ошибка перестроения индекса: {e}")
            return False, f"Ошибка: {str(e)}"
        
    def search_with_filters(self, 
                           query: str, 
                           user_id: str, 
                           scope = None, 
                           limit: int = 10,
                           tags = None,
                           author = None,
                           date_from = None,
                           date_to = None):
        """
        Метод для обратной совместимости.
        Используйте search() вместо search_with_filters().
        """
        import warnings
        warnings.warn(
            "search_with_filters() deprecated, use search() instead",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Преобразуем scope если нужно
        if scope is None:
            from .search_engine import SearchScope
            scope = SearchScope.ALL
        
        return self.search(
            query=query,
            user_id=user_id,
            scope=scope,
            limit=limit,
            tags=tags,
            author=author,
            date_from=date_from,
            date_to=date_to
        )