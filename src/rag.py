import pytest
import sys
from pathlib import Path
import json
import tempfile
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class SearchScope(Enum):
    """Область поиска"""

    ALL = "all"  # Все источники
    USER_ONLY = "user"  # Только документы пользователя
    HABR_ONLY = "habr"  # Только статьи Habr


class RAGAgent:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Поддиректории
        self.habr_dir = self.data_dir / "habr_articles"
        self.uploads_dir = self.data_dir / "user_uploads"
        self.vector_db_dir = self.data_dir / "vector_db"

        for directory in [self.habr_dir, self.uploads_dir, self.vector_db_dir]:
            directory.mkdir(exist_ok=True, parents=True)

        # Загружаем статьи
        self.habr_articles = self._load_habr_articles()
        self.user_articles = self._load_user_articles()
        self.all_articles = self.habr_articles + self.user_articles

        print(f"Загружено {len(self.habr_articles)} статей Habr")
        print(f"Загружено {len(self.user_articles)} пользовательских документов")
        print(f"Всего документов в системе: {len(self.all_articles)}")

    def _load_habr_articles(self) -> List[Dict[str, Any]]:
        """Загружает статьи Habr из JSONL файлов"""
        articles = []

        # Ищем JSONL файлы
        search_paths = [
            self.data_dir / "habr_articles.jsonl",
            self.data_dir / "articles_batch.jsonl",
            self.habr_dir / "*.jsonl",
        ]

        for path_pattern in search_paths:
            if "*" in str(path_pattern):
                for jsonl_file in Path(".").glob(str(path_pattern)):
                    articles.extend(self._load_jsonl_file(jsonl_file))
            elif path_pattern.exists():
                articles.extend(self._load_jsonl_file(path_pattern))

        # Уникализация по ID
        unique_articles = {}
        for article in articles:
            if article.get("has_content", True) and len(article.get("text", "")) > 100:
                if "id" not in article or not article["id"]:
                    article["id"] = self._generate_article_id(
                        article.get("url", ""), "habr"
                    )
                unique_articles[article["id"]] = article

        return list(unique_articles.values())

    def _load_user_articles(self) -> List[Dict[str, Any]]:
        """Загружает пользовательские статьи"""
        articles = []

        if not self.uploads_dir.exists():
            return articles

        for user_dir in self.uploads_dir.iterdir():
            if user_dir.is_dir() and user_dir.name.startswith("user_"):
                user_id = user_dir.name.replace("user_", "")
                user_articles_file = user_dir / "articles.jsonl"

                if user_articles_file.exists():
                    user_articles = self._load_jsonl_file(user_articles_file)
                    for article in user_articles:
                        article["user_id"] = user_id
                        article["source"] = "User Upload"
                        articles.append(article)

        return articles

    def _load_jsonl_file(self, path: Path) -> List[Dict[str, Any]]:
        """Загружает JSONL файл"""
        articles = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            article = json.loads(line)
                            articles.append(article)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Ошибка загрузки {path}: {e}")

        return articles

    def _generate_article_id(self, url: str, source: str) -> str:
        """Генерирует уникальный ID для статьи"""
        if not url:
            url = str(datetime.now().timestamp())
        hash_str = hashlib.md5(f"{source}_{url}".encode()).hexdigest()[:12]
        return f"{source[:3]}_{hash_str}"

    def add_user_article(self, article_data: Dict[str, Any], user_id: str) -> str:
        """Добавляет пользовательскую статью"""
        if "id" not in article_data or not article_data["id"]:
            article_data["id"] = self._generate_article_id(
                article_data.get("url", f"user_{user_id}_{datetime.now().timestamp()}"),
                "user",
            )

        article_data["user_id"] = user_id
        article_data["source"] = "User Upload"
        article_data["uploaded_at"] = datetime.now().isoformat()

        if "has_content" not in article_data:
            article_data["has_content"] = len(article_data.get("text", "")) > 100

        if not article_data["has_content"]:
            return ""

        # Сохраняем
        user_dir = self.uploads_dir / f"user_{user_id}"
        user_dir.mkdir(exist_ok=True)

        user_articles_file = user_dir / "articles.jsonl"
        with open(user_articles_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(article_data, ensure_ascii=False) + "\n")

        self.user_articles.append(article_data)
        self.all_articles.append(article_data)

        print(
            f"Добавлен документ пользователя {user_id}: {article_data.get('title', 'Без названия')}"
        )
        return article_data["id"]

    def get_user_articles(self, user_id: str) -> List[Dict[str, Any]]:
        """Возвращает статьи пользователя"""
        user_articles = []
        user_dir = self.uploads_dir / f"user_{user_id}"

        if user_dir.exists():
            user_articles_file = user_dir / "articles.jsonl"
            if user_articles_file.exists():
                user_articles = self._load_jsonl_file(user_articles_file)

        return user_articles

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        scope: SearchScope = SearchScope.ALL,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Ищет статьи по запросу с указанием области поиска"""

        # Определяем, в каких статьях искать
        if scope == SearchScope.USER_ONLY:
            if not user_id:
                return []
            # Только документы пользователя
            search_pool = [a for a in self.all_articles if a.get("user_id") == user_id]
        elif scope == SearchScope.HABR_ONLY:
            # Только статьи Habr
            search_pool = [a for a in self.all_articles if a.get("source") == "Habr"]
        else:  # SearchScope.ALL
            # Все статьи, но приоритет документам пользователя
            search_pool = self.all_articles

        results = []
        query_lower = query.lower()

        for article in search_pool:
            score = 0
            is_user_doc = article.get("user_id") == user_id if user_id else False

            # Поиск в заголовке
            title = article.get("title", "").lower()
            if query_lower in title:
                score += 5.0
            elif any(word in title for word in query_lower.split() if len(word) > 3):
                score += 2.0

            # Поиск в тегах
            tags = " ".join(article.get("tags", [])).lower()
            if any(tag in query_lower for tag in article.get("tags", [])):
                score += 4.0

            # Поиск в тексте
            text = article.get("text", "").lower()
            if query_lower in text:
                score += 2.0
            elif any(word in text for word in query_lower.split() if len(word) > 3):
                score += 0.5

            # Бонус за документы пользователя (если ищем везде)
            if scope == SearchScope.ALL and is_user_doc:
                score += 3.0

            if score > 0:
                results.append(
                    {
                        "article": article,
                        "score": score,
                        "is_user_document": is_user_doc,
                    }
                )

        # Сортируем по релевантности
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def generate_answer(
        self, query: str, user_id: str = None, scope: SearchScope = SearchScope.ALL
    ) -> Dict[str, Any]:
        """Генерирует ответ на основе найденных статей"""
        search_results = self.search(query, user_id, scope, limit=5)

        if not search_results:
            return self._get_empty_response(scope)

        # Формируем ответ
        context = self._build_context(search_results)
        answer = self._generate_response(query, context, search_results, scope)

        sources = self._format_sources(search_results)
        questions = self._generate_questions(query, search_results)

        return {
            "answer": answer,
            "sources": sources,
            "questions": questions,
            "found_in_user_docs": any(r["is_user_document"] for r in search_results),
            "total_found": len(search_results),
            "scope": scope.value,
        }

    def _get_empty_response(self, scope: SearchScope) -> Dict[str, Any]:
        """Возвращает ответ при отсутствии результатов"""
        scope_text = {
            SearchScope.ALL: "во всей базе",
            SearchScope.USER_ONLY: "в ваших документах",
            SearchScope.HABR_ONLY: "в статьях Habr",
        }.get(scope, "в выбранной области")

        return {
            "answer": f"🤔 К сожалению, я не нашел информации по вашему запросу {scope_text}.\n\n",
            "sources": [],
            "questions": [
                "Попробуйте изменить область поиска",
                "Попробуйте переформулировать запрос",
            ],
            "found_in_user_docs": False,
            "total_found": 0,
            "scope": scope.value,
        }

    def _build_context(self, search_results: List[Dict]) -> str:
        """Строит контекст из найденных статей"""
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):
            article = result["article"]
            source_type = "📁 Ваш документ" if result["is_user_document"] else "🌐 Habr"

            context_parts.append(
                f"【{source_type}】{article.get('title', 'Без названия')}\n"
                f"{article.get('text', '')[:500]}..."
            )
        return "\n\n".join(context_parts)

    def _generate_response(
        self, query: str, context: str, search_results: List[Dict], scope: SearchScope
    ) -> str:
        """Генерирует текстовый ответ"""
        scope_text = {
            SearchScope.ALL: "во всех источниках",
            SearchScope.USER_ONLY: "в ваших документах",
            SearchScope.HABR_ONLY: "в статьях Habr",
        }.get(scope, "")

        user_docs_found = any(r["is_user_document"] for r in search_results)

        response = f"🔍 **Поиск {scope_text}**\n\n"
        response += f"По запросу **'{query}'** найдено **{len(search_results)}** источников.\n\n"

        if scope == SearchScope.ALL and user_docs_found:
            response += "✅ *Найдены совпадения в ваших документах*\n\n"

        response += "**Ключевая информация:**\n"

        # Краткое содержание первых 3 результатов
        for i, result in enumerate(search_results[:3], 1):
            article = result["article"]
            title = article.get("title", "Без названия")
            response += f"{i}. {title}\n"

        response += "\n💡 Для деталей смотрите источники ниже."

        return response

    def _format_sources(self, search_results: List[Dict]) -> List[Dict[str, Any]]:
        """Форматирует список источников"""
        sources = []
        for result in search_results[:5]:
            article = result["article"]
            sources.append(
                {
                    "title": article.get("title", "Без названия"),
                    "url": article.get("url", "#"),
                    "source": article.get("source", "Unknown"),
                    "is_user_document": result["is_user_document"],
                    "relevance": f"{result['score']:.1f}",
                }
            )
        return sources

    def _generate_questions(self, query: str, search_results: List[Dict]) -> List[str]:
        """Генерирует вопросы для уточнения"""
        questions = []

        # Базовые вопросы
        base_questions = [
            "Какая информация была наиболее полезна?",
            "Нужны ли дополнительные детали по теме?",
            "Искать в других источниках?",
        ]

        # Контекстные вопросы
        if search_results:
            first_article = search_results[0]["article"]
            tags = first_article.get("tags", [])

            if tags:
                questions.append(f"Интересны ли темы: {', '.join(tags[:3])}?")

        return base_questions[:2] + questions[:2]

    def get_statistics(self, user_id: str = None) -> Dict[str, Any]:
        """Возвращает статистику системы"""
        user_articles_count = 0
        if user_id:
            user_articles_count = len(self.get_user_articles(user_id))

        return {
            "total_articles": len(self.all_articles),
            "habr_articles": len(self.habr_articles),
            "user_articles": len(self.user_articles),
            "current_user_articles": user_articles_count,
            "last_update": datetime.now().isoformat(),
        }


@pytest.fixture
def test_rag_agent(tmp_path):
    """Создает RAGAgent с временной директорией для тестов."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()

    # Создаем тестовый JSONL файл с Habr статьями
    habr_file = data_dir / "habr_articles.jsonl"
    test_articles = [
        {
            "id": "habr_12345",
            "title": "Тестовая статья о Python",
            "text": "Python - это язык программирования. Он популярен в машинном обучении.",
            "author": "Тестовый автор",
            "date": "2024-01-01",
            "tags": ["python", "programming"],
            "url": "https://habr.com/test",
            "source": "Habr",
            "has_content": True,
        },
        {
            "id": "habr_67890",
            "title": "RAG архитектура",
            "text": "RAG - это Retrieval-Augmented Generation. Используется в языковых моделях.",
            "author": "AI Researcher",
            "date": "2024-02-01",
            "tags": ["ai", "rag", "llm"],
            "url": "https://habr.com/rag",
            "source": "Habr",
            "has_content": True,
        },
    ]

    with open(habr_file, "w", encoding="utf-8") as f:
        for article in test_articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    return RAGAgent(data_dir=str(data_dir))


def test_rag_agent_initialization(test_rag_agent):
    """Тест на успешную инициализацию RAG агента."""
    assert test_rag_agent is not None
    assert hasattr(test_rag_agent, "all_articles")
    assert isinstance(test_rag_agent.all_articles, list)
    assert len(test_rag_agent.all_articles) > 0


def test_search_scope_enum():
    """Тест на корректность работы перечисления SearchScope."""
    assert SearchScope.ALL.value == "all"
    assert SearchScope.USER_ONLY.value == "user"
    assert SearchScope.HABR_ONLY.value == "habr"


def test_search_all_scope(test_rag_agent):
    """Тест поиска по всем источникам."""
    results = test_rag_agent.search("Python", scope=SearchScope.ALL)
    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0]["article"]["title"] == "Тестовая статья о Python"


def test_search_habr_scope(test_rag_agent):
    """Тест поиска только в статьях Habr."""
    results = test_rag_agent.search("RAG", scope=SearchScope.HABR_ONLY)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "RAG" in results[0]["article"]["title"]


def test_search_user_scope_no_docs(test_rag_agent):
    """Тест поиска в документах пользователя (когда их нет)."""
    results = test_rag_agent.search(
        "Python", user_id="test_user", scope=SearchScope.USER_ONLY
    )
    assert isinstance(results, list)
    assert len(results) == 0  # У пользователя нет документов


def test_add_user_article(test_rag_agent):
    """Тест добавления пользовательской статьи."""
    article_data = {
        "title": "Мой документ",
        "text": "Это тестовый документ пользователя о машинном обучении.",
        "author": "Пользователь",
        "tags": ["ml", "test"],
    }

    user_id = "test_user_123"
    article_id = test_rag_agent.add_user_article(article_data, user_id)

    assert article_id is not None
    assert len(article_id) > 0

    # Проверяем, что статья добавлена
    user_articles = test_rag_agent.get_user_articles(user_id)
    assert len(user_articles) == 1
    assert user_articles[0]["title"] == "Мой документ"


def test_generate_answer(test_rag_agent):
    """Тест генерации ответа."""
    result = test_rag_agent.generate_answer("Что такое Python?")

    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert "questions" in result
    assert len(result["answer"]) > 0


def test_generate_answer_empty(test_rag_agent):
    """Тест генерации ответа при отсутствии результатов."""
    result = test_rag_agent.generate_answer("абсолютно несуществующий запрос xyz123")

    assert isinstance(result, dict)
    assert "answer" in result
    assert "🤔" in result["answer"] or "не нашел" in result["answer"]


def test_get_statistics(test_rag_agent):
    """Тест получения статистики."""
    stats = test_rag_agent.get_statistics()

    assert isinstance(stats, dict)
    assert "total_articles" in stats
    assert "habr_articles" in stats
    assert "user_articles" in stats
    assert stats["total_articles"] > 0


@pytest.mark.parametrize(
    "query,expected_titles",
    [
        ("python", ["Тестовая статья о Python"]),
        ("rag", ["RAG архитектура"]),
        ("машинное обучение", ["Тестовая статья о Python"]),
    ],
)
def test_search_queries(test_rag_agent, query, expected_titles):
    """Параметризованный тест различных запросов."""
    results = test_rag_agent.search(query, scope=SearchScope.ALL)

    if expected_titles:
        assert len(results) > 0
        found_titles = [r["article"]["title"] for r in results]
        for expected_title in expected_titles:
            assert any(expected_title in title for title in found_titles)
