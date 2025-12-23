"""
Клиент для работы с локальными LLM через Ollama API
"""

import requests
import json
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """Универсальный клиент для работы с локальными LLM через Ollama"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model: str = "llama3.2:3b",
                 timeout: int = 120):
        """
        Инициализация клиента для локальной LLM
        
        Args:
            base_url: URL сервера Ollama (по умолчанию localhost:11434)
            model: Название модели в Ollama
            timeout: Таймаут запроса в секундах
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        
    def __del__(self):
        """Деструктор для закрытия сессии"""
        if hasattr(self, 'session'):
            self.session.close()
    
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: int = 1000) -> str:
        """
        Генерация ответа на промпт
        
        Args:
            prompt: Пользовательский промпт
            system_prompt: Системный промпт (роль модели)
            temperature: Креативность (0.0-1.0)
            max_tokens: Максимальное количество токенов в ответе
            
        Returns:
            Сгенерированный текст
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "message" in result and "content" in result["message"]:
                return result["message"]["content"]
            else:
                logger.error(f"Неожиданный ответ от LLM: {result}")
                return "Ошибка: Неожиданный формат ответа от LLM."
            
        except requests.exceptions.ConnectionError:
            logger.error("Не удается подключиться к серверу Ollama.")
            return "Ошибка: Не удается подключиться к локальному серверу LLM. Убедитесь, что Ollama запущен командой 'ollama serve'."
        except requests.exceptions.Timeout:
            logger.error(f"Таймаут при запросе к LLM (timeout={self.timeout}s)")
            return "Ошибка: Превышено время ожидания ответа от LLM."
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON от LLM: {e}")
            return "Ошибка: Некорректный ответ от сервера LLM."
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            return f"Ошибка при генерации ответа: {str(e)[:200]}"
    
    def generate_article_summary(self, 
                                article_text: str, 
                                article_title: str,
                                query: Optional[str] = None) -> str:
        """
        Генерация краткого резюме статьи
        
        Args:
            article_text: Текст статьи
            article_title: Заголовок статьи
            query: Опциональный запрос для контекста
            
        Returns:
            Краткое резюме статьи
        """
        system_prompt = """Ты AI-ассистент, который анализирует технические статьи и создает краткие, информативные резюме.
        Твои резюме должны быть:
        1. Краткими (3-5 предложений)
        2. Информативными
        3. Сфокусированными на ключевых идеях
        4. Написанными простым и понятным языком
        5. Объективными и точными"""
        
        prompt = f"""Проанализируй статью и создай краткое резюме.
        
        Заголовок статьи: {article_title}
        
        Текст статьи (первые 2000 символов): {article_text[:2000]}
        """
        
        if query:
            prompt += f"\nКонтекст запроса пользователя: {query}"
            
        prompt += "\n\nСоздай краткое резюме этой статьи (3-5 предложений):"
        
        return self.generate_response(prompt, system_prompt, temperature=0.5)
    
    def generate_answer_with_context(self, 
                                   query: str, 
                                   context: List[Dict[str, Any]],
                                   top_k: int = 3) -> Dict[str, Any]:
        """
        Генерация ответа на вопрос с учетом контекста из RAG
        
        Args:
            query: Вопрос пользователя
            context: Список найденных релевантных чанков с метаданными
            top_k: Сколько топ-результатов использовать
            
        Returns:
            Словарь с ответом и метаданными
        """
        if not context:
            return {
                "answer": "Не найдено релевантного контекста для ответа на ваш вопрос.",
                "sources": [],
                "context_used": 0
            }
        
        context_text = ""
        sources = []
        
        for i, chunk in enumerate(context[:top_k]):
            chunk_text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            
            if chunk_text:
                context_text += f"[Источник {i+1}]: {chunk_text}\n\n"
                
                sources.append({
                    "title": metadata.get("title", "Без названия"),
                    "author": metadata.get("author", "Неизвестен"),
                    "url": metadata.get("url", ""),
                    "source": metadata.get("source", "Unknown"),
                    "relevance_rank": i + 1
                })
        
        if not context_text.strip():
            return {
                "answer": "Контекст не содержит текста для анализа.",
                "sources": sources,
                "context_used": 0
            }
        
        system_prompt = """Ты AI-агент для интеллектуального поиска технических статей.
        Твоя задача — отвечать на вопросы пользователей на основе предоставленного контекста из найденных статей.
        
        Твои ответы должны быть:
        1. Краткими и информативными
        2. Основанными ТОЛЬКО на предоставленном контексте
        3. С указанием источников в формате [1], [2] и т.д.
        4. Структурированными и понятными
        5. Объективными и точными
        
        Если в контексте нет информации для ответа, честно скажи об этом."""
        
        prompt = f"""Вопрос пользователя: {query}
        
        Контекст из найденных статей:
        {context_text}
        
        Ответь на вопрос пользователя на основе предоставленного контекста.
        В ответе указывай источники информации в квадратных скобках, например [1], [2].
        Если используешь информацию из нескольких источников, указывай все соответствующие номера.
        
        Если в контексте нет информации для ответа на вопрос, напиши: "В предоставленном контексте нет информации для ответа на этот вопрос."
        
        Твой ответ:"""
        
        answer = self.generate_response(prompt, system_prompt, temperature=0.3)
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(context[:top_k])
        }
    
    def generate_questions(self, 
                          article_text: str, 
                          num_questions: int = 3) -> List[str]:
        """
        Генерация вопросов для самопроверки по статье
        
        Args:
            article_text: Текст статьи
            num_questions: Количество вопросов для генерации (1-10)
            
        Returns:
            Список вопросов
        """
        # Ограничиваем количество вопросов
        num_questions = max(1, min(num_questions, 10))
        
        # Если текст слишком короткий, возвращаем общие вопросы
        if not article_text or len(article_text.strip()) < 100:
            return self._get_default_questions(num_questions)
        
        system_prompt = """Ты AI-ассистент, который создает проверочные вопросы по техническим статьям.
        Твои вопросы должны:
        1. Проверять понимание ключевых концепций статьи
        2. Быть конкретными и однозначными
        3. Иметь четкие ответы в тексте статьи
        4. Быть разнообразными (фактологические, концептуальные, прикладные)
        5. Подходить для самопроверки знаний
        
        **ВАЖНО:** 
        - Верни ТОЛЬКО вопросы, по одному на строке
        - Без нумерации, без маркеров списка
        - Без пояснений, комментариев, вступлений
        - Каждый вопрос должен заканчиваться знаком вопроса
        
        Пример правильного ответа:
        Какие основные технологии обсуждаются в статье?
        Как решается проблема производительности?
        Какие практические рекомендации дает автор?"""
        
        prompt = f"""На основе следующего текста статьи сгенерируй {num_questions} вопроса для самопроверки понимания:

Текст статьи:
{article_text[:1500]}

Сгенерируй {num_questions} разнообразных вопроса, которые помогут проверить понимание материала.
Каждый вопрос должен быть на новой строке.
Не нумеруй вопросы, не добавляй маркеры списка, только сами вопросы."""

        try:
            response = self.generate_response(prompt, system_prompt, temperature=0.5, max_tokens=500)
            
            if not response or "Ошибка:" in response:
                logger.warning(f"LLM вернул ошибку: {response}")
                return self._get_default_questions(num_questions)
            
            # Очистка и парсинг ответа
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Убираем нумерацию (1., 2., - , * , • и т.д.)
                line = re.sub(r'^[\d\-*•]+\s*', '', line)
                # Убираем кавычки и лишние символы
                line = re.sub(r'^["\'«»]|["\'«»]$', '', line)
                # Убираем префиксы типа "Вопрос:", "Q:", "В:"
                line = re.sub(r'^(Вопрос:|Q:|В:)\s*', '', line, flags=re.IGNORECASE)
                
                # Проверяем, что это вопрос
                if (line and 
                    len(line) > 10 and 
                    '?' in line and
                    not line.startswith(('Ответ:', 'A:', 'Примечание:', 'Замечание:'))):
                    questions.append(line)
            
            # Если получили вопросы, возвращаем их (даже если меньше запрошенного)
            if questions:
                logger.info(f"Сгенерировано {len(questions)} вопросов из запрошенных {num_questions}")
                return questions[:num_questions]
            
            # Если не получили нормальных вопросов, возвращаем дефолтные
            logger.warning("LLM не сгенерировал вопросы, возвращаю дефолтные")
            return self._get_default_questions(num_questions)
            
        except Exception as e:
            logger.error(f"Ошибка генерации вопросов: {e}")
            return self._get_default_questions(num_questions)
    
    def _get_default_questions(self, num_questions: int) -> List[str]:
        """Возвращает дефолтные вопросы для самопроверки"""
        default_questions = [
            "Какие ключевые идеи представлены в материале?",
            "Какую проблему или задачу решает автор?",
            "Какие методы, технологии или подходы обсуждаются?",
            "Какие выводы или результаты представлены?",
            "Как можно применить эти знания на практике?",
            "Какие преимущества и недостатки у обсуждаемых решений?",
            "С какими другими технологиями или подходами это связано?",
            "Какие рекомендации дает автор для реализации?",
            "Какие метрики или критерии успеха упоминаются?",
            "Какие дальнейшие исследования или развитие возможны?"
        ]
        return default_questions[:num_questions]
    
    def generate_tags_for_document(self, text: str, title: str = "", max_tags: int = 5) -> List[str]:
        """
        Генерация тегов для документа с помощью LLM
        
        Args:
            text: Текст документа (первые 2000 символов достаточно)
            title: Заголовок документа
            max_tags: Максимальное количество тегов
            
        Returns:
            Список тегов
        """
        max_tags = max(1, min(max_tags, 10))  # Ограничиваем от 1 до 10
        
        system_prompt = """Ты AI-ассистент для категоризации технических документов.
        Твоя задача - проанализировать документ и выделить ключевые теги для поиска и категоризации.

        **СТРОГИЕ ПРАВИЛА:**
        1. Верни ТОЛЬКО теги через запятую, без пояснений
        2. Не пиши вступлений, заключений, комментариев
        3. Не используй фразы типа "ключевые теги:", "теги документа:"
        4. Только список тегов через запятую

        **Примеры правильного ответа:**
        - машинное обучение, python, нейронные сети
        - docker, контейнеризация, devops
        - базы данных, sql, оптимизация запросов

        **Примеры НЕПРАВИЛЬНОГО ответа:**
        - "После анализа я выделил теги: машинное обучение" ❌
        - "Ключевые теги документа: python, алгоритмы" ❌
        - "Вот теги: docker, kubernetes" ❌"""

        # Берем часть текста для анализа
        preview = text[:1500]
        
        prompt = f"""Проанализируй документ и выдели {max_tags} ключевых тегов.

Заголовок документа: {title}

Содержимое документа (начало):
{preview}

**Инструкции:**
1. Проанализируй тему и содержание документа
2. Выдели ключевые технические концепции
3. Выбери наиболее релевантные термины
4. Верни только теги через запятую

Теги (через запятую):"""

        try:
            response = self.generate_response(prompt, system_prompt, temperature=0.3, max_tokens=200)
            
            if not response or "Ошибка:" in response:
                logger.warning(f"LLM вернул ошибку при генерации тегов: {response}")
                return self._get_fallback_tags(title, max_tags)
            
            # Обработка ответа
            tags = []
            
            # Разделяем по запятым, точкам с запятой или переносам строк
            raw_tags = re.split(r'[,;]|\n', response)
            
            for tag in raw_tags:
                tag_clean = tag.strip()
                
                # Убираем номера, маркеры списка, кавычки
                tag_clean = re.sub(r'^\d+[\.\)]\s*', '', tag_clean)
                tag_clean = re.sub(r'^[\-\*•]\s*', '', tag_clean)
                tag_clean = re.sub(r'^["\'«»]|["\'«»]$', '', tag_clean)
                
                # Проверяем, что это валидный тег
                if (tag_clean and 
                    2 <= len(tag_clean) <= 50 and 
                    not tag_clean.lower().startswith(('тег', 'пример', 'ответ', 'вопрос')) and
                    ':' not in tag_clean):
                    tags.append(tag_clean.lower())
            
            # Убираем дубликаты
            unique_tags = []
            seen = set()
            for tag in tags:
                if tag not in seen and len(unique_tags) < max_tags:
                    seen.add(tag)
                    unique_tags.append(tag)
            
            # Если LLM не вернул теги или их мало, добавляем fallback
            if len(unique_tags) < 2:
                logger.warning(f"LLM вернул мало тегов ({len(unique_tags)}), добавляю fallback")
                fallback_tags = self._get_fallback_tags(title, max_tags)
                # Объединяем, избегая дубликатов
                for tag in fallback_tags:
                    if tag not in seen and len(unique_tags) < max_tags:
                        unique_tags.append(tag)
            
            logger.info(f"Сгенерировано {len(unique_tags)} тегов для документа")
            return unique_tags[:max_tags]
            
        except Exception as e:
            logger.error(f"Ошибка генерации тегов через LLM: {e}")
            return self._get_fallback_tags(title, max_tags)
    
    def _get_fallback_tags(self, title: str, max_tags: int) -> List[str]:
        """Fallback теги на основе заголовка"""
        tags = []
        
        # Добавляем теги на основе заголовка
        if title:
            # Извлекаем ключевые слова из заголовка
            words = re.findall(r'\b\w+\b', title.lower())
            for word in words:
                if (len(word) > 3 and 
                    word not in ['статья', 'документ', 'материал', 'технический'] and
                    word not in tags):
                    tags.append(word)
                    if len(tags) >= max_tags:
                        break
        
        # Добавляем общие теги если нужно
        general_tags = ["технический документ", "материал", "статья", "информация", "контент"]
        for tag in general_tags:
            if len(tags) < max_tags and tag not in tags:
                tags.append(tag)
        
        return tags[:max_tags]
    
    def test_connection(self) -> bool:
        """
        Проверка подключения к серверу Ollama и доступности модели
        
        Returns:
            True если подключение успешно, иначе False
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                
                # Проверяем наличие нашей модели
                if self.model in models:
                    logger.info(f"Модель {self.model} доступна")
                    return True
                
                # Если нашей модели нет, но есть другие
                if models:
                    logger.warning(f"Модель {self.model} не найдена. Доступные модели: {models}")
                    
                    # Пробуем найти похожую модель
                    similar_models = [
                        m for m in models 
                        if "llama" in m.lower() or "mistral" in m.lower() or "qwen" in m.lower()
                    ]
                    
                    if similar_models:
                        self.model = similar_models[0]
                        logger.info(f"Автоматически выбрана модель: {self.model}")
                        return True
                    else:
                        # Выбираем первую доступную модель
                        self.model = models[0]
                        logger.info(f"Выбрана модель: {self.model}")
                        return True
                
                logger.error("На сервере Ollama нет доступных моделей")
                return False
            
            logger.error(f"Ошибка HTTP {response.status_code} при проверке подключения")
            return False
            
        except requests.exceptions.ConnectionError:
            logger.error("Не удается подключиться к серверу Ollama. Убедитесь, что Ollama запущен.")
            return False
        except requests.exceptions.Timeout:
            logger.error("Таймаут при подключении к серверу Ollama")
            return False
        except Exception as e:
            logger.error(f"Ошибка проверки подключения: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Получение списка доступных моделей в Ollama
        
        Returns:
            Список названий моделей
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except:
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о текущей модели
        
        Returns:
            Словарь с информацией о модели
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                json={"model": self.model},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Ошибка получения информации о модели: {e}")
            return {}