"""
Клиент для работы с локальными LLM через Ollama API
"""

import requests
import json
from typing import List, Dict, Any, Optional
import logging

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
            
            return result["message"]["content"]
            
        except requests.exceptions.ConnectionError:
            logger.error("Не удается подключиться к серверу Ollama. Убедитесь, что Ollama запущен.")
            return "Ошибка: Не удается подключиться к локальному серверу LLM. Убедитесь, что Ollama запущен командой 'ollama serve'."
        except requests.exceptions.Timeout:
            logger.error(f"Таймаут при запросе к LLM (timeout={self.timeout}s)")
            return "Ошибка: Превышено время ожидания ответа от LLM."
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            return f"Ошибка при генерации ответа: {str(e)}"
    
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
            
        prompt += "\n\nСоздай краткое резюме этой статьи:"
        
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
        context_text = ""
        sources = []
        
        for i, chunk in enumerate(context[:top_k]):
            chunk_text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            
            context_text += f"[Источник {i+1}]: {chunk_text}\n\n"
            
            sources.append({
                "title": metadata.get("title", "Без названия"),
                "author": metadata.get("author", "Неизвестен"),
                "url": metadata.get("url", ""),
                "source": metadata.get("source", "Unknown"),
                "relevance_rank": i + 1
            })
        
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
            num_questions: Количество вопросов для генерации
            
        Returns:
            Список вопросов
        """
        system_prompt = """Ты AI-ассистент, который создает проверочные вопросы по техническим статьям.
        Твои вопросы должны:
        1. Проверять понимание ключевых концепций статьи
        2. Быть конкретными и однозначными
        3. Иметь четкие ответы в тексте статьи
        4. Быть разнообразными (фактологические, концептуальные, прикладные)
        5. Подходить для самопроверки знаний"""
        
        prompt = f"""На основе следующего текста статьи сгенерируй {num_questions} вопроса для самопроверки понимания:
        
        Текст статьи (первые 1500 символов): {article_text[:1500]}
        
        Сгенерируй {num_questions} разнообразных вопроса, которые помогут проверить понимание материала.
        Каждый вопрос должен быть на новой строке.
        Не нумеруй вопросы, просто каждый с новой строки."""
        
        response = self.generate_response(prompt, system_prompt, temperature=0.5)
        
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        
        if not questions or len(questions) < num_questions:
            default_questions = [
                "Какая основная проблема или задача рассматривается в статье?",
                "Какие ключевые методы или подходы предлагаются в статье?",
                "Какие основные выводы или результаты представлены в статье?"
            ]
            return default_questions[:num_questions]
        
        return questions[:num_questions]
    
    def test_connection(self) -> bool:
        """
        Проверка подключения к серверу Ollama
        
        Returns:
            True если подключение успешно, иначе False
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except:
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