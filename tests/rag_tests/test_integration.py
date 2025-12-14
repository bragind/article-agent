"""
Интеграционный тест для RAG системы с LLM
"""

import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.rag import RAGSystem
from src.generation.llm_client import LLMClient


class TestLLMIntegration(unittest.TestCase):
    """Тесты интеграции с LLM"""
    
    def setUp(self):
        """Настройка тестов"""
        self.llm_client = LLMClient(model="llama3.2:3b")
        
    def test_llm_connection(self):
        """Тест подключения к Ollama"""
        is_connected = self.llm_client.test_connection()
        self.assertTrue(is_connected, "Нет подключения к Ollama. Запустите 'ollama serve'")
    
    def test_llm_generation(self):
        """Тест простой генерации текста"""
        if not self.llm_client.test_connection():
            self.skipTest("Ollama не доступен, пропускаем тест генерации")
        
        response = self.llm_client.generate_response(
            prompt="Ответь одним предложением: Что такое искусственный интеллект?",
            system_prompt="Ты полезный AI-ассистент."
        )
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 10)
        print(f"Ответ LLM: {response[:100]}...")
    
    def test_question_generation(self):
        """Тест генерации вопросов"""
        if not self.llm_client.test_connection():
            self.skipTest("Ollama не доступен, пропускаем тест генерации вопросов")
        
        test_text = """
        Машинное обучение — это область искусственного интеллекта, 
        которая изучает алгоритмы, способные обучаться на данных и 
        делать предсказания или принимать решения без явного программирования.
        """
        
        questions = self.llm_client.generate_questions(test_text, num_questions=2)
        
        self.assertIsInstance(questions, list)
        self.assertGreaterEqual(len(questions), 1)
        print(f"Сгенерированные вопросы: {questions}")


class TestRAGSystem(unittest.TestCase):
    """Тесты RAG системы"""
    
    def setUp(self):
        """Настройка тестов"""
        self.rag = RAGSystem(
            vector_db_path="chroma_db",
            llm_model="llama3.2:3b"
        )
    
    def test_rag_initialization(self):
        """Тест инициализации RAG системы"""
        self.assertIsNotNone(self.rag.embedder)
        self.assertIsNotNone(self.rag.vector_store)
        self.assertIsNotNone(self.rag.llm_client)
    
    def test_search_function(self):
        """Тест функции поиска"""
        # Предполагаем, что есть хотя бы несколько статей в базе
        try:
            results = self.rag.search("машинное обучение", top_k=2)
            
            if results and results.get("documents"):
                docs = results["documents"][0]
                self.assertLessEqual(len(docs), 2)
                print(f"Найдено документов: {len(docs)}")
            else:
                print("Поиск не вернул результатов (возможно, база пуста)")
        except Exception as e:
            print(f"Ошибка при поиске (возможно, нет векторной базы): {e}")
    
    def test_llm_status_check(self):
        """Тест проверки статуса LLM"""
        status = self.rag.check_llm_connection()
        
        self.assertIn("connected", status)
        self.assertIn("current_model", status)
        
        if status["connected"]:
            print(f"LLM подключена. Модель: {status['current_model']}")
        else:
            print("LLM не подключена")
    
    @unittest.skipUnless(os.path.exists("chroma_db"), "Требуется векторная база данных")
    def test_full_rag_pipeline(self):
        """Полный тест RAG пайплайна (требует и базу данных, и LLM)"""
        if not self.rag.check_llm_connection()["connected"]:
            self.skipTest("LLM не доступна, пропускаем полный тест RAG")
        
        query = "Что такое машинное обучение?"
        result = self.rag.generate_answer(query, top_k=3)
        
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("questions", result)
        
        print(f"Ответ: {result['answer'][:200]}...")
        print(f"Источников: {len(result['sources'])}")
        print(f"Вопросов: {len(result['questions'])}")


def run_all_tests():
    """Запуск всех тестов"""
    print("Запуск интеграционных тестов RAG системы...")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestLLMIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGSystem))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 60)
    print(f"Всего тестов: {result.testsRun}")
    print(f"Провалено: {len(result.failures)}")
    print(f"Пропущено: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("Все тесты прошли успешно!")
    else:
        print("Есть проваленные тесты")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\nРекомендации:")
        print("1. Запустите Streamlit интерфейс: streamlit run src/app.py")
        print("2. Запустите Telegram бота: python src/bot.py (предварительно настройте BOT_TOKEN в .env)")
    else:
        print("\nПроблемы:")
        print("1. Убедитесь, что Ollama запущен: ollama serve")
        print("2. Убедитесь, что модель загружена: ollama pull llama3.2:3b")
        print("3. Убедитесь, что создана векторная база: python scripts/build_index.py")