# test_generation.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.rag import RAGSystem
import time

def test_generation_pipeline():
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ПОЛНОГО RAG ПАЙПЛАЙНА")
    print("=" * 70)
    
    rag = RAGSystem(vector_db_path="chroma_full_db", llm_model="llama3.2:3b")
    
    test_queries = [
        {
            "query": "Объясни, что такое машинное обучение простыми словами",
            "min_answer_length": 100,
            "expected_in_answer": ["обучение", "данные", "алгоритм", "модель"]
        },
        {
            "query": "Какие есть типы баз данных?",
            "min_answer_length": 80,
            "expected_in_answer": ["SQL", "NoSQL", "реляционные", "документные"]
        },
        {
            "query": "Что такое Python и для чего он используется?",
            "min_answer_length": 80,
            "expected_in_answer": ["язык программирования", "Python", "программирование"]
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n[ТЕСТ {i}] Запрос: '{test['query']}'")
        
        try:
            start_time = time.time()
            result = rag.generate_answer(test["query"], top_k=3)
            total_time = time.time() - start_time
            
            print(f"  Время выполнения: {total_time:.2f} секунд")
            
            # Проверяем ответ
            answer = result.get("answer", "")
            print(f"  Длина ответа: {len(answer)} символов")
            
            if len(answer) < test["min_answer_length"]:
                print(f"  Ответ слишком короткий (минимум {test['min_answer_length']} символов)")
                print(f"  Ответ: {answer[:200]}...")
            else:
                print(f"  Ответ достаточной длины")
                
                # Проверяем наличие ожидаемых слов
                found_keywords = []
                for keyword in test["expected_in_answer"]:
                    if keyword.lower() in answer.lower():
                        found_keywords.append(keyword)
                
                if found_keywords:
                    print(f"  Найдены ключевые слова: {found_keywords}")
                else:
                    print(f"  Ключевые слова не найдены")
                
                print(f"  Ответ (первые 300 символов):")
                print(f"  {answer[:300]}...")
            
            # Проверяем источники
            sources = result.get("sources", [])
            print(f"  Количество источников: {len(sources)}")
            if sources:
                print(f"  Пример источника: {sources[0].get('title', 'Без названия')}")
            
            # Проверяем вопросы
            questions = result.get("questions", [])
            print(f"  Сгенерировано вопросов: {len(questions)}")
            if questions:
                print(f"  Пример вопроса: {questions[0]}")
            
            print(f"  {'-'*60}")
            
        except Exception as e:
            print(f"  Ошибка: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Тестирование пайплайна завершено")

if __name__ == "__main__":
    test_generation_pipeline()