"""
Тестирование RAG системы с полной базой данных chroma_full_db
"""

import sys
import os

# Добавляем корень проекта в путь Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.rag import RAGSystem
from src.ingest.vector_store import VectorStore


def test_full_database():
    """Тестирование работы с полной базой данных"""
    print("Тестирование RAG системы с полной базой данных...")
    print("=" * 60)
    
    # 1. Проверяем размеры баз данных
    print("1. Проверка размеров баз данных:")
    
    try:
        # Тестовая база
        test_store = VectorStore(path="chroma_db")
        test_count = test_store.count()
        print(f"   Статей в тестовой базе (chroma_db): {test_count}")
    except Exception as e:
        print(f"   Ошибка при проверке chroma_db: {e}")
        test_count = 0
    
    try:
        # Полная база
        full_store = VectorStore(path="chroma_full_db")
        full_count = full_store.count()
        print(f"   Статей в полной базе (chroma_full_db): {full_count}")
        
        if test_count > 0 and full_count > 0:
            ratio = full_count / test_count
            print(f"   Полная база в {ratio:.1f} раз больше тестовой")
    except Exception as e:
        print(f"   Ошибка при проверке chroma_full_db: {e}")
        print("   Убедитесь, что база данных существует и доступна")
        full_count = 0
    
    print("\n" + "=" * 60)
    
    # 2. Инициализируем RAG систему с полной базой
    print("2. Инициализация RAG системы с полной базой...")
    
    try:
        rag = RAGSystem(
            vector_db_path="chroma_full_db",
            llm_model="llama3.2:3b",
            llm_base_url="http://localhost:11434"
        )
        print("   RAG система успешно инициализирована")
    except Exception as e:
        print(f"   Ошибка инициализации: {e}")
        return False
    
    # 3. Проверяем подключение к LLM
    print("\n3. Проверка подключения к LLM...")
    llm_status = rag.check_llm_connection()
    
    if llm_status["connected"]:
        print(f"   LLM подключена: Да")
        print(f"   Модель: {llm_status['current_model']}")
        
        if llm_status["available_models"]:
            print(f"   Доступные модели: {', '.join(llm_status['available_models'][:3])}")
    else:
        print("   LLM не подключена")
        print("   Убедитесь, что Ollama запущен: ollama serve")
    
    print("\n" + "=" * 60)
    
    # 4. Тестируем поиск
    print("4. Тестирование поиска в полной базе...")
    
    test_queries = [
        "машинное обучение",
        "искусственный интеллект",
        "базы данных",
        "Python программирование"
    ]
    
    for query in test_queries[:2]:  # Тестируем первые 2 запроса
        print(f"\n   Запрос: '{query}'")
        
        try:
            # Простой поиск
            search_results = rag.search(query, top_k=2)
            
            if search_results and search_results.get("documents"):
                docs = search_results["documents"][0]
                print(f"   Найдено документов: {len(docs)}")
                
                # Показываем первый результат
                if docs:
                    # Показываем метаданные
                    if search_results.get("metadatas"):
                        meta = search_results["metadatas"][0][0]
                        print(f"   Первый результат:")
                        print(f"     Заголовок: {meta.get('title', 'N/A')}")
                        print(f"     Автор: {meta.get('author', 'N/A')}")
                        print(f"     Источник: {meta.get('source', 'N/A')}")
            else:
                print("   По запросу ничего не найдено")
                
        except Exception as e:
            print(f"   Ошибка при поиске: {e}")
    
    print("\n" + "=" * 60)
    
    # 5. Тестируем полный RAG пайплайн (только если LLM подключена)
    print("5. Тестирование полного RAG пайплайна...")
    
    if llm_status["connected"] and full_count > 0:
        test_query = "Что такое машинное обучение?"
        print(f"\n   Запрос для генерации: '{test_query}'")
        
        try:
            result = rag.generate_answer(test_query, top_k=3)
            
            print(f"   Ответ успешно сгенерирован!")
            print(f"\n   Ответ (первые 300 символов):")
            print(f"   {result.get('answer', 'Нет ответа')[:300]}...")
            
            sources = result.get("sources", [])
            if sources:
                print(f"\n   Найдено источников: {len(sources)}")
                for i, src in enumerate(sources[:2], 1):
                    print(f"   {i}. {src.get('title', 'Без названия')}")
            
            questions = result.get("questions", [])
            if questions:
                print(f"\n   Вопросы для самопроверки:")
                for q in questions:
                    print(f"   • {q}")
                    
            return True
            
        except Exception as e:
            print(f"   Ошибка при генерации ответа: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("   Пропускаем тест генерации:")
        if not llm_status["connected"]:
            print("   - LLM не подключена")
        if full_count == 0:
            print("   - База данных пуста")
        return False


def compare_search_results():
    """Сравнение результатов поиска в тестовой и полной базах"""
    print("\n" + "=" * 60)
    print("Сравнение результатов поиска в разных базах...")
    print("=" * 60)
    
    queries = ["машинное обучение", "искусственный интеллект"]
    
    for query in queries:
        print(f"\nЗапрос: '{query}'")
        print("-" * 40)
        
        # Поиск в тестовой базе
        try:
            rag_test = RAGSystem(vector_db_path="chroma_db", llm_model="llama3.2:3b")
            test_results = rag_test.search(query, top_k=2)
            test_count = len(test_results["documents"][0]) if test_results.get("documents") else 0
            print(f"В тестовой базе (chroma_db): {test_count} результатов")
        except Exception as e:
            print(f"Ошибка в тестовой базе: {e}")
            test_count = 0
        
        # Поиск в полной базе
        try:
            rag_full = RAGSystem(vector_db_path="chroma_full_db", llm_model="llama3.2:3b")
            full_results = rag_full.search(query, top_k=2)
            full_count = len(full_results["documents"][0]) if full_results.get("documents") else 0
            print(f"В полной базе (chroma_full_db): {full_count} результатов")
        except Exception as e:
            print(f"Ошибка в полной базе: {e}")
            full_count = 0
        
        if test_count > 0 and full_count > 0:
            print(f"Разница: +{full_count - test_count} результатов в полной базе")


if __name__ == "__main__":
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ RAG СИСТЕМЫ С ПОЛНОЙ БАЗОЙ ДАННЫХ")
    print("=" * 60)
    
    # Запускаем основной тест
    success = test_full_database()
    
    # Если основной тест прошел, запускаем сравнение
    if success:
        compare_search_results()
    
    print("\n" + "=" * 60)
    
    if success:
        print("РЕЗУЛЬТАТ: Все тесты пройдены успешно!")
        print("\nРекомендации:")
        print("1. Обновите файлы для использования полной базы:")
        print("   - В src/bot.py замените 'chroma_db' на 'chroma_full_db'")
        print("   - В src/app.py замените 'chroma_db' на 'chroma_full_db'")
        print("\n2. Запустите систему:")
        print("   streamlit run src/app.py")
        print("   или")
        print("   python src/bot.py")
    else:
        print("РЕЗУЛЬТАТ: Есть проблемы с системой.")
        print("\nПроверьте:")
        print("1. Запущен ли Ollama: ollama serve")
        print("2. Существует ли папка chroma_full_db/")
        print("3. Содержит ли база данных статьи (запустите scripts/build_index.py)")
    
    print("=" * 60)