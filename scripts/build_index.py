#!/usr/bin/env python3
"""
Построение векторного индекса для статей
"""

import jsonlines
import sys
import os
import argparse
from typing import List, Dict, Any

# Добавляем корень проекта в путь Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest.chunker import TextChunker
from src.ingest.embedder import Embedder
from src.ingest.vector_store import VectorStore


def load_articles(data_path: str) -> List[Dict[str, Any]]:
    """Загрузка статей из JSONL файла"""
    articles = []
    try:
        with jsonlines.open(data_path) as reader:
            for article in reader:
                articles.append(article)
        print(f"Загружено {len(articles)} статей из {data_path}")
    except Exception as e:
        print(f"Ошибка загрузки статей: {e}")
        articles = []
    return articles


def prepare_chunks(articles: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    """
    Подготовка чанков из статей
    
    Args:
        articles: Список статей
        chunk_size: Размер чанка в символах
        chunk_overlap: Перекрытие между чанками
        
    Returns:
        Список чанков с текстом и метаданными
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    
    for article in articles:
        article_text = article.get("text", "")
        if not article_text or len(article_text) < 100:
            continue
            
        # Разбиваем текст на чанки
        text_chunks = chunker.split_text(article_text)
        
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text) < 50:  # Пропускаем слишком короткие чанки
                continue
                
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "article_id": article.get("id", ""),
                    "title": article.get("title", "Без названия"),
                    "author": article.get("author", "Неизвестен"),
                    "url": article.get("url", ""),
                    "source": article.get("source", "Unknown"),
                    "date": article.get("date", ""),
                    "chunk_index": i,
                    "total_chunks": len(text_chunks)
                }
            })
    
    return chunks


def build_index(data_path: str, output_dir: str, chunk_size: int = 500, chunk_overlap: int = 50, batch_size: int = 4000):
    """
    Построение векторного индекса
    
    Args:
        data_path: Путь к файлу с данными
        output_dir: Директория для сохранения векторной БД
        chunk_size: Размер чанка
        chunk_overlap: Перекрытие чанков
        batch_size: Размер батча для ChromaDB
    """
    print(f"Начинаем построение индекса...")
    print(f"Данные: {data_path}")
    print(f"Выходная директория: {output_dir}")
    print(f"Размер чанка: {chunk_size}, перекрытие: {chunk_overlap}")
    print(f"Размер батча: {batch_size}")
    
    # 1. Загрузка статей
    articles = load_articles(data_path)
    if not articles:
        print("Не удалось загрузить статьи. Выход.")
        return
    
    # 2. Подготовка чанков
    print("\nРазбиение текста на чанки...")
    chunks = prepare_chunks(articles, chunk_size, chunk_overlap)
    print(f"Создано {len(chunks)} чанков из {len(articles)} статей")
    
    if not chunks:
        print("Нет чанков для индексации. Выход.")
        return
    
    # 3. Генерация эмбеддингов
    print("\nГенерация эмбеддингов...")
    embedder = Embedder()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed(texts)
    print(f"Сгенерировано {len(embeddings)} эмбеддингов")
    
    # 4. Сохранение в векторную БД
    print("\nСохранение в векторную базу данных...")
    vector_store = VectorStore(path=output_dir, batch_size=batch_size)
    vector_store.add_chunks(chunks, embeddings)
    
    # 5. Проверка результата
    count = vector_store.count()
    print(f"\nИндекс успешно построен!")
    print(f"Сохранено {count} чанков в {output_dir}")
    print(f"Статистика:")
    print(f"  - Статей: {len(articles)}")
    print(f"  - Чанков: {len(chunks)}")
    print(f"  - Среднее чанков на статью: {len(chunks)/len(articles):.1f}")


def test_search(query: str, index_dir: str = "chroma_db", top_k: int = 3):
    """
    Тестирование поиска по построенному индексу
    
    Args:
        query: Текст запроса
        index_dir: Директория с индексом
        top_k: Количество возвращаемых результатов
    """
    try:
        from src.ingest.embedder import Embedder
        from src.ingest.vector_store import VectorStore
        
        print(f"\nТестируем поиск: '{query}'")
        
        # Инициализация
        embedder = Embedder()
        vector_store = VectorStore(path=index_dir)
        
        # Генерация эмбеддинга запроса
        query_embedding = embedder.embed([query])[0]
        
        # Поиск
        results = vector_store.search(query_embedding, top_k=top_k)
        
        if results and results["documents"]:
            print(f"Найдено {len(results['documents'][0])} результатов:")
            for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                print(f"\n{i+1}. {metadata.get('title', 'Без названия')}")
                print(f"   Автор: {metadata.get('author', 'Неизвестен')}")
                print(f"   Источник: {metadata.get('source', 'Unknown')}")
                print(f"   Текст: {doc[:150]}...")
                if metadata.get('url'):
                    print(f"   URL: {metadata['url']}")
        else:
            print("Ничего не найдено")
            
    except Exception as e:
        print(f"Ошибка при тестировании поиска: {e}")


def main():
    parser = argparse.ArgumentParser(description="Построение векторного индекса для статей")
    parser.add_argument("--data", default="data/articles_sample.jsonl", 
                       help="Путь к файлу с данными (по умолчанию: data/articles_sample.jsonl)")
    parser.add_argument("--output", default="chroma_db", 
                       help="Директория для сохранения векторной БД (по умолчанию: chroma_db)")
    parser.add_argument("--chunk-size", type=int, default=800,
                    help="Размер чанка в символах (по умолчанию: 800)")
    parser.add_argument("--chunk-overlap", type=int, default=150,
                    help="Перекрытие чанков в символах (по умолчанию: 150)")
    parser.add_argument("--batch-size", type=int, default=4000,
                       help="Размер батча для ChromaDB (по умолчанию: 4000)")
    parser.add_argument("--test", action="store_true",
                       help="Протестировать поиск после построения индекса")
    parser.add_argument("--test-query", default="машинное обучение",
                       help="Запрос для тестирования поиска")
    
    args = parser.parse_args()
    
    # Проверка существования файла с данными
    if not os.path.exists(args.data):
        print(f"Файл {args.data} не найден!")
        print("Создайте тестовые данные командой:")
        print("python scripts/create_sample.py --size 50")
        sys.exit(1)
    
    # Построение индекса
    build_index(
        data_path=args.data,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size
    )
    
    # Тестирование поиска если нужно
    if args.test:
        test_search(args.test_query, args.output)


if __name__ == "__main__":
    main()