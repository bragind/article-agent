#!/usr/bin/env python3
"""
Построение векторного индекса для статей
"""

import jsonlines
import json
import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Добавляем корень проекта в путь Python
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.ingest.chunker import TextChunker
from src.ingest.embedder import Embedder
from src.ingest.vector_store import VectorStore


def load_articles(data_path: str) -> List[Dict[str, Any]]:
    """Загрузка статей из JSONL файла И пользовательских статей"""
    articles = []
    
    # 1. Загружаем статьи Habr
    try:
        with jsonlines.open(data_path) as reader:
            for article in reader:
                article["is_user_document"] = False
                article["source_type"] = "habr"
                article["uploaded_by"] = "system"
                articles.append(article)
        print(f"Загружено {len(articles)} статей Habr из {data_path}")
    except Exception as e:
        print(f"Ошибка загрузки статей Habr: {e}")
    
    # 2. Загружаем пользовательские статьи
    user_dir = "data/user_uploads"
    if os.path.exists(user_dir):
        user_articles = []
        for filename in os.listdir(user_dir):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(user_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        article = json.load(f)
                        # Добавляем обязательные поля
                        article["is_user_document"] = True
                        article["source_type"] = "user"
                        if "uploaded_by" not in article:
                            # Пытаемся извлечь из имени файла
                            article["uploaded_by"] = filename.replace("test_article_", "").replace(".json", "")
                        user_articles.append(article)
                        print(f"  Загружена пользовательская статья: {article.get('title', 'Без названия')}")
                except Exception as e:
                    print(f"Ошибка загрузки пользовательской статьи {filename}: {e}")
        
        print(f"Загружено {len(user_articles)} пользовательских статей")
        articles.extend(user_articles)
    else:
        print("Директория пользовательских статей не найдена")
    
    print(f"Всего загружено статей: {len(articles)}")
    return articles


def prepare_chunks(articles: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    """Разбивка статей на чанки"""
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []

    for article in articles:
        text = article.get("text", "")
        if not text or len(text) < 100:
            continue

        text_chunks = chunker.split_text(text)
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text) < 50:
                continue
            
            # Критические поля для фильтрации
            is_user_document = article.get("is_user_document", False)
            source_type = article.get("source_type", "habr")
            uploaded_by = article.get("uploaded_by", "system")
            
            def format_tags(tags):
                """Преобразует теги в строку формата: 'tag1, tag2, tag3'"""
                if isinstance(tags, list):
                    unique_tags = []
                    seen = set()
                    for tag in tags:
                        if tag and str(tag).strip():
                            tag_str = str(tag).strip()
                            if tag_str not in seen and len(tag_str) < 50:
                                seen.add(tag_str)
                                unique_tags.append(tag_str)
                    return ", ".join(unique_tags[:10])
                elif isinstance(tags, str):
                    return tags.replace(";", ",").replace("|", ",")[:200]
                else:
                    return str(tags)[:200] if tags else ""

            tags_str = format_tags(article.get("tags", []))
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "article_id": article.get("id", ""),
                    "title": article.get("title", "Без названия")[:200],
                    "author": article.get("author", "Неизвестен")[:100],
                    "url": article.get("url", "")[:200],
                    "source": article.get("source", "Unknown")[:100],
                    "date": str(article.get("date", ""))[:50],
                    "tags": tags_str,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "is_user_document": is_user_document,
                    "source_type": source_type,
                    "uploaded_by": uploaded_by[:50]
                }
            })

    return chunks


def build_index(data_path: str, output_dir: str, chunk_size: int = 500, chunk_overlap: int = 50, batch_size: int = 1000):
    """Построение векторного индекса"""
    output_path = str(Path(output_dir).resolve())
    print(f"Индекс будет сохранён в: {output_path}")

    articles = load_articles(data_path)
    if not articles:
        print("Нет статей для индексации")
        return

    chunks = prepare_chunks(articles, chunk_size, chunk_overlap)
    if not chunks:
        print("Нет чанков для индексации")
        return

    print(f"Создано {len(chunks)} чанков из {len(articles)} статей")
    
    # Подсчет по типам
    user_chunks = sum(1 for chunk in chunks if chunk["metadata"].get("is_user_document"))
    habr_chunks = len(chunks) - user_chunks
    print(f"   - Habr чанков: {habr_chunks}")
    print(f"   - Пользовательских чанков: {user_chunks}")

    # Генерация эмбеддингов
    embedder = Embedder()
    texts = [chunk["text"] for chunk in chunks]
    print(f"Генерация эмбеддингов для {len(texts)} текстов...")
    embeddings = embedder.embed(texts)
    
    if embeddings is None:
        print("Ошибка: не удалось сгенерировать эмбеддинги")
        return
    
    if hasattr(embeddings, 'shape'):
        if embeddings.shape[0] != len(chunks):
            print(f"Ошибка: количество эмбеддингов ({embeddings.shape[0]}) "
                  f"не совпадает с количеством чанков ({len(chunks)})")
            return
        print(f"Сгенерировано {embeddings.shape[0]} эмбеддингов "
              f"(размерность: {embeddings.shape[1]})")
    else:
        print(f"Сгенерировано {len(embeddings)} эмбеддингов")

    # Создание VectorStore и очистка старой коллекции
    vector_store = VectorStore(path=output_path, batch_size=batch_size)

    print("\nОчистка старого индекса (если есть)...")
    try:
        # Получаем все существующие записи
        results = vector_store.collection.get()
        if results and results.get("ids"):
            print(f"Найдено {len(results['ids'])} существующих записей")
            vector_store.collection.delete(ids=results["ids"])
            print(f"Удалено {len(results['ids'])} старых записей")
        else:
            print("Коллекция пуста")
    except Exception as e:
        print(f"Не удалось очистить коллекцию: {e}")
        # Альтернатива: удалить и пересоздать коллекцию
        try:
            print("Пытаюсь пересоздать коллекцию...")
            vector_store.client.delete_collection(name="articles")
            print("Коллекция удалена")
            # Пересоздаем
            vector_store.collection = vector_store.client.get_or_create_collection(
                name="articles",
                metadata={"description": "Чанки статей для RAG поиска"}
            )
            print("Коллекция создана заново")
        except Exception as e2:
            print(f"Не удалось пересоздать коллекцию: {e2}")

    # Добавляем чанки в базу
    print(f"\nДобавление чанков в векторную базу...")
    try:
        vector_store.add_chunks(chunks, embeddings)
    except Exception as e:
        print(f"Ошибка при добавлении чанков: {e}")
        import traceback
        traceback.print_exc()
        return

    # Проверка результата
    try:
        count = vector_store.count()
        print(f"\nИндекс успешно построен!")
        print(f"Статистика:")
        print(f"   - Статей: {len(articles)}")
        print(f"   - Чанков: {len(chunks)}")
        print(f"   - В базе: {count} записей")
        print(f"   - Habr чанков: {habr_chunks}")
        print(f"   - Пользовательских чанков: {user_chunks}")
        print(f"   - Среднее чанков на статью: {len(chunks)/len(articles):.1f}")
        
        # Проверка на дубликаты
        if count > len(chunks) * 1.1:  # На 10% больше
            print(f"Внимание: в базе {count} записей, но чанков только {len(chunks)}")
            print(f"Возможны дубликаты")
            
    except Exception as e:
        print(f"Ошибка получения статистики: {e}")


def test_search(query: str, index_dir: str = "chroma_db", top_k: int = 3):
    """Тестирование поиска с подробной статистикой"""
    print(f"\nТестируем поиск: '{query}'")
    print(f"   Коллекция: {index_dir}")
    print(f"   Количество результатов: {top_k}")
    
    embedder = Embedder()
    vector_store = VectorStore(path=str(Path(index_dir).resolve()))
    
    # Проверяем размер коллекции
    try:
        count = vector_store.count()
        print(f"Размер коллекции: {count} записей")
    except Exception as e:
        print(f"Не удалось получить размер коллекции: {e}")
    
    # Генерация эмбеддинга запроса
    query_embedding = embedder.embed([query])[0]
    
    # Поиск
    results = vector_store.search(query_embedding, top_k=top_k)
    
    if results and results.get("documents"):
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        
        print(f"Найдено {len(docs)} результатов:")
        
        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            source_type = "Ваш документ" if meta.get("is_user_document") else "🌐 Habr"
            print(f"\n{i}. {source_type} {meta.get('title', 'Без названия')}")
            print(f"   Автор: {meta.get('author', 'Неизвестен')}")
            print(f"   Дата: {meta.get('date', '')[:10]}")
            print(f"   Теги: {meta.get('tags', [])[:3]}")
            print(f"   Фрагмент: {doc[:150]}...")
            
            # Для пользовательских документов показываем uploaded_by
            if meta.get("is_user_document"):
                print(f"   uploaded_by: {meta.get('uploaded_by', 'неизвестно')}")
    else:
        print("Ничего не найдено")


def main():
    parser = argparse.ArgumentParser(description="Построение векторного индекса для статей")
    parser.add_argument("--data", default="data/articles_batch.jsonl")
    parser.add_argument("--output", default="chroma_db")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-query", default="машинное обучение")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Файл {args.data} не найден!")
        return

    build_index(
        data_path=args.data,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size
    )

    if args.test:
        test_search(args.test_query, args.output)


if __name__ == "__main__":
    main()