#!/usr/bin/env python3
"""
Основной скрипт для парсинга статей с Habr
"""

import os
import sys
from datetime import datetime

# Добавляем путь к модулю парсера
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser_articles.parsers.habr_parser import HabrParser


def main():
    """Основная функция парсинга"""
    
    # Создаем парсер с задержкой 2 секунды
    parser = HabrParser(delay=2.0)
    
    # Страницы для парсинга
    pages_to_parse = [
        'https://habr.com/ru/all/',
        'https://habr.com/ru/flows/develop/',
        'https://habr.com/ru/flows/admin/',
    ]
    
    all_articles = []
    
    # Парсим каждую страницу
    for page_url in pages_to_parse:
        articles = parser.get_articles_from_page(page_url, limit=3)
        all_articles.extend(articles)
    
    # Фильтруем только статьи с контентом
    valid_articles = [a for a in all_articles if a.get('has_content')]
    
    if not valid_articles:
        print("\n❌ Не удалось спарсить ни одной статьи!")
        return
    
    # Создаем папки для данных
    os.makedirs('parser_articles/data', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Генерируем имя файла с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Сохраняем в разные форматы
    jsonl_filename = f'data/articles_{timestamp}.jsonl'
    parser.save_to_jsonl(valid_articles, jsonl_filename)
    
    # Также сохраняем как последний результат
    latest_filename = 'data/articles_latest.jsonl'
    parser.save_to_jsonl(valid_articles, latest_filename)
    
    # Выводим статистику
    print_statistics(valid_articles)


def print_statistics(articles):
    """Выводит статистику по спарсенным статьям"""
    
    print("\n" + "="*60)
    print("СТАТИСТИКА ПАРСИНГА")
    print("="*60)
    
    total_articles = len(articles)
    print(f"Всего статей: {total_articles}")
    
    if not articles:
        return
    
    # Статистика по авторам
    authors = [a.get('author', '').strip() for a in articles if a.get('author', '').strip()]
    unique_authors = len(set(authors))
    print(f"Уникальных авторов: {unique_authors}")
    
    # Статистика по тегам
    all_tags = []
    for article in articles:
        all_tags.extend(article.get('tags', []))
    unique_tags = len(set(all_tags))
    print(f"Уникальных тегов: {unique_tags}")
    
    # Статистика по длине текста
    text_lengths = [a.get('text_length', 0) for a in articles]
    avg_length = sum(text_lengths) / len(text_lengths)
    max_length = max(text_lengths)
    min_length = min(text_lengths)
    
    print(f"\nДлина текста:")
    print(f"  Средняя: {avg_length:.0f} символов")
    print(f"  Максимальная: {max_length} символов")
    print(f"  Минимальная: {min_length} символов")
    
    # Примеры статей
    print(f"\nПРИМЕРЫ СТАТЕЙ:")
    for i, article in enumerate(articles[:2], 1):
        print(f"\n{i}. {article.get('title', 'Без названия')[:80]}...")
        print(f"   Автор: {article.get('author', 'Не указан')}")
        print(f"   Дата: {article.get('date', '')[:10]}")
        print(f"   Теги: {', '.join(article.get('tags', [])[:3])}")
        print(f"   Символов: {article.get('text_length', 0)}")
    
    print("\n✅ Парсинг завершен успешно!")


if __name__ == "__main__":
    main()