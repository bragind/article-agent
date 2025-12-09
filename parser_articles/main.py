#!/usr/bin/env python3
"""
Скрипт для массового парсинга статей с сохранением прогресса
"""

import json
import time
import random
from datetime import datetime
import sys
import os

# Добавляем корень проекта в путь Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Абсолютные импорты
from parser_articles.parsers.config import (
    MAX_ARTICLES, SECTIONS, BASE_DELAY, ERROR_DELAY
)
from parser_articles.parsers.habr_parser import HabrParser


def load_progress(filename='parser_articles/data/progress.json'):
    """Загружает прогресс из файла"""
    import os
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'parsed_articles': [], 'failed_urls': [], 'last_run': None}


def save_progress(progress, filename='parser_articles/data/progress.json'):
    """Сохраняет прогресс в файл"""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def mass_parse(max_articles=MAX_ARTICLES):
    """Основная функция массового парсинга"""
    import os
    
    results_dir = 'parser_articles/data/mass_parse_results'
    os.makedirs(results_dir, exist_ok=True)
    
    parser = HabrParser(delay=BASE_DELAY, timeout=15)
    
    progress = load_progress()
    already_parsed = set(progress['parsed_articles'])
    
    print("=" * 60)
    print(f"НАЧИНАЕМ СБОР ССЫЛОК ДЛЯ {max_articles} СТАТЕЙ")
    print("=" * 60)
    
    all_links = []
    for section in SECTIONS:
        if len(all_links) >= max_articles:
            break
            
        print(f"Собираем ссылки из раздела: {section}")
        links = parser.get_article_links_with_pagination(
            base_url=section,
            max_articles=max_articles - len(all_links)
        )
        
        new_links = [link for link in links if link not in already_parsed]
        all_links.extend(new_links)
        print(f"Добавлено {len(new_links)} новых ссылок из этого раздела")
    
    all_links = all_links[:max_articles]
    print(f"Собрано {len(all_links)} уникальных ссылок для парсинга")
    
    print("\n" + "=" * 60)
    print("НАЧИНАЕМ ПАРСИНГ СТАТЕЙ")
    print("=" * 60)
    
    articles = []
    failed_urls = []
    
    for i, url in enumerate(all_links, 1):
        try:
            print(f"[{i}/{len(all_links)}] Парсим: {url}")
            
            article = parser.parse_article(url)
            
            if article.get('has_content'):
                articles.append(article)
                progress['parsed_articles'].append(url)
                print(f"Успешно: {article['title'][:70]}...")
                print(f"Автор: {article.get('author', 'N/A')}, Теги: {len(article.get('tags', []))}, Символов: {article.get('text_length', 0)}")
            else:
                print(f"Пропущено: недостаточно контента")
                failed_urls.append(url)
            
            if i % 10 == 0:
                progress['failed_urls'] = failed_urls
                progress['last_run'] = datetime.now().isoformat()
                save_progress(progress)
                print(f"Автосохранение прогресса после {i} статей")
            
            delay = BASE_DELAY + 2 + random.uniform(0, 3)
            time.sleep(delay)
            
        except Exception as e:
            print(f"Ошибка: {str(e)[:100]}")
            failed_urls.append(url)
            time.sleep(ERROR_DELAY)
    
    print("\n" + "=" * 60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_file = f'{results_dir}/articles_{timestamp}.jsonl'
    parser.save_to_jsonl(articles, output_file)
    
    stats = {
        'total_attempted': len(all_links),
        'successfully_parsed': len(articles),
        'failed': len(failed_urls),
        'date': datetime.now().isoformat(),
        'output_file': output_file
    }
    
    stats_file = f'{results_dir}/stats_{timestamp}.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    if failed_urls:
        failed_file = f'{results_dir}/failed_{timestamp}.json'
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_urls, f, ensure_ascii=False, indent=2)
    
    progress['last_run'] = datetime.now().isoformat()
    progress['failed_urls'] = failed_urls
    save_progress(progress)
    
    print(f"ИТОГОВАЯ СТАТИСТИКА:")
    print(f"Всего обработано: {len(all_links)}")
    print(f"Успешно спарсено: {len(articles)} ({len(articles)/len(all_links)*100:.1f}%)")
    print(f"Не удалось: {len(failed_urls)}")
    print(f"Результаты сохранены в: {output_file}")
    print(f"Статистика: {stats_file}")
    if failed_urls:
        print(f"Список ошибок: {failed_file}")
    
    return articles, output_file


if __name__ == "__main__":
    print("Запуск массового парсера Habr")
    print(f"Цель: собрать {MAX_ARTICLES} статей")
    print("Для остановки нажмите Ctrl+C")
    
    try:
        articles, output_file = mass_parse(max_articles=MAX_ARTICLES)
        
        if articles:
            parser = HabrParser()
            
            main_data_dir = 'data'
            import os
            os.makedirs(main_data_dir, exist_ok=True)
            
            main_output = f'{main_data_dir}/articles_batch.jsonl'
            parser.save_to_jsonl(articles, main_output)
            print(f"Основной файл для RAG обновлен: {main_output}")
            print(f"Готово! Вы можете использовать {main_output} для RAG пайплайна.")
            
    except KeyboardInterrupt:
        print("Парсинг прерван пользователем. Прогресс сохранен.")
    except Exception as e:
        print(f"Критическая ошибка: {e}")