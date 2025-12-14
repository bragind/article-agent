#!/usr/bin/env python3
"""
Создание тестового набора данных
"""

import jsonlines
import random
import argparse


def create_sample(input_path: str, output_path: str, sample_size: int):
    """
    Создает выборку из основного датасета
    
    Args:
        input_path: Путь к исходному файлу
        output_path: Путь для сохранения выборки
        sample_size: Размер выборки
    """
    try:
        # Читаем все статьи
        with jsonlines.open(input_path) as reader:
            articles = list(reader)
        
        print(f"Входной файл: {input_path}")
        print(f"Всего статей: {len(articles)}")
        
        # Выбираем случайные статьи
        if len(articles) <= sample_size:
            sampled_articles = articles
            print(f"Размер выборки больше чем доступно статей, берем все")
        else:
            sampled_articles = random.sample(articles, sample_size)
        
        # Сохраняем выборку
        with jsonlines.open(output_path, 'w') as writer:
            writer.write_all(sampled_articles)
        
        print(f"Создан тестовый набор: {output_path}")
        print(f"Размер выборки: {len(sampled_articles)} статей")
        
    except FileNotFoundError:
        print(f"Файл {input_path} не найден")
        print("Убедитесь, что вы выполнили: git lfs pull")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Создание тестового набора данных")
    parser.add_argument("--input", default="data/articles_batch.jsonl",
                       help="Входной файл с данными")
    parser.add_argument("--output", default="data/articles_sample.jsonl",
                       help="Выходной файл для выборки")
    parser.add_argument("--size", type=int, default=50,
                       help="Количество статей в выборке")
    
    args = parser.parse_args()
    create_sample(args.input, args.output, args.size)