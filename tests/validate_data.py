#!/usr/bin/env python3
"""
Автоматический тест качества спарсенных статей
Проверяет целостность, полноту и корректность данных
"""

import json
import os
import sys
import glob
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class ValidationResult:
    """Результат валидации"""
    passed: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict


class DataValidator:
    """Валидатор данных спарсенных статей"""
    
    # Критические пороги (можно менять)
    MIN_TEXT_LENGTH = 100
    MIN_TITLE_LENGTH = 10
    MAX_TITLE_LENGTH = 300
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or self.find_latest_data()
        self.data = []
        
    def find_latest_data(self) -> str:
        """Находит последний JSONL файл с данными"""
        # Сначала проверяем основную папку data
        data_dirs = [
            'data',
            'parser_articles/data/mass_parse_results',
            'parser_articles/data'
        ]
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                jsonl_files = glob.glob(os.path.join(data_dir, '*.jsonl'))
                if jsonl_files:
                    # Берем самый новый файл по времени модификации
                    latest = max(jsonl_files, key=os.path.getmtime)
                    print(f"Найден файл данных: {latest}")
                    return latest
        
        raise FileNotFoundError("Не найден ни один файл данных (.jsonl)")
    
    def load_data(self) -> List[Dict]:
        """Загружает данные из JSONL файла"""
        data = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            article = json.loads(line)
                            article['_line_num'] = line_num
                            data.append(article)
                        except json.JSONDecodeError as e:
                            print(f"Ошибка JSON в строке {line_num}: {e}")
                            continue
            
            print(f"Загружено статей: {len(data)}")
            return data
            
        except Exception as e:
            print(f"Ошибка загрузки файла {self.data_path}: {e}")
            raise
    
    def validate(self) -> ValidationResult:
        """Запускает все проверки"""
        print(f"\n" + "="*60)
        print(f"ВАЛИДАЦИЯ ДАННЫХ: {os.path.basename(self.data_path)}")
        print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"="*60)
        
        # Загружаем данные
        self.data = self.load_data()
        
        if not self.data:
            return ValidationResult(
                passed=False,
                errors=["Нет данных для валидации"],
                warnings=[],
                stats={}
            )
        
        errors = []
        warnings = []
        stats = {}
        
        # Запускаем все проверки
        self._check_basic_structure(errors)
        self._check_missing_fields(errors, warnings)
        self._check_field_content(errors, warnings)
        self._check_duplicates(errors, warnings)
        self._check_data_consistency(errors, warnings)
        
        # Собираем статистику
        stats = self._collect_statistics()
        
        # Выводим результаты
        self._print_report(errors, warnings, stats)
        
        passed = len(errors) == 0
        
        return ValidationResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    def _check_basic_structure(self, errors: List[str]):
        """Проверяет базовую структуру данных"""
        required_fields = ['id', 'title', 'text', 'url', 'source']
        
        for field in required_fields:
            if field not in self.data[0]:
                errors.append(f"Отсутствует обязательное поле: {field}")
    
    def _check_missing_fields(self, errors: List[str], warnings: List[str]):
        """Проверяет пропущенные значения"""
        critical_fields = ['title', 'text', 'url']
        important_fields = ['author', 'date', 'tags']
        
        for field in critical_fields:
            missing = sum(1 for article in self.data if not article.get(field))
            if missing > 0:
                errors.append(f"Пропущено {missing} значений в критическом поле: {field}")
        
        for field in important_fields:
            missing = sum(1 for article in self.data if not article.get(field))
            if missing > len(self.data) * 0.1:  # Более 10% пропусков
                warnings.append(f"Много пропусков в поле {field}: {missing} ({missing/len(self.data)*100:.1f}%)")
    
    def _check_field_content(self, errors: List[str], warnings: List[str]):
        """Проверяет содержание полей"""
        short_texts = 0
        short_titles = 0
        long_titles = 0
        empty_tags = 0
        
        for i, article in enumerate(self.data):
            # Проверка текста
            text = article.get('text', '')
            if len(str(text)) < self.MIN_TEXT_LENGTH:
                short_texts += 1
                if short_texts <= 5:  # Показываем только первые 5
                    warnings.append(f"Строка {article.get('_line_num', i+1)}: Текст слишком короткий ({len(str(text))} символов)")
            
            # Проверка заголовка
            title = article.get('title', '')
            title_len = len(str(title))
            if title_len < self.MIN_TITLE_LENGTH:
                short_titles += 1
            if title_len > self.MAX_TITLE_LENGTH:
                long_titles += 1
            
            # Проверка тегов
            tags = article.get('tags', [])
            if not tags or (isinstance(tags, list) and len(tags) == 0):
                empty_tags += 1
        
        # Агрегированные предупреждения
        if short_texts > 0:
            errors.append(f"Найдено {short_texts} статей с текстом короче {self.MIN_TEXT_LENGTH} символов")
        
        if short_titles > 0:
            warnings.append(f"Найдено {short_titles} статей с короткими заголовками (<{self.MIN_TITLE_LENGTH} символов)")
        
        if long_titles > 0:
            warnings.append(f"Найдено {long_titles} статей с длинными заголовками (>{self.MAX_TITLE_LENGTH} символов)")
        
        if empty_tags > len(self.data) * 0.3:  # Более 30% без тегов
            warnings.append(f"Много статей без тегов: {empty_tags} ({empty_tags/len(self.data)*100:.1f}%)")
    
    def _check_duplicates(self, errors: List[str], warnings: List[str]):
        """Проверяет дубликаты"""
        # Проверка по URL
        urls = [article.get('url') for article in self.data]
        url_counts = Counter(urls)
        duplicates = {url: count for url, count in url_counts.items() if count > 1}
        
        if duplicates:
            errors.append(f"Найдено {len(duplicates)} дубликатов по URL")
            for url, count in list(duplicates.items())[:5]:
                warnings.append(f"URL дублируется {count} раз: {url[:80]}...")
        
        # Проверка по заголовкам (менее строгая)
        titles = [article.get('title', '').strip().lower() for article in self.data]
        title_counts = Counter(titles)
        title_duplicates = {title: count for title, count in title_counts.items() 
                           if count > 1 and title}
        
        if title_duplicates:
            warnings.append(f"Найдено {len(title_duplicates)} возможных дубликатов по заголовкам")
    
    def _check_data_consistency(self, errors: List[str], warnings: List[str]):
        """Проверяет согласованность данных"""
        # Проверка формата даты
        date_errors = 0
        for article in self.data:
            date_str = article.get('date', '')
            if date_str and 'T' not in date_str and '-' not in date_str:
                date_errors += 1
        
        if date_errors > 0:
            warnings.append(f"Найдено {date_errors} статей с нестандартным форматом даты")
        
        # Проверка источника
        sources = Counter(article.get('source', 'unknown') for article in self.data)
        if len(sources) > 1:
            warnings.append(f"Найдено несколько источников: {dict(sources)}")
        
        # Проверка структуры тегов
        tag_format_errors = 0
        for article in self.data:
            tags = article.get('tags', [])
            if tags and not isinstance(tags, list):
                tag_format_errors += 1
        
        if tag_format_errors > 0:
            errors.append(f"Найдено {tag_format_errors} статей с некорректным форматом тегов (не список)")
    
    def _collect_statistics(self) -> Dict:
        """Собирает статистику по данным"""
        if not self.data:
            return {}
        
        stats = {
            'total_articles': len(self.data),
            'validation_time': datetime.now().isoformat(),
            'data_file': os.path.basename(self.data_path)
        }
        
        # Статистика по полям
        fields_stats = {}
        for field in ['title', 'text', 'author', 'date', 'tags', 'url', 'source']:
            non_empty = sum(1 for article in self.data if article.get(field))
            fields_stats[field] = {
                'non_empty': non_empty,
                'empty': len(self.data) - non_empty,
                'percentage': (non_empty / len(self.data)) * 100 if self.data else 0
            }
        
        stats['fields'] = fields_stats
        
        # Дополнительная статистика
        text_lengths = [len(str(a.get('text', ''))) for a in self.data]
        stats['text_length'] = {
            'min': min(text_lengths) if text_lengths else 0,
            'max': max(text_lengths) if text_lengths else 0,
            'avg': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            'short_count': sum(1 for l in text_lengths if l < self.MIN_TEXT_LENGTH)
        }
        
        # Теги
        all_tags = []
        for article in self.data:
            tags = article.get('tags', [])
            if isinstance(tags, list):
                all_tags.extend(tags)
        
        stats['tags'] = {
            'total_unique': len(set(all_tags)),
            'total_occurrences': len(all_tags),
            'articles_with_tags': sum(1 for a in self.data if a.get('tags')),
            'articles_without_tags': sum(1 for a in self.data if not a.get('tags'))
        }
        
        # Источники
        sources = Counter(a.get('source', 'unknown') for a in self.data)
        stats['sources'] = dict(sources)
        
        return stats
    
    def _print_report(self, errors: List[str], warnings: List[str], stats: Dict):
        """Выводит отчет о валидации"""
        print(f"\n" + "="*60)
        print("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ")
        print(f"="*60)
        
        # Ошибки
        if errors:
            print(f"\nКРИТИЧЕСКИЕ ОШИБКИ ({len(errors)}):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
        else:
            print("\nКритических ошибок нет")
        
        # Предупреждения
        if warnings:
            print(f"\nПРЕДУПРЕЖДЕНИЯ ({len(warnings)}):")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        else:
            print("\nПредупреждений нет")
        
        # Статистика
        print(f"\n" + "="*60)
        print("СТАТИСТИКА")
        print(f"="*60)
        
        if stats:
            print(f"\nОбщее количество статей: {stats['total_articles']}")
            print(f"Файл данных: {stats['data_file']}")
            
            print("\nЗаполненность полей:")
            for field, field_stats in stats.get('fields', {}).items():
                print(f"  {field}: {field_stats['non_empty']}/{stats['total_articles']} "
                      f"({field_stats['percentage']:.1f}%)")
            
            if 'text_length' in stats:
                tl = stats['text_length']
                print(f"\nДлина текста:")
                print(f"  Минимальная: {tl['min']} символов")
                print(f"  Максимальная: {tl['max']} символов")
                print(f"  Средняя: {tl['avg']:.0f} символов")
                print(f"  Коротких текстов (<{self.MIN_TEXT_LENGTH} символов): {tl['short_count']}")
            
            if 'tags' in stats:
                t = stats['tags']
                print(f"\nТеги:")
                print(f"  Уникальных тегов: {t['total_unique']}")
                print(f"  Всего использований: {t['total_occurrences']}")
                print(f"  Статей с тегами: {t['articles_with_tags']}")
                print(f"  Статей без тегов: {t['articles_without_tags']}")
        
        # Итог
        print(f"\n" + "="*60)
        if errors:
            print("ВАЛИДАЦИЯ НЕ ПРОЙДЕНА")
        elif warnings:
            print("ВАЛИДАЦИЯ ПРОЙДЕНА С ПРЕДУПРЕЖДЕНИЯМИ")
        else:
            print("ВАЛИДАЦИЯ ПРОЙДЕНА УСПЕШНО")
        print(f"="*60)


def main():
    """Точка входа"""
    parser = argparse.ArgumentParser(description='Валидация данных спарсенных статей')
    parser.add_argument('--file', '-f', help='Путь к JSONL файлу с данными')
    parser.add_argument('--strict', '-s', action='store_true', 
                       help='Строгий режим (предупреждения считаются ошибками)')
    
    args = parser.parse_args()
    
    try:
        validator = DataValidator(args.file)
        result = validator.validate()
        
        # Возвращаем код выхода
        if result.passed and (not args.strict or not result.warnings):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("\nВозможные пути к данным:")
        print("  - data/ (основная папка)")
        print("  - parser_articles/data/mass_parse_results/")
        print("  - parser_articles/data/")
        sys.exit(2)
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()