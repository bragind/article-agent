#!/usr/bin/env python3
"""
Тесты для проверки качества данных
"""

import unittest
import tempfile
import os
import json
from validate_data import DataValidator


class TestDataValidator(unittest.TestCase):
    """Тесты валидатора данных"""
    
    def setUp(self):
        """Создание тестовых данных"""
        self.test_data = [
            {
                "id": "test_1",
                "title": "Тестовая статья 1",
                "text": "Это полный текст статьи с достаточным количеством символов для прохождения валидации.",
                "author": "Тестовый автор",
                "date": "2024-01-15T10:30:00.000Z",
                "tags": ["тест", "валидация"],
                "url": "https://example.com/article1",
                "source": "Habr"
            },
            {
                "id": "test_2",
                "title": "Тестовая статья 2",
                "text": "Еще один полный текст статьи.",
                "author": "Другой автор",
                "date": "2024-01-14T09:15:00.000Z",
                "tags": ["python", "тестирование"],
                "url": "https://example.com/article2",
                "source": "Habr"
            }
        ]
    
    def create_test_file(self, data):
        """Создает временный файл с тестовыми данными"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
        for item in data:
            temp_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        temp_file.close()
        return temp_file.name
    
    def test_valid_data(self):
        """Тест с корректными данными"""
        test_file = self.create_test_file(self.test_data)
        
        try:
            validator = DataValidator(test_file)
            result = validator.validate()
            
            self.assertTrue(result.passed)
            self.assertEqual(len(result.errors), 0)
            self.assertEqual(len(result.warnings), 0)
        finally:
            os.unlink(test_file)
    
    def test_missing_text(self):
        """Тест с отсутствующим текстом"""
        invalid_data = self.test_data.copy()
        invalid_data[0]['text'] = ""
        
        test_file = self.create_test_file(invalid_data)
        
        try:
            validator = DataValidator(test_file)
            result = validator.validate()
            
            self.assertFalse(result.passed)
            self.assertGreater(len(result.errors), 0)
        finally:
            os.unlink(test_file)
    
    def test_short_text(self):
        """Тест с коротким текстом"""
        invalid_data = self.test_data.copy()
        invalid_data[0]['text'] = "Коротко"
        
        test_file = self.create_test_file(invalid_data)
        
        try:
            validator = DataValidator(test_file)
            result = validator.validate()
            
            self.assertFalse(result.passed)
            self.assertGreater(len(result.errors), 0)
        finally:
            os.unlink(test_file)
    
    def test_duplicate_urls(self):
        """Тест с дублирующимися URL"""
        invalid_data = self.test_data.copy()
        invalid_data[1]['url'] = invalid_data[0]['url']  # Делаем URL одинаковыми
        
        test_file = self.create_test_file(invalid_data)
        
        try:
            validator = DataValidator(test_file)
            result = validator.validate()
            
            self.assertFalse(result.passed)
            self.assertGreater(len(result.errors), 0)
        finally:
            os.unlink(test_file)
    
    def test_missing_required_field(self):
        """Тест с отсутствующим обязательным полем"""
        invalid_data = self.test_data.copy()
        del invalid_data[0]['title']  # Удаляем обязательное поле
        
        test_file = self.create_test_file(invalid_data)
        
        try:
            validator = DataValidator(test_file)
            result = validator.validate()
            
            self.assertFalse(result.passed)
            self.assertGreater(len(result.errors), 0)
        finally:
            os.unlink(test_file)


def run_tests():
    """Запуск тестов"""
    unittest.main()


if __name__ == "__main__":
    run_tests()