import requests
from bs4 import BeautifulSoup
import time
import random
import re
from datetime import datetime
from typing import List, Dict
import json


class HabrParser:
    """
    Современный парсер статей с Habr.com
    Использует точные селекторы из структуры HTML
    """
    
    BASE_URL = 'https://habr.com'
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    def __init__(self, delay: float = 1.5, timeout: int = 10):
        """
        :param delay: Базовая задержка между запросами (секунды)
        :param timeout: Таймаут для HTTP-запросов
        """
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def get_article_links(self, page_url: str, limit: int = 5) -> List[str]:
        """
        Получает список ссылок на статьи с указанной страницы
        
        :param page_url: URL страницы для парсинга
        :param limit: Максимальное количество ссылок
        :return: Список абсолютных URL статей
        """
        try:
            resp = self.session.get(page_url, timeout=self.timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'lxml')
            
            links = []
            
            # Основной селектор для ссылок на статьи
            for a in soup.select('a.tm-title__link')[:limit * 2]:  # Берем с запасом
                href = a.get('href', '')
                if href:
                    # Преобразуем относительные ссылки в абсолютные
                    if href.startswith('/'):
                        full_url = self.BASE_URL + href
                    else:
                        full_url = href
                    
                    # Проверяем, что это статья
                    if '/articles/' in full_url and full_url not in links:
                        links.append(full_url)
                        if len(links) >= limit:
                            break
            
            print(f"[INFO] Найдено {len(links)} уникальных ссылок на статьи с {page_url}")
            return links[:limit]
            
        except Exception as e:
            print(f"[ERROR] Ошибка при получении ссылок с {page_url}: {e}")
            return []
    
    def parse_article(self, url: str) -> Dict:
        """
        Парсит статью по URL
        
        :param url: URL статьи
        :return: Словарь с данными статьи
        """
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 1. Заголовок
            title = ''
            title_tag = soup.select_one('h1.tm-title')
            if title_tag:
                title_span = title_tag.find('span')
                if title_span:
                    title = title_span.get_text(strip=True)
                else:
                    title = title_tag.get_text(strip=True)
            
            # 2. Автор
            author = ''
            author_tag = soup.select_one('a.tm-user-info__username')
            if author_tag:
                author = author_tag.get_text(strip=True).lstrip('@')
            
            # 3. Дата публикации
            date = ''
            time_tag = soup.find('time')
            if time_tag and time_tag.get('datetime'):
                date = time_tag['datetime']
            
            # 4. Текст статьи
            text = ''
            
            # Ищем блок с текстом статьи (версии 1 и 2)
            article_body = None
            for version in ['version-2', 'version-1']:
                article_body = soup.select_one(f'div.article-formatted-body_version-{version}')
                if article_body:
                    break
            
            # Если не нашли по версиям, ищем общий класс
            if not article_body:
                article_body = soup.select_one('div.article-formatted-body')
            
            if article_body:
                # Извлекаем текст из всех параграфов
                paragraphs = article_body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                text_parts = []
                for elem in paragraphs:
                    elem_text = elem.get_text(strip=True)
                    if elem_text and len(elem_text) > 10:
                        text_parts.append(elem_text)
                text = ' '.join(text_parts)
            
            # Очистка текста
            text = self._clean_text(text)
            
            # 5. Теги
            tags = []
            tag_list_div = soup.select_one('div.tag-list')
            if tag_list_div:
                # Ищем все ссылки внутри блока тегов
                for link in tag_list_div.select('a.link'):
                    span_tag = link.find('span')
                    if span_tag:
                        tag_text = span_tag.get_text(strip=True)
                        if tag_text and tag_text not in tags:
                            tags.append(tag_text)
            
            # 6. Дополнительные метаданные
            # Просмотры
            views = 0
            views_elem = soup.select_one('span.tm-icon-counter__value')
            if views_elem:
                views_text = views_elem.get_text(strip=True)
                try:
                    if 'K' in views_text.upper():
                        views = int(float(views_text.upper().replace('K', '')) * 1000)
                    else:
                        views = int(re.sub(r'\D', '', views_text))
                except:
                    views = 0
            
            # Рейтинг
            rating = 0
            rating_elem = soup.select_one('span.tm-votes-lever__score-counter')
            if rating_elem:
                rating_text = rating_elem.get_text(strip=True)
                try:
                    rating = int(rating_text)
                except:
                    rating = 0
            
            # Время чтения
            reading_time = ''
            reading_time_elem = soup.select_one('span.tm-article-reading-time__label')
            if reading_time_elem:
                reading_time = reading_time_elem.get_text(strip=True)
            
            # Уникальный ID статьи (на основе URL)
            article_id = f"habr_{hash(url) & 0xFFFFFFFF:08x}"
            
            return {
                'id': article_id,
                'title': title,
                'author': author,
                'date': date,
                'text': text,
                'tags': tags,
                'url': url,
                'source': 'Habr',
                'views': views,
                'rating': rating,
                'reading_time': reading_time,
                'parsed_at': datetime.now().isoformat(),
                'text_length': len(text),
                'has_content': len(text) > 100
            }
            
        except Exception as e:
            print(f"[ERROR] Ошибка при парсинге статьи {url}: {e}")
            return {
                'id': f"habr_error_{hash(url) & 0xFFFFFFFF:08x}",
                'title': '',
                'author': '',
                'date': '',
                'text': '',
                'tags': [],
                'url': url,
                'source': 'Habr',
                'views': 0,
                'rating': 0,
                'reading_time': '',
                'parsed_at': datetime.now().isoformat(),
                'text_length': 0,
                'has_content': False,
                'error': str(e)
            }
    
    def parse_articles(self, urls: List[str]) -> List[Dict]:
        """
        Парсит несколько статей с задержками
        
        :param urls: Список URL статей
        :return: Список словарей с данными статей
        """
        articles = []
        total = len(urls)
        
        for i, url in enumerate(urls, 1):
            # Задержка для избежания блокировки
            if i > 1:
                sleep_time = self.delay + random.uniform(0, 1.0)
                time.sleep(sleep_time)
            
            print(f"[{i}/{total}] Парсинг: {url[:80]}...")
            article = self.parse_article(url)
            
            if article.get('has_content'):
                articles.append(article)
                print(f"     ✓ {article['title'][:60]}...")
                print(f"       Автор: {article['author'] or 'Не указан'}, "
                      f"Теги: {article['tags'][:3] or 'Нет'}, "
                      f"Символов: {article['text_length']}")
            else:
                print(f"     ✗ Статья не содержит достаточно текста")
        
        return articles
    
    def get_articles_from_page(self, page_url: str, limit: int = 5) -> List[Dict]:
        """
        Полный цикл: получение ссылок + парсинг статей
        
        :param page_url: URL страницы со статьями
        :param limit: Количество статей для парсинга
        :return: Список статей
        """
        print(f"\n{'='*60}")
        print(f"Начинаем парсинг страницы: {page_url}")
        print('='*60)
        
        # Получаем ссылки на статьи
        links = self.get_article_links(page_url, limit=limit)
        
        if not links:
            print("[WARNING] Не найдено ссылок на статьи")
            return []
        
        # Парсим статьи
        articles = self.parse_articles(links)
        
        print(f"\n[ИТОГО] Успешно спаршено: {len(articles)} из {len(links)} статей")
        return articles
    
    def _clean_text(self, text: str) -> str:
        """
        Очистка текста
        
        :param text: Исходный текст
        :return: Очищенный текст
        """
        if not text:
            return ''
        
        # Убираем HTML-сущности
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
        
        # Убираем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text)
        
        # Убираем специальные символы, но сохраняем кириллицу и пунктуацию
        text = re.sub(r'[^\w\s.,!?:;()\-—«»"\'`\u0400-\u04FF]', ' ', text, flags=re.UNICODE)
        
        return text.strip()
    
    def save_to_jsonl(self, articles: List[Dict], filename: str):
        """
        Сохраняет статьи в формате JSONL
        
        :param articles: Список статей
        :param filename: Имя файла для сохранения
        """
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        print(f"[SAVED] Сохранено {len(articles)} статей в {filename}")