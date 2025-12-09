import requests
from bs4 import BeautifulSoup
import time
import random
import re
from datetime import datetime
from typing import List, Dict
import json
import os

# Импортируем конфигурацию
from .config import (
    MAX_ARTICLES, ARTICLES_PER_PAGE, PAGES_TO_CHECK, 
    ARTICLES_PER_SECTION, BASE_DELAY, MAX_RANDOM_DELAY,
    ERROR_DELAY, SECTIONS
)
# Импортируем чистку текста из отдельного модуля
from ..utils.clean_text import clean_html


class HabrParser:
    """
    Парсер статей с Habr.com
    """
    
    BASE_URL = 'https://habr.com'
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    def __init__(self, delay: float = BASE_DELAY, timeout: int = 15):
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def get_article_links(self, page_url: str, limit: int = ARTICLES_PER_SECTION) -> List[str]:
        """
        Получает список ссылок на статьи с указанной страницы
        """
        try:
            resp = self.session.get(page_url, timeout=self.timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'lxml')
            
            links = []
            for a in soup.select('a.tm-title__link')[:limit * 2]:
                href = a.get('href', '')
                if href:
                    if href.startswith('/'):
                        full_url = self.BASE_URL + href
                    else:
                        full_url = href
                    
                    if '/articles/' in full_url and full_url not in links:
                        links.append(full_url)
                        if len(links) >= limit:
                            break
            
            print(f"Найдено {len(links)} ссылок на статьи с {page_url}")
            return links[:limit]
            
        except Exception as e:
            print(f"Ошибка при получении ссылок с {page_url}: {e}")
            return []
    
    def get_article_links_with_pagination(self, base_url: str, max_articles: int = MAX_ARTICLES) -> List[str]:
        """
        Собирает ссылки на статьи с нескольких страниц (пагинация)
        """
        all_links = []
        page_num = 1
        
        while len(all_links) < max_articles and page_num <= PAGES_TO_CHECK:
            if page_num == 1:
                page_url = base_url
            else:
                page_url = f"{base_url}page{page_num}/"
            
            print(f"Парсим страницу {page_num}: {page_url}")
            
            try:
                links = self.get_article_links(page_url, limit=ARTICLES_PER_PAGE)
                new_links = [link for link in links if link not in all_links]
                all_links.extend(new_links)
                
                print(f"Найдено {len(new_links)} новых ссылок. Всего: {len(all_links)}")
                
                if len(new_links) == 0:
                    print("Новые ссылки не найдены. Прекращаем парсинг.")
                    break
                    
            except Exception as e:
                print(f"Ошибка при парсинге страницы {page_num}: {e}")
            
            time.sleep(self.delay + random.uniform(0, MAX_RANDOM_DELAY))
            page_num += 1
        
        return all_links[:max_articles]
    
    def parse_article(self, url: str) -> Dict:
        """
        Парсит статью по URL
        """
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            title = ''
            title_tag = soup.select_one('h1.tm-title')
            if title_tag:
                title_span = title_tag.find('span')
                if title_span:
                    title = title_span.get_text(strip=True)
                else:
                    title = title_tag.get_text(strip=True)
            
            author = ''
            author_tag = soup.select_one('a.tm-user-info__username')
            if author_tag:
                author = author_tag.get_text(strip=True).lstrip('@')
            
            date = ''
            time_tag = soup.find('time')
            if time_tag and time_tag.get('datetime'):
                date = time_tag['datetime']
            
            text = ''
            article_body = None
            for version in ['version-2', 'version-1']:
                article_body = soup.select_one(f'div.article-formatted-body_version-{version}')
                if article_body:
                    break
            
            if not article_body:
                article_body = soup.select_one('div.article-formatted-body')
            
            if article_body:
                paragraphs = article_body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                text_parts = []
                for elem in paragraphs:
                    elem_text = elem.get_text(strip=True)
                    if elem_text and len(elem_text) > 10:
                        text_parts.append(elem_text)
                text = ' '.join(text_parts)
            
            text = clean_html(text)
            
            tags = []
            tag_list_div = soup.select_one('div.tag-list')
            if tag_list_div:
                for link in tag_list_div.select('a.link'):
                    span_tag = link.find('span')
                    if span_tag:
                        tag_text = span_tag.get_text(strip=True)
                        if tag_text and tag_text not in tags:
                            tags.append(tag_text)
            
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
            
            rating = 0
            rating_elem = soup.select_one('span.tm-votes-lever__score-counter')
            if rating_elem:
                rating_text = rating_elem.get_text(strip=True)
                try:
                    rating = int(rating_text)
                except:
                    rating = 0
            
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
                'parsed_at': datetime.now().isoformat(),
                'text_length': len(text),
                'has_content': len(text) > 100
            }
            
        except Exception as e:
            print(f"Ошибка при парсинге статьи {url}: {e}")
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
                'parsed_at': datetime.now().isoformat(),
                'text_length': 0,
                'has_content': False,
                'error': str(e)
            }
    
    def parse_articles(self, urls: List[str]) -> List[Dict]:
        """
        Парсит несколько статей с задержками
        """
        articles = []
        total = len(urls)
        
        for i, url in enumerate(urls, 1):
            if i > 1:
                sleep_time = self.delay + random.uniform(0, MAX_RANDOM_DELAY)
                time.sleep(sleep_time)
            
            print(f"[{i}/{total}] Парсинг: {url[:80]}...")
            article = self.parse_article(url)
            
            if article.get('has_content'):
                articles.append(article)
                print(f"Успешно: {article['title'][:60]}...")
                print(f"Автор: {article['author'] or 'Не указан'}, Теги: {article['tags'][:3] or 'Нет'}, Символов: {article['text_length']}")
            else:
                print(f"Статья не содержит достаточно текста")
        
        return articles
    
    def save_to_jsonl(self, articles: List[Dict], filename: str):
        """
        Сохраняет статьи в формате JSONL
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        print(f"Сохранено {len(articles)} статей в {filename}")