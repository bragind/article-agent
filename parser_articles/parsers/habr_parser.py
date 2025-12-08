import requests
from bs4 import BeautifulSoup
from utils.clean_text import clean_html
from typing import List

BASE_URL = 'https://habr.com'

def get_article_links(page_url: str, limit: int = 5) -> List[str]:
    '''
    Парсит главную страницу и возвращает список ссылок на статьи
    '''
    resp = requests.get(page_url)
    soup = BeautifulSoup(resp.text, 'lxml')

    links = []
    for a in soup.select('a.tm-title__link')[:limit]:
        href = a['href']
        if href.startswith('/ru/'):
            links.append(BASE_URL + href)
    return links

def parse_article(url: str) -> dict:
    '''
    Парсит одну статью
    '''
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'lxml')

    try:
        title = soup.find('h1').text.strip()
    except:
        title = ''

    try:
        author = soup.select_one('a.user-info__nickname')
    except:
        author = ''

    try:
        date = soup.find('time')['datetime']
    except:
        date = ''

    try:
        text_blocks = soup.select('div.article-formatted-body > p')
        text = ' '.join([p.text for p in text_blocks])
        text = clean_html(text)
    except:
        text = ''

    try:
        tags = [t.text.strip() for t in soup.select('a.tm-article-snippet__hubs-item-link')]
    except:
        tags = []

    return {
        'title': title,
        'author': author,
        'date': date,
        'text': text,
        'tags': tags,
        'url': url,
        'source': 'Habr'
    }

def parse_articles(urls: List[str]) -> List[dict]:
    '''
    Цикл по всем статьям
    '''
    articles = []
    for url in urls:
        try:
            article = parse_article(url)
            articles.append(article)
            print(f'[OK] {url}')
        except Exception as e:
            print (f'[ERROR] {url} -> {e}')
    return articles