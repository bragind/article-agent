import pandas as pd
from parsers.habr_parser import get_article_links, parse_articles

if __name__ == "__main__":
    # Парсим тестово 5 статей
    habr_main = 'https://habr.com/ru/all/'
    urls = get_article_links(habr_main, limit=5)

    print("Собрано ссылок:", len(urls))

    articles = parse_articles(urls)

    # Преобразуем в DataFrame
    df = pd.DataFrame(articles)

    # Сохраняем CSV
    df.to_csv('parser_articles/data/articles.csv', index=False, encoding='utf-8')
    print("Готово! Файл сохранен: parser_articles/data/articles.csv")