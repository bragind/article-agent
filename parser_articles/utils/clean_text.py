import re
import html


def clean_html(text: str) -> str:
    """
    Очистка текста от HTML-тегов и лишних символов
    Сохраняет кириллицу и основную пунктуацию
    """
    if not text:
        return ''
    
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[\xa0\u2000-\u200f\u2028-\u202f]', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:()\-—«»"\'`\u0400-\u04FF]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()