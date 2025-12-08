import re

def clean_html(text: str) -> str:
    '''
    Простая очистка текста
    '''
    text = re.sub(r'\s+', ' ', text)  # лишние пробелы и переносы
    text = re.sub(r'[^\x00-\x7F]+',' ', text)  # не ASCII символы
    return text.strip()