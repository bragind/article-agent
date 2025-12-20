"""
Разбиение текста на чанки с учетом семантических границ
"""

import re
from typing import List


class TextChunker:
    """Разбивает текст на чанки с учетом структуры текста"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Args:
            chunk_size: Размер чанка в символах (рекомендуется 600-1000 для лучшего контекста)
            chunk_overlap: Перекрытие между чанками в символах
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        Разбивает текст на чанки, стараясь сохранить смысловые границы
        
        Args:
            text: Входной текст
            
        Returns:
            Список чанков
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        # Сначала пробуем разбить по абзацам
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Если текущий чанк + абзац не превышает размер чанка, добавляем
            if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Сохраняем текущий чанк, если он не пустой
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Если абзац сам по себе больше чанка, разбиваем его
                if len(paragraph) > self.chunk_size:
                    # Разбиваем большой абзац на части
                    sub_chunks = self._split_long_paragraph(paragraph)
                    chunks.extend(sub_chunks[:-1])  # Все кроме последнего
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Если не получилось разбить по абзацам, используем старый метод
        if not chunks or (len(chunks) == 1 and len(chunks[0]) > self.chunk_size * 1.5):
            return self._split_by_fixed_size(text)
        
        return chunks
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Разбивает длинный абзац на части по предложениям"""
        # Простое разбиение по предложениям (точка, восклицательный, вопросительный знак)
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_fixed_size(self, text: str) -> List[str]:
        """Разбивает текст на чанки фиксированного размера (старый метод)"""
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Пытаемся закончить на границе предложения
            if end < len(text):
                # Ищем конец предложения ближе к концу чанк
                for lookahead in range(100):
                    if end + lookahead >= len(text):
                        break
                    if text[end + lookahead] in '.!?':
                        end = end + lookahead + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            
            if start >= len(text):
                break
        
        return chunks