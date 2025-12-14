"""
Генерация эмбеддингов для текста
"""

import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """Генератор эмбеддингов на основе Sentence Transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            model_name: Название многоязычной модели для эмбеддингов
                      Рекомендуется для русского языка:
                      - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                      - 'intfloat/multilingual-e5-small' 
        """
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Загрузка модели эмбеддингов: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info(f"Модель загружена. Размерность эмбеддингов: {self.model.get_sentence_embedding_dimension()}")
        except ImportError:
            raise ImportError("Для использования Embedder установите sentence-transformers: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_name}: {e}")
            # Fallback на английскую модель
            logger.info("Использую fallback модель: all-MiniLM-L6-v2")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Генерация эмбеддингов для списка текстов
        
        Args:
            texts: Список текстов
            
        Returns:
            Матрица эмбеддингов
        """
        logger.info(f"Генерация эмбеддингов для {len(texts)} текстов")
        # normalize_embeddings=True улучшает качество поиска
        return self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)