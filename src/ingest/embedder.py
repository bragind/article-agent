"""
Генерация эмбеддингов для текста
"""

import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """Генератор эмбеддингов на основе Sentence Transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            model_name: Название модели для эмбеддингов
        """
        self.model = None
        self.model_name = model_name
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Загрузка модели эмбеддингов: {model_name}")
            
            # Безопасная загрузка модели
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Модель загружена. Размерность: {self.model.get_sentence_embedding_dimension()}")
            except Exception as e:
                logger.error(f"Ошибка загрузки основной модели {model_name}: {e}")
                # Fallback на простую модель
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info(f"Использую fallback модель: all-MiniLM-L6-v2")
                
        except ImportError as e:
            logger.error(f"Не удалось импортировать SentenceTransformer: {e}")
            raise ImportError(
                "Для использования Embedder установите sentence-transformers:\n"
                "pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Критическая ошибка инициализации Embedder: {e}")
            self.model = None
    
    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Генерация эмбеддингов для списка текстов
        
        Args:
            texts: Список текстов
            
        Returns:
            Матрица эмбеддингов или None при ошибке
        """
        if not self.model:
            logger.error("Модель эмбеддингов не инициализирована")
            return None
        
        if not texts:
            logger.warning("Пустой список текстов для эмбеддинга")
            return np.array([])
        
        try:
            logger.info(f"Генерация эмбеддингов для {len(texts)} текстов")
            
            # Безопасная генерация эмбеддингов
            if len(texts) == 1:
                # Для одного текста
                result = self.model.encode(
                    texts[0], 
                    show_progress_bar=False, 
                    normalize_embeddings=True
                )
                return np.array([result])
            else:
                # Для нескольких текстов
                result = self.model.encode(
                    texts, 
                    show_progress_bar=True, 
                    normalize_embeddings=True,
                    batch_size=32
                )
                return result
                
        except Exception as e:
            logger.error(f"Ошибка генерации эмбеддингов: {e}")
            return None
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Получение размерности эмбеддингов"""
        if not self.model:
            return None
        
        try:
            return self.model.get_sentence_embedding_dimension()
        except:
            return None
    
    def is_ready(self) -> bool:
        """Проверка готовности модели"""
        return self.model is not None