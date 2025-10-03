"""
Embedding generation using OpenAI API.
"""

import logging
import time
from typing import List, Union, Optional
import numpy as np
from openai import OpenAI

from ..config import get_config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using OpenAI API."""
    
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        rate_limit_delay: float = 0.1
    ):
        """
        Initialize embedding generator.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (if None, loads from config)
            rate_limit_delay: Delay between API calls (seconds)
        """
        config = get_config()
        self.model = model
        self.api_key = api_key or config.openai.api_key
        self.rate_limit_delay = rate_limit_delay
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Get embedding dimension
        if "small" in model:
            self.embedding_dim = config.openai.embedding_dim_small
        else:
            self.embedding_dim = config.openai.embedding_dim_large
        
        logger.info(f"Initialized EmbeddingGenerator with model: {model} (dim={self.embedding_dim})")
    
    def generate(
        self,
        text: Union[str, List[str]],
        preprocess: bool = True,
        add_delay: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text.
        
        Args:
            text: Input text or list of texts
            preprocess: Whether to preprocess text
            add_delay: Whether to add delay (for rate limiting)
            
        Returns:
            Embedding array or list of embedding arrays
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        if preprocess:
            from ..utils.text_processing import TextProcessor
            processor = TextProcessor()
            texts = [processor.preprocess(t) for t in texts]
        
        embeddings = []
        
        try:
            for t in texts:
                if not t or not t.strip():
                    logger.warning("Empty text provided, using zero vector")
                    embeddings.append(np.zeros(self.embedding_dim))
                    continue
                
                response = self.client.embeddings.create(
                    input=[t],
                    model=self.model
                )
                
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                embeddings.append(embedding)
                
                if add_delay and len(texts) > 1:
                    time.sleep(self.rate_limit_delay)
            
            logger.debug(f"Generated {len(embeddings)} embeddings")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        
        return embeddings[0] if is_single else embeddings
    
    def generate_for_skills(
        self,
        skills: List[str],
        use_definitions: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings for skills, optionally with definitions.
        
        Args:
            skills: List of skill names
            use_definitions: Whether to generate contextual definitions
            
        Returns:
            List of embeddings
        """
        if use_definitions:
            logger.info("Generating contextual definitions for skills...")
            from .skill_sentence_generator import SkillSentenceGenerator
            generator = SkillSentenceGenerator(api_key=self.api_key)
            
            skill_texts = []
            for skill in skills:
                definition = generator.generate_definition(skill)
                skill_texts.append(definition if definition else skill)
        else:
            skill_texts = skills
        
        logger.info(f"Generating embeddings for {len(skills)} skills...")
        return self.generate(skill_texts, preprocess=True)
    
    def batch_generate(
        self,
        texts: List[str],
        batch_size: int = 50,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings in batches.
        
        Args:
            texts: List of texts
            batch_size: Number of texts per batch
            show_progress: Whether to show progress
            
        Returns:
            List of embeddings
        """
        all_embeddings = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), batch_size), desc="Generating embeddings")
        else:
            iterator = range(0, len(texts), batch_size)
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            embeddings = self.generate(batch, add_delay=True)
            all_embeddings.extend(embeddings)
        
        return all_embeddings

