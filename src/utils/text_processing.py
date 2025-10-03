"""
Text processing utilities using SpaCy.
"""

import logging
from typing import List, Optional
import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text preprocessing using SpaCy."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize text processor.
        
        Args:
            model_name: SpaCy model name
        """
        try:
            self.nlp: Language = spacy.load(model_name)
            logger.info(f"Loaded SpaCy model: {model_name}")
        except OSError:
            logger.error(f"SpaCy model '{model_name}' not found. Run: python -m spacy download {model_name}")
            raise
    
    def preprocess(self, text: str, remove_stopwords: bool = True, remove_punct: bool = True) -> str:
        """
        Preprocess text by removing stop words and punctuation, and lemmatizing.
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stop words
            remove_punct: Whether to remove punctuation
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            # Skip based on criteria
            if remove_stopwords and token.is_stop:
                continue
            if remove_punct and token.is_punct:
                continue
            if token.lemma_ == '-PRON-':
                continue
            
            tokens.append(token.text.lower().strip())
        
        return ' '.join(tokens)
    
    def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> List[dict]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            entity_types: List of entity types to extract (e.g., ['SKILL', 'ORG'])
            
        Returns:
            List of entities with their types and positions
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if entity_types is None or ent.label_ in entity_types:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        return entities
    
    def extract_pos_tags(self, text: str) -> List[dict]:
        """
        Extract part-of-speech tags from text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens with POS tags
        """
        doc = self.nlp(text)
        
        return [
            {
                'text': token.text,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_,
                'lemma': token.lemma_
            }
            for token in doc
        ]
    
    def extract_adjectives(self, text: str) -> List[str]:
        """
        Extract adjectives from text (useful for proficiency levels).
        
        Args:
            text: Input text
            
        Returns:
            List of adjectives
        """
        doc = self.nlp(text)
        return [token.text for token in doc if token.pos_ == 'ADJ']


# Proficiency level mapping (from methodology)
PROFICIENCY_LEVELS = {
    'basic': 1,
    'fundamental': 2,
    'beginner': 3,
    'developing': 3,
    'intermediate': 4,
    'capable': 4,
    'proficient': 5,
    'competent': 5,
    'skilled': 6,
    'reliable': 6,
    'advanced': 7,
    'accomplished': 7,
    'experienced': 8,
    'resourceful': 8,
    'expert': 9,
    'specialist': 9,
    'strong': 10,
    'master': 10
}


def get_proficiency_level(adjective: str) -> int:
    """
    Map adjective to proficiency level (1-10).
    
    Args:
        adjective: Proficiency adjective (e.g., 'expert', 'basic')
        
    Returns:
        Proficiency level (1-10), or 5 if not found
    """
    return PROFICIENCY_LEVELS.get(adjective.lower(), 5)

