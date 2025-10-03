"""
Generate contextual sentences for skills using ChatGPT.
"""

import logging
import json
from typing import List, Optional
from openai import OpenAI

from ..config import get_config

logger = logging.getLogger(__name__)


class SkillSentenceGenerator:
    """Generate contextual sentences for skills using ChatGPT."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize skill sentence generator.
        
        Args:
            api_key: OpenAI API key (if None, loads from config)
            model: ChatGPT model name
        """
        config = get_config()
        self.api_key = api_key or config.openai.api_key
        self.model = model
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized SkillSentenceGenerator with model: {model}")
    
    def generate_definition(self, skill: str) -> Optional[str]:
        """
        Generate a contextual definition for a skill.
        
        Args:
            skill: Skill name (e.g., "Python", "R")
            
        Returns:
            Contextual definition or None if generation fails
        """
        prompt = (
            f"Generate a brief professional definition for the skill '{skill}' "
            f"that captures its typical usage in the workplace. "
            f"Keep it to one sentence. "
            f"Return only the definition text without any preamble."
        )
        
        try:
            response = self.client.chat.completions.create(
                temperature=0.1,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise professional definitions."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            definition = response.choices[0].message.content.strip()
            logger.debug(f"Generated definition for '{skill}': {definition}")
            
            return definition
            
        except Exception as e:
            logger.error(f"Error generating definition for '{skill}': {e}")
            return None
    
    def generate_sentences(
        self,
        skills: List[str],
        num_sentences: int = 3
    ) -> dict:
        """
        Generate multiple contextual sentences for a batch of skills.
        
        Args:
            skills: List of skill names
            num_sentences: Number of sentences per skill
            
        Returns:
            Dictionary mapping skills to their sentences
        """
        skills_str = ", ".join(skills)
        
        prompt = (
            f"I have a list of resume skills, and I need to generate {num_sentences} sentences "
            f"that effectively describe each skill in a way that captures its typical usage "
            f"or relationship with other relevant technologies or concepts. "
            f"For each skill, generate sentences that include the skill and contextualize it "
            f"within a relevant scenario, industry practice, or common use case. "
            f"Skills: {skills_str}. "
            f"Return JSON format: {{\"skills\": [{{\"skill\": \"\", \"sentences\": [\"\", \"\", \"\"]}}]}}"
        )
        
        try:
            response = self.client.chat.completions.create(
                temperature=0.1,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.debug(f"Generated sentences for {len(skills)} skills")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating sentences for skills: {e}")
            return {"skills": []}

