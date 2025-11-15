"""Fact extraction from text."""

from typing import List, Dict, Any
import re
from config.logging import StructuredLogger

logger = StructuredLogger(__name__)


class FactExtractor:
    """Extract facts from conversational text."""

    def __init__(self) -> None:
        """Initialize fact extractor."""
        pass

    def extract_facts(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """Extract facts from text.

        Args:
            text: Input text
            context: Additional context

        Returns:
            List of extracted facts
        """
        facts = []

        # Simple fact extraction (in production, use an LLM or NLP model)
        sentences = self._split_sentences(text)

        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 3:
                continue

            # Extract potential facts
            if self._is_factual(sentence):
                fact = {
                    "content": sentence.strip(),
                    "confidence": self._calculate_confidence(sentence),
                    "type": self._classify_fact(sentence),
                    "context": context,
                }
                facts.append(fact)

        logger.debug(f"Extracted {len(facts)} facts from text", text_length=len(text))

        return facts

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_factual(self, sentence: str) -> bool:
        """Determine if a sentence is factual.

        Args:
            sentence: Input sentence

        Returns:
            True if likely factual
        """
        # Heuristic: look for factual indicators
        factual_verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'prefers', 'likes', 'dislikes']
        words = sentence.lower().split()

        return any(verb in words for verb in factual_verbs)

    def _calculate_confidence(self, sentence: str) -> float:
        """Calculate confidence score for a fact.

        Args:
            sentence: Input sentence

        Returns:
            Confidence score (0-1)
        """
        # Simple heuristic: longer sentences with specific terms = higher confidence
        base_confidence = 0.7

        # Boost for specific patterns
        if any(word in sentence.lower() for word in ['always', 'never', 'every', 'all']):
            base_confidence += 0.1

        # Reduce for uncertain language
        if any(word in sentence.lower() for word in ['maybe', 'perhaps', 'might', 'could']):
            base_confidence -= 0.2

        return max(0.1, min(1.0, base_confidence))

    def _classify_fact(self, sentence: str) -> str:
        """Classify the type of fact.

        Args:
            sentence: Input sentence

        Returns:
            Fact type
        """
        lower_sentence = sentence.lower()

        if 'prefer' in lower_sentence or 'like' in lower_sentence:
            return 'preference'
        elif 'is' in lower_sentence or 'are' in lower_sentence:
            return 'attribute'
        elif 'has' in lower_sentence or 'have' in lower_sentence:
            return 'possession'
        else:
            return 'general'
