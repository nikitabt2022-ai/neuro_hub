"""Test memory consolidation components."""

import pytest
from consolidation.fact_extractor import FactExtractor
from consolidation.entity_resolver import EntityResolver
from consolidation.memory_operations import MemoryOperations, MemoryOperation


class TestFactExtractor:
    """Test fact extraction."""

    def test_extract_facts_from_simple_text(self):
        """Test basic fact extraction."""
        extractor = FactExtractor()
        text = "User prefers dark mode. User is learning Python."

        facts = extractor.extract_facts(text)

        assert len(facts) >= 2
        assert all("content" in fact for fact in facts)
        assert all("confidence" in fact for fact in facts)
        assert any("dark mode" in fact["content"].lower() for fact in facts)
        assert any("python" in fact["content"].lower() for fact in facts)

    def test_fact_classification(self):
        """Test fact type classification."""
        extractor = FactExtractor()

        preference = "User prefers TypeScript"
        attribute = "Python is a programming language"
        possession = "User has a MacBook"

        assert extractor._classify_fact(preference) == "preference"
        assert extractor._classify_fact(attribute) == "attribute"
        assert extractor._classify_fact(possession) == "possession"

    def test_confidence_calculation(self):
        """Test confidence scoring."""
        extractor = FactExtractor()

        certain = "User always uses dark mode"
        uncertain = "User might prefer Python"

        cert_confidence = extractor._calculate_confidence(certain)
        uncert_confidence = extractor._calculate_confidence(uncertain)

        assert cert_confidence > uncert_confidence
        assert 0.0 <= cert_confidence <= 1.0
        assert 0.0 <= uncert_confidence <= 1.0

    def test_empty_text_handling(self):
        """Test handling of empty text."""
        extractor = FactExtractor()

        facts = extractor.extract_facts("")
        assert facts == []


class TestEntityResolver:
    """Test entity resolution."""

    def test_exact_match(self):
        """Test exact entity matching."""
        resolver = EntityResolver()
        existing = [
            {"name": "Python", "type": "concept"},
            {"name": "JavaScript", "type": "concept"},
        ]

        similar = resolver.find_similar_entities("Python", existing)

        assert len(similar) >= 1
        assert similar[0]["similarity"] == 1.0
        assert similar[0]["entity"]["name"] == "Python"

    def test_fuzzy_matching(self):
        """Test fuzzy entity matching."""
        resolver = EntityResolver()
        existing = [{"name": "PostgreSQL", "type": "database"}]

        similar = resolver.find_similar_entities("Postgres", existing)

        assert len(similar) >= 1
        assert similar[0]["similarity"] > 0.5

    def test_no_match(self):
        """Test no match scenario."""
        resolver = EntityResolver()
        existing = [{"name": "Python", "type": "concept"}]

        similar = resolver.find_similar_entities("CompletelyDifferent", existing)

        # Should have low or no matches above threshold
        high_similarity = [s for s in similar if s["similarity"] > 0.8]
        assert len(high_similarity) == 0

    def test_resolve_entity_with_match(self):
        """Test entity resolution with existing match."""
        resolver = EntityResolver()
        existing = [{"name": "Python", "type": "concept"}]

        resolved = resolver.resolve_entity("Python", "concept", existing)

        assert resolved is not None
        assert resolved["name"] == "Python"

    def test_resolve_entity_no_match(self):
        """Test entity resolution without match."""
        resolver = EntityResolver()
        existing = [{"name": "Python", "type": "concept"}]

        resolved = resolver.resolve_entity("Java", "concept", existing)

        # Should return None for new entity
        assert resolved is None


class TestMemoryOperations:
    """Test memory consolidation operations."""

    def test_determine_add_operation(self):
        """Test ADD operation for new fact."""
        ops = MemoryOperations()
        new_fact = {"content": "Brand new information", "confidence": 0.9}
        similar_memories = []

        operation, target = ops.determine_operation(new_fact, similar_memories)

        assert operation == MemoryOperation.ADD
        assert target is None

    def test_determine_noop_for_duplicate(self):
        """Test NOOP for exact duplicate."""
        ops = MemoryOperations()
        new_fact = {"content": "User likes Python", "confidence": 0.9}
        similar_memories = [
            {
                "content": "User likes Python",
                "similarity_score": 0.95,
                "metadata": {"confidence": 0.9},
            }
        ]

        operation, target = ops.determine_operation(new_fact, similar_memories)

        assert operation == MemoryOperation.NOOP

    def test_contradictory_facts_detection(self):
        """Test contradictory fact detection."""
        ops = MemoryOperations()

        fact1 = {"content": "User prefers light mode"}
        fact2 = {"content": "User does not prefer light mode"}

        is_contradictory = ops._are_contradictory(fact1, fact2)

        assert is_contradictory is True

    def test_merge_contents(self):
        """Test content merging."""
        ops = MemoryOperations()

        content1 = "User likes Python"
        content2 = "User is learning Python"

        merged = ops._merge_contents(content1, content2)

        assert "Python" in merged
        # Should contain both pieces of information
        assert len(merged) > len(content1)

    def test_execute_add_operation(self):
        """Test executing ADD operation."""
        ops = MemoryOperations()
        new_fact = {"content": "New memory", "confidence": 0.9}

        result = ops.execute_operation(MemoryOperation.ADD, new_fact)

        assert result["operation"] == "add"
        assert "memory" in result

    def test_execute_noop_operation(self):
        """Test executing NOOP operation."""
        ops = MemoryOperations()
        new_fact = {"content": "Duplicate", "confidence": 0.9}

        result = ops.execute_operation(MemoryOperation.NOOP, new_fact)

        assert result["operation"] == "noop"
