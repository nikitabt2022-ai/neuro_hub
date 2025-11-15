# 🧪 Neuro Hub - Core Functionality Testing Plan

**Focused testing strategy for critical system components**

---

## 📋 Testing Philosophy

**Focus Areas:**
1. **Core Memory Operations** - Store, retrieve, search, delete
2. **Data Integrity** - Consistency across storage layers
3. **Search Quality** - Semantic and hybrid search accuracy
4. **API Contracts** - Request/response validation
5. **MCP Integration** - Tool execution and resource access
6. **Performance** - Latency and throughput benchmarks

**Testing Levels:**
- ✅ **Unit Tests** - Individual components (70% coverage target)
- ✅ **Integration Tests** - Component interactions (20% coverage target)
- ✅ **E2E Tests** - Full workflows (10% coverage target)

---

## 🎯 Test Priority Matrix

| Component | Priority | Complexity | Coverage Target |
|-----------|----------|------------|-----------------|
| Memory Services | 🔴 Critical | Medium | 90% |
| Vector Store | 🔴 Critical | High | 85% |
| Graph Store | 🟡 High | High | 80% |
| Embeddings | 🔴 Critical | Medium | 85% |
| Consolidation | 🟡 High | High | 75% |
| Hybrid Search | 🟡 High | Medium | 80% |
| API Endpoints | 🔴 Critical | Low | 90% |
| MCP Tools | 🔴 Critical | Medium | 85% |
| Cache Store | 🟢 Medium | Low | 70% |

---

## 🧪 Test Suite Structure

```
tests/
├── unit/
│   ├── test_models.py              # Pydantic model validation
│   ├── test_embeddings.py          # Embedding generation
│   ├── test_vector_store.py        # Vector operations
│   ├── test_graph_store.py         # Graph operations
│   ├── test_cache_store.py         # Cache operations
│   ├── test_hybrid_search.py       # Search algorithms
│   ├── test_consolidation.py       # Consolidation logic
│   └── test_services.py            # Service layer
│
├── integration/
│   ├── test_memory_workflows.py    # End-to-end memory flows
│   ├── test_storage_layer.py       # Multi-storage operations
│   ├── test_api_endpoints.py       # API integration
│   └── test_mcp_server.py          # MCP tool execution
│
├── e2e/
│   ├── test_full_memory_lifecycle.py
│   ├── test_knowledge_graph_building.py
│   └── test_search_scenarios.py
│
├── performance/
│   ├── test_search_latency.py
│   ├── test_throughput.py
│   └── test_concurrent_operations.py
│
├── conftest.py                      # Pytest fixtures
├── test_data/                       # Sample data
│   ├── memories.json
│   ├── entities.json
│   └── conversations.txt
└── fixtures/                        # Test fixtures
    ├── mock_embeddings.py
    └── mock_databases.py
```

---

## 📝 Core Test Cases

### 1. Memory Services Tests

#### 1.1 Episodic Memory Service
```python
# tests/unit/test_services.py

import pytest
from uuid import uuid4
from models.memory import MemoryType, MemoryCreate, MemoryMetadata
from services.episodic_memory_service import EpisodicMemoryService

class TestEpisodicMemoryService:
    """Test episodic memory operations."""

    @pytest.mark.asyncio
    async def test_store_conversation(self, episodic_service):
        """Test storing a conversation memory."""
        # Arrange
        content = "User asked about Python tutorials"
        session_id = "session_123"

        # Act
        memory = await episodic_service.store_conversation(
            content=content,
            session_id=session_id,
            user_id="user_456"
        )

        # Assert
        assert memory.id is not None
        assert memory.type == MemoryType.EPISODIC
        assert memory.content == content
        assert memory.metadata.get("session_id") == session_id
        assert memory.embedding is not None
        assert len(memory.embedding) == 384  # Model dimension

    @pytest.mark.asyncio
    async def test_retrieve_memory(self, episodic_service):
        """Test retrieving a specific memory."""
        # Arrange
        stored = await episodic_service.store_conversation(
            content="Test conversation",
            session_id="session_789"
        )

        # Act
        retrieved = await episodic_service.get(stored.id)

        # Assert
        assert retrieved is not None
        assert retrieved.id == stored.id
        assert retrieved.content == stored.content

    @pytest.mark.asyncio
    async def test_search_by_content(self, episodic_service, sample_memories):
        """Test semantic search."""
        # Arrange
        await episodic_service.store_conversation(
            content="User prefers dark mode",
            session_id="session_1"
        )
        await episodic_service.store_conversation(
            content="User likes Python programming",
            session_id="session_2"
        )

        # Act
        results = await episodic_service.search(
            query="What does the user prefer?",
            limit=5,
            min_similarity=0.3
        )

        # Assert
        assert len(results) > 0
        assert any("dark mode" in r.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_delete_memory(self, episodic_service):
        """Test memory deletion."""
        # Arrange
        memory = await episodic_service.store_conversation(
            content="Temporary memory",
            session_id="temp"
        )

        # Act
        deleted = await episodic_service.delete(memory.id)

        # Assert
        assert deleted is True
        retrieved = await episodic_service.get(memory.id)
        assert retrieved is None
```

#### 1.2 Semantic Memory Service
```python
class TestSemanticMemoryService:
    """Test semantic memory operations."""

    @pytest.mark.asyncio
    async def test_store_fact(self, semantic_service):
        """Test storing a semantic fact."""
        # Arrange
        fact = "User is allergic to peanuts"
        entities = ["User", "peanuts"]

        # Act
        memory = await semantic_service.store_fact(
            fact=fact,
            entities=entities,
            confidence=0.95,
            source="conversation"
        )

        # Assert
        assert memory.type == MemoryType.SEMANTIC
        assert memory.content == fact
        assert memory.metadata.get("entities") == entities
        assert memory.metadata.get("confidence") == 0.95

    @pytest.mark.asyncio
    async def test_search_facts_by_confidence(self, semantic_service):
        """Test filtering facts by confidence threshold."""
        # Arrange
        await semantic_service.store_fact(
            fact="High confidence fact",
            confidence=0.9
        )
        await semantic_service.store_fact(
            fact="Low confidence fact",
            confidence=0.3
        )

        # Act
        results = await semantic_service.search_facts(
            query="fact",
            min_confidence=0.7
        )

        # Assert
        assert len(results) >= 1
        assert all(r.metadata.get("confidence", 0) >= 0.7 for r in results)
```

---

### 2. Storage Layer Tests

#### 2.1 Vector Store (Milvus)
```python
# tests/unit/test_vector_store.py

import pytest
from storage.vector_store import VectorStore
from models.memory import Memory, MemoryType, MemoryMetadata

class TestVectorStore:
    """Test vector database operations."""

    def test_connect_to_milvus(self, vector_store):
        """Test Milvus connection."""
        # Act
        vector_store.connect()

        # Assert
        assert vector_store._connected is True

    def test_create_collection(self, vector_store):
        """Test collection creation."""
        # Act
        vector_store.create_collection("test_collection", MemoryType.SEMANTIC)

        # Assert
        collections = vector_store.list_collections()
        assert any("test_collection" in c for c in collections)

    def test_insert_and_retrieve(self, vector_store, sample_memory):
        """Test inserting and retrieving a memory."""
        # Arrange
        memory = sample_memory
        memory.embedding = [0.1] * 384  # Mock embedding

        # Act - Insert
        vector_store.insert(memory)

        # Act - Retrieve
        result = vector_store.get(
            memory.id,
            memory.collection,
            memory.type
        )

        # Assert
        assert result is not None
        assert result["id"] == str(memory.id)
        assert result["content"] == memory.content

    def test_similarity_search(self, vector_store):
        """Test vector similarity search."""
        # Arrange
        memories = [
            create_test_memory("User likes Python", [0.1, 0.2] + [0.0] * 382),
            create_test_memory("User prefers Java", [0.9, 0.8] + [0.0] * 382),
            create_test_memory("Python is great", [0.15, 0.25] + [0.0] * 382),
        ]

        for mem in memories:
            vector_store.insert(mem)

        # Act
        query_embedding = [0.1, 0.2] + [0.0] * 382
        results = vector_store.search(
            embedding=query_embedding,
            collection="default",
            memory_type=MemoryType.SEMANTIC,
            limit=2
        )

        # Assert
        assert len(results) == 2
        assert results[0]["similarity_score"] > results[1]["similarity_score"]
```

#### 2.2 Graph Store (Neo4j)
```python
# tests/unit/test_graph_store.py

class TestGraphStore:
    """Test graph database operations."""

    def test_create_entity(self, graph_store):
        """Test entity creation."""
        # Arrange
        entity = Entity(
            name="John Doe",
            type=EntityType.PERSON,
            description="Software engineer"
        )

        # Act
        graph_store.create_entity(entity)
        result = graph_store.get_entity(entity.id)

        # Assert
        assert result is not None
        assert result["name"] == "John Doe"
        assert result["type"] == "person"

    def test_create_relationship(self, graph_store):
        """Test relationship creation."""
        # Arrange
        entity1 = create_and_store_entity(graph_store, "Alice", EntityType.PERSON)
        entity2 = create_and_store_entity(graph_store, "Acme Corp", EntityType.ORGANIZATION)

        relationship = Relationship(
            source_id=entity1.id,
            target_id=entity2.id,
            type=RelationType.WORKS_FOR,
            weight=1.0
        )

        # Act
        graph_store.create_relationship(relationship)

        # Assert
        rels = graph_store.get_entity_relationships(entity1.id, "outgoing")
        assert len(rels) >= 1
        assert rels[0]["relationship"]["type"] == "works_for"

    def test_find_path(self, graph_store):
        """Test path finding between entities."""
        # Arrange - Create graph: A -> B -> C
        entity_a = create_and_store_entity(graph_store, "A", EntityType.PERSON)
        entity_b = create_and_store_entity(graph_store, "B", EntityType.PERSON)
        entity_c = create_and_store_entity(graph_store, "C", EntityType.PERSON)

        create_relationship(graph_store, entity_a.id, entity_b.id)
        create_relationship(graph_store, entity_b.id, entity_c.id)

        # Act
        path = graph_store.find_path(entity_a.id, entity_c.id, max_depth=3)

        # Assert
        assert path is not None
        assert len(path) >= 3  # A, rel, B, rel, C
```

---

### 3. Embedding Tests

```python
# tests/unit/test_embeddings.py

class TestTextEmbedder:
    """Test text embedding generation."""

    def test_load_model(self, embedder):
        """Test model loading."""
        # Act
        embedder.load_model()

        # Assert
        assert embedder._model is not None

    def test_embed_single_text(self, embedder):
        """Test embedding a single text."""
        # Arrange
        text = "This is a test sentence"

        # Act
        embedding = embedder.embed(text)

        # Assert
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_batch(self, embedder):
        """Test batch embedding."""
        # Arrange
        texts = [
            "First sentence",
            "Second sentence",
            "Third sentence"
        ]

        # Act
        embeddings = embedder.embed(texts)

        # Assert
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

    def test_similarity_calculation(self, embedder):
        """Test cosine similarity."""
        # Arrange
        text1 = "I love Python programming"
        text2 = "Python is my favorite language"
        text3 = "Java is used in enterprise"

        emb1 = embedder.embed(text1)
        emb2 = embedder.embed(text2)
        emb3 = embedder.embed(text3)

        # Act
        sim_12 = embedder.similarity(emb1, emb2)
        sim_13 = embedder.similarity(emb1, emb3)

        # Assert
        assert sim_12 > sim_13  # More similar sentences
        assert 0.0 <= sim_12 <= 1.0
        assert 0.0 <= sim_13 <= 1.0

class TestEmbeddingCache:
    """Test embedding cache."""

    def test_cache_miss(self, embedding_cache):
        """Test cache miss."""
        # Act
        result = embedding_cache.get("new text")

        # Assert
        assert result is None

    def test_cache_hit(self, embedding_cache):
        """Test cache hit."""
        # Arrange
        text = "cached text"
        embedding = [0.1, 0.2, 0.3]
        embedding_cache.set(text, embedding)

        # Act
        result = embedding_cache.get(text)

        # Assert
        assert result == embedding

    def test_lru_eviction(self):
        """Test LRU cache eviction."""
        # Arrange
        cache = EmbeddingCache(max_size=3)

        # Act - Fill cache beyond capacity
        cache.set("text1", [0.1])
        cache.set("text2", [0.2])
        cache.set("text3", [0.3])
        cache.set("text4", [0.4])  # Should evict text1

        # Assert
        assert cache.get("text1") is None
        assert cache.get("text4") is not None
```

---

### 4. Consolidation Tests

```python
# tests/unit/test_consolidation.py

class TestFactExtractor:
    """Test fact extraction."""

    def test_extract_facts_from_text(self, fact_extractor):
        """Test extracting facts from conversation."""
        # Arrange
        text = "User prefers dark mode. User is learning Python. User lives in California."

        # Act
        facts = fact_extractor.extract_facts(text)

        # Assert
        assert len(facts) >= 2
        assert any("dark mode" in f["content"].lower() for f in facts)
        assert all("confidence" in f for f in facts)

    def test_fact_classification(self, fact_extractor):
        """Test fact type classification."""
        # Arrange
        preference = "User prefers TypeScript"
        attribute = "Python is a programming language"

        # Act
        pref_type = fact_extractor._classify_fact(preference)
        attr_type = fact_extractor._classify_fact(attribute)

        # Assert
        assert pref_type == "preference"
        assert attr_type == "attribute"

class TestEntityResolver:
    """Test entity resolution."""

    def test_find_exact_match(self, entity_resolver):
        """Test finding exact entity match."""
        # Arrange
        existing = [
            {"name": "Python", "type": "concept"},
            {"name": "JavaScript", "type": "concept"}
        ]

        # Act
        similar = entity_resolver.find_similar_entities(
            "Python",
            existing
        )

        # Assert
        assert len(similar) >= 1
        assert similar[0]["similarity"] == 1.0

    def test_fuzzy_matching(self, entity_resolver):
        """Test fuzzy entity matching."""
        # Arrange
        existing = [{"name": "PostgreSQL", "type": "database"}]

        # Act
        similar = entity_resolver.find_similar_entities(
            "Postgres",
            existing
        )

        # Assert
        assert len(similar) >= 1
        assert similar[0]["similarity"] > 0.7

class TestMemoryOperations:
    """Test memory consolidation operations."""

    def test_determine_add_operation(self, memory_ops):
        """Test ADD operation detection."""
        # Arrange
        new_fact = {"content": "Brand new information", "confidence": 0.9}
        similar_memories = []

        # Act
        operation, target = memory_ops.determine_operation(
            new_fact,
            similar_memories
        )

        # Assert
        assert operation == MemoryOperation.ADD
        assert target is None

    def test_determine_duplicate_noop(self, memory_ops):
        """Test NOOP for duplicate."""
        # Arrange
        new_fact = {"content": "User likes Python", "confidence": 0.9}
        similar_memories = [
            {"content": "User likes Python", "similarity_score": 0.95}
        ]

        # Act
        operation, target = memory_ops.determine_operation(
            new_fact,
            similar_memories
        )

        # Assert
        assert operation == MemoryOperation.NOOP

    def test_determine_update_for_contradiction(self, memory_ops):
        """Test UPDATE for contradictory facts."""
        # Arrange
        new_fact = {"content": "User now prefers dark mode", "confidence": 0.95}
        similar_memories = [
            {
                "content": "User prefers light mode",
                "similarity_score": 0.85,
                "metadata": {"confidence": 0.7}
            }
        ]

        # Act
        operation, target = memory_ops.determine_operation(
            new_fact,
            similar_memories
        )

        # Assert
        assert operation in [MemoryOperation.UPDATE, MemoryOperation.ADD]
```

---

### 5. Hybrid Search Tests

```python
# tests/unit/test_hybrid_search.py

class TestHybridSearch:
    """Test hybrid search functionality."""

    def test_bm25_search(self, hybrid_search):
        """Test BM25 keyword search."""
        # Arrange
        documents = [
            {"content": "Python programming tutorial"},
            {"content": "Java enterprise development"},
            {"content": "Python web development"}
        ]
        query = "Python"

        # Act
        results = hybrid_search._bm25_search(query, documents, limit=2)

        # Assert
        assert len(results) == 2
        assert all("Python" in documents[idx]["content"] for idx, _ in results)

    def test_reciprocal_rank_fusion(self, hybrid_search):
        """Test RRF fusion algorithm."""
        # Arrange
        bm25_results = [(0, 0.9), (1, 0.7), (2, 0.5)]
        vector_results = [(2, 0.95), (0, 0.8), (3, 0.6)]

        # Act
        fused = hybrid_search._reciprocal_rank_fusion(
            bm25_results,
            vector_results
        )

        # Assert
        assert len(fused) >= 3
        # Verify scoring combines both methods
        assert fused[0][1] > 0  # Has a score

    def test_hybrid_search_integration(self, hybrid_search):
        """Test full hybrid search."""
        # Arrange
        documents = [
            {"content": "Machine learning with Python"},
            {"content": "Deep learning neural networks"},
            {"content": "Python data science tutorial"}
        ]
        vector_scores = [(0, 0.9), (2, 0.85), (1, 0.6)]

        # Act
        results = hybrid_search.search(
            query="Python machine learning",
            documents=documents,
            vector_scores=vector_scores,
            limit=2
        )

        # Assert
        assert len(results) <= 2
        assert all("rank" in r for r in results)
        assert all("hybrid_score" in r for r in results)
```

---

### 6. API Integration Tests

```python
# tests/integration/test_api_endpoints.py

import pytest
from httpx import AsyncClient
from api.main import app

class TestMemoryEndpoints:
    """Test REST API endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_store_semantic_memory(self):
        """Test storing semantic memory via API."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "type": "semantic",
                "content": "User is allergic to shellfish",
                "metadata": {
                    "tags": ["health", "dietary"],
                    "confidence": 0.95
                },
                "collection": "test"
            }

            response = await client.post("/api/v1/semantic/", json=payload)

            assert response.status_code == 201
            data = response.json()
            assert data["content"] == payload["content"]
            assert "id" in data

    @pytest.mark.asyncio
    async def test_search_memories(self):
        """Test search endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Store test data
            await client.post("/api/v1/semantic/", json={
                "type": "semantic",
                "content": "User loves Python programming",
                "collection": "test"
            })

            # Search
            response = await client.get(
                "/api/v1/search/",
                params={
                    "query": "What programming languages?",
                    "memory_type": "semantic",
                    "collection": "test"
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
```

---

### 7. MCP Server Tests

```python
# tests/integration/test_mcp_server.py

class TestMCPTools:
    """Test MCP server tools."""

    @pytest.mark.asyncio
    async def test_store_memory_tool(self, mcp_client):
        """Test MCP store_memory tool."""
        # Arrange
        params = {
            "content": "User prefers concise responses",
            "memory_type": "semantic",
            "tags": ["preference", "communication"],
            "confidence": 0.9
        }

        # Act
        result = await mcp_client.call_tool("store_memory", params)

        # Assert
        assert "id" in result
        assert result["type"] == "semantic"
        assert result["content"] == params["content"]

    @pytest.mark.asyncio
    async def test_search_memories_tool(self, mcp_client):
        """Test MCP search_memories tool."""
        # Arrange
        await mcp_client.call_tool("store_memory", {
            "content": "User is learning machine learning",
            "memory_type": "semantic"
        })

        # Act
        result = await mcp_client.call_tool("search_memories", {
            "query": "What is the user learning?",
            "memory_type": "semantic",
            "limit": 5
        })

        # Assert
        assert "results" in result
        assert result["count"] >= 1

    @pytest.mark.asyncio
    async def test_create_entity_tool(self, mcp_client):
        """Test MCP create_entity tool."""
        # Arrange
        params = {
            "name": "Claude AI",
            "entity_type": "organization",
            "description": "AI assistant"
        }

        # Act
        result = await mcp_client.call_tool("create_entity", params)

        # Assert
        assert "id" in result
        assert result["name"] == params["name"]
```

---

### 8. End-to-End Tests

```python
# tests/e2e/test_full_memory_lifecycle.py

class TestFullMemoryLifecycle:
    """Test complete memory workflows."""

    @pytest.mark.asyncio
    async def test_conversation_to_knowledge_graph(
        self,
        episodic_service,
        semantic_service,
        graph_service,
        fact_extractor
    ):
        """Test extracting knowledge from conversation."""
        # 1. Store conversation
        conversation = await episodic_service.store_conversation(
            content="Alice works at Google and lives in San Francisco",
            session_id="session_1"
        )

        # 2. Extract facts
        facts = fact_extractor.extract_facts(conversation.content)
        assert len(facts) >= 2

        # 3. Store facts as semantic memories
        for fact in facts:
            await semantic_service.store_fact(
                fact=fact["content"],
                confidence=fact["confidence"]
            )

        # 4. Create entities
        alice = await graph_service.create_entity(EntityCreate(
            name="Alice",
            type=EntityType.PERSON
        ))

        google = await graph_service.create_entity(EntityCreate(
            name="Google",
            type=EntityType.ORGANIZATION
        ))

        # 5. Create relationship
        await graph_service.create_relationship(RelationshipCreate(
            source_id=alice.id,
            target_id=google.id,
            type=RelationType.WORKS_FOR
        ))

        # 6. Verify knowledge graph
        rels = await graph_service.get_entity_relationships(alice.id)
        assert len(rels) >= 1
        assert rels[0]["relationship"]["type"] == "works_for"

    @pytest.mark.asyncio
    async def test_memory_consolidation_workflow(
        self,
        semantic_service,
        memory_ops,
        entity_resolver
    ):
        """Test memory consolidation pipeline."""
        # 1. Store initial fact
        fact1 = await semantic_service.store_fact(
            fact="User prefers Python",
            confidence=0.8
        )

        # 2. Store similar fact (should deduplicate)
        similar_facts = await semantic_service.search_facts(
            query="Python preference",
            limit=5
        )

        operation, target = memory_ops.determine_operation(
            new_fact={"content": "User likes Python", "confidence": 0.9},
            similar_memories=[{
                "content": f.content,
                "similarity_score": 0.95
            } for f in similar_facts]
        )

        # 3. Verify consolidation decision
        assert operation == MemoryOperation.NOOP  # Duplicate detected
```

---

## 🚀 Performance Tests

```python
# tests/performance/test_search_latency.py

class TestSearchLatency:
    """Test search performance benchmarks."""

    @pytest.mark.performance
    async def test_vector_search_latency(self, semantic_service, benchmark):
        """Test vector search meets latency target."""
        # Arrange
        for i in range(100):
            await semantic_service.store_fact(
                fact=f"Test fact number {i}",
                confidence=0.9
            )

        # Act & Benchmark
        result = benchmark(
            lambda: semantic_service.search(
                query="test fact",
                limit=10
            )
        )

        # Assert - Target: < 50ms p95
        assert benchmark.stats["mean"] < 0.050  # 50ms

    @pytest.mark.performance
    async def test_hybrid_search_latency(self, benchmark):
        """Test hybrid search meets latency target."""
        # Target: < 100ms p95
        pass

class TestThroughput:
    """Test throughput benchmarks."""

    @pytest.mark.performance
    async def test_concurrent_writes(self, semantic_service):
        """Test concurrent write throughput."""
        import asyncio

        # Arrange
        tasks = []
        for i in range(1000):
            task = semantic_service.store_fact(
                fact=f"Concurrent fact {i}",
                confidence=0.9
            )
            tasks.append(task)

        # Act
        start = time.time()
        await asyncio.gather(*tasks)
        duration = time.time() - start

        # Assert - Target: 1000+ writes/sec
        throughput = 1000 / duration
        assert throughput >= 100  # At least 100 QPS for this test
```

---

## 🔧 Test Fixtures

```python
# tests/conftest.py

import pytest
import asyncio
from storage.vector_store import VectorStore
from storage.graph_store import GraphStore
from storage.cache_store import CacheStore
from embeddings.text_embedder import TextEmbedder
from embeddings.embedding_cache import EmbeddingCache
from services.episodic_memory_service import EpisodicMemoryService
from services.semantic_memory_service import SemanticMemoryService

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def vector_store():
    """Provide vector store instance."""
    store = VectorStore()
    store.connect()
    yield store
    store.disconnect()

@pytest.fixture
async def cache_store():
    """Provide cache store instance."""
    store = CacheStore()
    await store.connect()
    yield store
    await store.disconnect()

@pytest.fixture
def graph_store():
    """Provide graph store instance."""
    store = GraphStore()
    store.connect()
    yield store
    # Cleanup test data
    store.disconnect()

@pytest.fixture
def embedder():
    """Provide text embedder."""
    embedder = TextEmbedder()
    embedder.load_model()
    return embedder

@pytest.fixture
def embedding_cache():
    """Provide embedding cache."""
    return EmbeddingCache(max_size=100)

@pytest.fixture
def episodic_service(vector_store, embedder, embedding_cache):
    """Provide episodic memory service."""
    return EpisodicMemoryService(vector_store, embedder, embedding_cache)

@pytest.fixture
def semantic_service(vector_store, embedder, embedding_cache):
    """Provide semantic memory service."""
    return SemanticMemoryService(vector_store, embedder, embedding_cache)

@pytest.fixture
def sample_memory():
    """Provide sample memory for testing."""
    from models.memory import Memory, MemoryType, MemoryMetadata
    return Memory(
        type=MemoryType.SEMANTIC,
        content="Test memory content",
        metadata=MemoryMetadata(tags=["test"], confidence=0.9),
        collection="test"
    )
```

---

## 📊 Test Execution Plan

### Phase 1: Unit Tests (Week 1)
```bash
# Run all unit tests
pytest tests/unit/ -v --cov=. --cov-report=html

# Target: 70% coverage
```

**Priority:**
1. ✅ Models validation
2. ✅ Embedding generation
3. ✅ Storage operations
4. ✅ Service layer logic
5. ✅ Consolidation algorithms

### Phase 2: Integration Tests (Week 2)
```bash
# Run integration tests
pytest tests/integration/ -v --cov-append

# Target: 85% combined coverage
```

**Priority:**
1. ✅ API endpoints
2. ✅ MCP tools
3. ✅ Multi-storage workflows
4. ✅ Service interactions

### Phase 3: E2E Tests (Week 3)
```bash
# Run E2E tests
pytest tests/e2e/ -v -s

# Focus on critical paths
```

**Scenarios:**
1. ✅ Full memory lifecycle
2. ✅ Knowledge graph building
3. ✅ Search workflows
4. ✅ Consolidation pipeline

### Phase 4: Performance Tests (Ongoing)
```bash
# Run performance benchmarks
pytest tests/performance/ -v --benchmark-only

# Monitor regressions
```

**Metrics:**
- ✅ Search latency (p50, p95, p99)
- ✅ Throughput (QPS)
- ✅ Concurrent operations
- ✅ Memory usage

---

## 🎯 Success Criteria

### Code Coverage
- ✅ **Overall**: > 80%
- ✅ **Critical Paths**: > 90%
- ✅ **Services**: > 85%
- ✅ **Storage**: > 85%

### Performance
- ✅ **Vector Search**: < 50ms (p95)
- ✅ **Hybrid Search**: < 100ms (p95)
- ✅ **Graph Query**: < 150ms (p95)
- ✅ **Write Throughput**: > 100 QPS

### Reliability
- ✅ **Test Pass Rate**: > 95%
- ✅ **Flaky Tests**: < 2%
- ✅ **CI/CD Integration**: All tests automated

---

## 🔄 Continuous Testing

### Pre-Commit Hooks
```bash
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: pytest-unit
      name: Run unit tests
      entry: pytest tests/unit/ -v
      language: system
      pass_filenames: false
```

### CI/CD Pipeline
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v --cov --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 📝 Test Documentation

Each test should include:
1. ✅ **Clear docstring** explaining purpose
2. ✅ **Arrange-Act-Assert** structure
3. ✅ **Meaningful assertions**
4. ✅ **Edge case coverage**
5. ✅ **Performance expectations** (if applicable)

---

## 🚨 Known Test Challenges

1. **Milvus Dependency**: Requires running instance
   - **Solution**: Use Docker Compose for test environment

2. **Neo4j State**: Graph accumulates test data
   - **Solution**: Clear database in teardown

3. **Embedding Model**: Large model download
   - **Solution**: Cache model in CI/CD, use mock embeddings for unit tests

4. **Async Tests**: Complexity with asyncio
   - **Solution**: Use pytest-asyncio plugin

---

This focused testing plan covers the critical functionality while remaining practical and executable. Start with Phase 1 unit tests and progressively build coverage!
