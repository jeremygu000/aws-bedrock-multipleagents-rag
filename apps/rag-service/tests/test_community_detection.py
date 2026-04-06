"""Unit tests for community detection module.

Tests cover:
- Pydantic models (Community, CommunitySummary) with defaults and validation
- Community prompt building
- CommunityDetector with mocked Neo4j and Leiden
- CommunitySummarizer with mocked Bedrock
- CommunityStore with mocked Neo4j
- Integration with answer_generator._build_community_evidence_block
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.answer_generator import _build_community_evidence_block
from app.community_detection import (
    Community,
    CommunityDetector,
    CommunityStore,
    CommunitySummarizer,
    CommunitySummary,
    _build_community_prompt,
)
from app.config import Settings

# ---------------------------------------------------------------------------
# Fixtures & Factories
# ---------------------------------------------------------------------------


@pytest.fixture
def settings():
    """Create a test Settings object with community detection enabled."""
    return Settings(
        enable_community_detection=True,
        enable_neo4j=True,
        community_resolution=1.0,
        community_max_levels=3,
        community_min_size=2,
        community_summary_model="amazon.nova-pro-v1:0",
        community_summary_max_tokens=2000,
        aws_region="ap-southeast-2",
    )


def _make_community(
    community_id: str = "L0_C0",
    level: int = 0,
    parent_id: str | None = None,
    entity_ids: list[str] | None = None,
    entity_names: list[str] | None = None,
    relation_descriptions: list[str] | None = None,
) -> Community:
    """Factory for creating test Community objects."""
    return Community(
        community_id=community_id,
        level=level,
        parent_id=parent_id,
        entity_ids=entity_ids or ["e1", "e2"],
        entity_names=entity_names or ["Entity One", "Entity Two"],
        relation_descriptions=relation_descriptions or ["e1 -> e2"],
    )


def _make_community_summary(
    community_id: str = "L0_C0",
    level: int = 0,
    title: str = "Test Community",
    rating: float = 7.5,
) -> CommunitySummary:
    """Factory for creating test CommunitySummary objects."""
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    return CommunitySummary(
        community_id=community_id,
        level=level,
        title=title,
        summary="This is a test community summary.",
        findings=["Finding 1", "Finding 2"],
        rating=rating,
        rating_explanation="Test rating explanation.",
        entity_ids=["e1", "e2"],
        entity_count=2,
        updated_at=now_iso,
    )


def _mock_driver():
    """Create a mocked Neo4j driver and session."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


# ---------------------------------------------------------------------------
# Pydantic Model Tests
# ---------------------------------------------------------------------------


class TestCommunityModelDefaults:
    """Test Community Pydantic model defaults and initialization."""

    def test_community_minimal_fields(self):
        """Create Community with only required fields."""
        community = Community(
            community_id="L0_C0",
            level=0,
        )
        assert community.community_id == "L0_C0"
        assert community.level == 0
        assert community.parent_id is None
        assert community.entity_ids == []
        assert community.entity_names == []
        assert community.relation_descriptions == []

    def test_community_with_all_fields(self):
        """Create Community with all fields specified."""
        community = _make_community()
        assert community.community_id == "L0_C0"
        assert community.level == 0
        assert community.parent_id is None
        assert community.entity_ids == ["e1", "e2"]
        assert community.entity_names == ["Entity One", "Entity Two"]
        assert community.relation_descriptions == ["e1 -> e2"]

    def test_community_level_validation(self):
        """Level must be >= 0."""
        with pytest.raises(ValueError):
            Community(
                community_id="L-1_C0",
                level=-1,
            )

    def test_community_hierarchical_parent_child(self):
        """Test parent_id linkage for hierarchy."""
        child = Community(
            community_id="L1_C5",
            level=1,
            parent_id="L0_C0",
        )
        assert child.parent_id == "L0_C0"


class TestCommunitySummaryModelDefaults:
    """Test CommunitySummary Pydantic model defaults and validation."""

    def test_community_summary_minimal_fields(self):
        """Create CommunitySummary with only required fields."""
        summary = CommunitySummary(
            community_id="L0_C0",
            level=0,
            title="Test",
            summary="A test.",
        )
        assert summary.community_id == "L0_C0"
        assert summary.level == 0
        assert summary.title == "Test"
        assert summary.summary == "A test."
        assert summary.findings == []
        assert summary.rating == 0.0
        assert summary.rating_explanation == ""
        assert summary.entity_ids == []
        assert summary.entity_count == 0
        assert summary.updated_at == ""

    def test_community_summary_rating_clamped_low(self):
        """Rating < 0 should clamp to 0."""
        with pytest.raises(ValueError):
            CommunitySummary(
                community_id="L0_C0",
                level=0,
                title="Test",
                summary="Test",
                rating=-1.0,
            )

    def test_community_summary_rating_clamped_high(self):
        """Rating > 10 should clamp to 10."""
        with pytest.raises(ValueError):
            CommunitySummary(
                community_id="L0_C0",
                level=0,
                title="Test",
                summary="Test",
                rating=11.0,
            )

    def test_community_summary_rating_valid_boundary(self):
        """Rating at boundaries should be valid."""
        summary_low = CommunitySummary(
            community_id="L0_C0",
            level=0,
            title="Test",
            summary="Test",
            rating=0.0,
        )
        assert summary_low.rating == 0.0

        summary_high = CommunitySummary(
            community_id="L0_C0",
            level=0,
            title="Test",
            summary="Test",
            rating=10.0,
        )
        assert summary_high.rating == 10.0

    def test_community_summary_mid_range_rating(self):
        """Rating within valid range should work."""
        summary = CommunitySummary(
            community_id="L0_C0",
            level=0,
            title="Test",
            summary="Test",
            rating=5.5,
        )
        assert summary.rating == 5.5


# ---------------------------------------------------------------------------
# _build_community_prompt Tests
# ---------------------------------------------------------------------------


class TestBuildCommunityPrompt:
    """Test _build_community_prompt function."""

    def test_build_community_prompt_contains_entities(self):
        """Prompt should include entity IDs and names."""
        community = _make_community(
            entity_ids=["e1", "e2"],
            entity_names=["Alice", "Bob"],
        )
        prompt = _build_community_prompt(community)
        assert "[e1] Alice" in prompt
        assert "[e2] Bob" in prompt

    def test_build_community_prompt_contains_relations(self):
        """Prompt should include relation descriptions."""
        community = _make_community(
            relation_descriptions=["Alice -> Bob (knows)"],
        )
        prompt = _build_community_prompt(community)
        assert "Alice -> Bob (knows)" in prompt

    def test_build_community_prompt_empty_entities(self):
        """Empty entities should show '(no entities)'."""
        community = Community(
            community_id="L0_C0",
            level=0,
            entity_ids=[],
            entity_names=[],
        )
        prompt = _build_community_prompt(community)
        assert "(no entities)" in prompt

    def test_build_community_prompt_empty_relations(self):
        """Empty relations should show '(no relations)'."""
        community = Community(
            community_id="L0_C0",
            level=0,
            entity_ids=["e1"],
            entity_names=["Alice"],
            relation_descriptions=[],
        )
        prompt = _build_community_prompt(community)
        assert "(no relations)" in prompt

    def test_build_community_prompt_template_structure(self):
        """Prompt should follow GraphRAG template structure."""
        community = _make_community()
        prompt = _build_community_prompt(community)
        assert "---COMMUNITY ENTITIES---" in prompt
        assert "---INTRA-COMMUNITY RELATIONS---" in prompt
        assert "---INSTRUCTIONS---" in prompt


# ---------------------------------------------------------------------------
# CommunityDetector Tests
# ---------------------------------------------------------------------------


class TestCommunityDetectorInit:
    """Test CommunityDetector initialization."""

    def test_detector_stores_repo_and_settings(self):
        """CommunityDetector should store repo and settings."""
        mock_repo = MagicMock()
        settings = Settings(enable_community_detection=True)
        detector = CommunityDetector(mock_repo, settings)
        assert detector._repo is mock_repo
        assert detector._settings is settings


class TestCommunityDetectorRaisesWithoutLeiden:
    """Test CommunityDetector raises when leiden is unavailable."""

    def test_detect_communities_raises_without_leidenalg(self):
        """detect_communities should raise when HAS_LEIDEN=False."""
        mock_repo = MagicMock()
        settings = Settings(enable_community_detection=True)
        detector = CommunityDetector(mock_repo, settings)

        with patch("app.community_detection.HAS_LEIDEN", False):
            with pytest.raises(RuntimeError, match="leidenalg and igraph are required"):
                detector.detect_communities()


class TestCommunityDetectorEmptyGraph:
    """Test CommunityDetector with empty graph."""

    @patch("app.community_detection.HAS_LEIDEN", True)
    def test_detect_communities_no_entities_returns_empty(self):
        """When Neo4j has no entities, return empty list."""
        mock_repo = MagicMock()
        settings = Settings(enable_community_detection=True)
        detector = CommunityDetector(mock_repo, settings)

        # Mock _export_graph to return empty entities
        detector._export_graph = MagicMock(return_value=([], []))

        result = detector.detect_communities()
        assert result == []


class TestCommunityDetectorBuildIgraph:
    """Test _build_igraph method."""

    @patch("app.community_detection.HAS_LEIDEN", True)
    def test_build_igraph_skips_self_loops(self):
        """Self-loop relations (src_idx == tgt_idx) should be skipped."""
        mock_repo = MagicMock()
        settings = Settings(enable_community_detection=True)
        detector = CommunityDetector(mock_repo, settings)

        entities = [
            {"entity_id": "e1", "name": "Alice", "type": "Person"},
            {"entity_id": "e2", "name": "Bob", "type": "Person"},
        ]
        relations = [
            {"source_id": "e1", "target_id": "e1", "weight": 1.0},  # self-loop
            {"source_id": "e1", "target_id": "e2", "weight": 1.0},  # valid
        ]

        with patch("app.community_detection.ig.Graph") as mock_graph_class:
            mock_graph = MagicMock()
            mock_graph_class.return_value = mock_graph
            mock_graph.vcount.return_value = 2
            mock_graph.ecount.return_value = 1

            detector._build_igraph(entities, relations)

            # Should have called add_edges with only 1 edge (the non-self-loop)
            mock_graph.add_edges.assert_called_once()
            edges = mock_graph.add_edges.call_args[0][0]
            assert len(edges) == 1

    @patch("app.community_detection.HAS_LEIDEN", True)
    def test_build_igraph_skips_missing_endpoints(self):
        """Relations referencing non-existent entities should be skipped."""
        mock_repo = MagicMock()
        settings = Settings(enable_community_detection=True)
        detector = CommunityDetector(mock_repo, settings)

        entities = [
            {"entity_id": "e1", "name": "Alice", "type": "Person"},
        ]
        relations = [
            {"source_id": "e1", "target_id": "e99", "weight": 1.0},  # missing endpoint
            {"source_id": "e1", "target_id": "e1", "weight": 1.0},  # self-loop
        ]

        with patch("app.community_detection.ig.Graph") as mock_graph_class:
            mock_graph = MagicMock()
            mock_graph_class.return_value = mock_graph
            mock_graph.vcount.return_value = 1
            mock_graph.ecount.return_value = 0

            detector._build_igraph(entities, relations)

            # Should not call add_edges (no valid edges)
            mock_graph.add_edges.assert_not_called()


# ---------------------------------------------------------------------------
# CommunitySummarizer Tests
# ---------------------------------------------------------------------------


class TestCommunitySummarizerInit:
    """Test CommunitySummarizer initialization."""

    def test_summarizer_stores_settings(self, settings):
        """CommunitySummarizer should store settings."""
        summarizer = CommunitySummarizer(settings)
        assert summarizer._settings is settings
        assert summarizer._bedrock is None


class TestCommunitySummarizerSummarize:
    """Test summarize_community method."""

    @patch("builtins.open", create=True)
    @patch("boto3.client")
    def test_summarize_returns_community_summary(self, mock_boto_client, mock_open, settings):
        """summarize_community should return CommunitySummary."""
        mock_bedrock = MagicMock()
        mock_boto_client.return_value = mock_bedrock

        # Mock successful Bedrock response
        mock_bedrock.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": json.dumps({
                                "title": "Alice & Bob Network",
                                "summary": "A tight-knit pair of collaborators.",
                                "findings": ["They work together", "They are friends"],
                                "rating": 8.5,
                                "rating_explanation": "Significant partnership.",
                            })
                        }
                    ]
                }
            }
        }

        summarizer = CommunitySummarizer(settings)
        community = _make_community()

        result = summarizer.summarize_community(community)

        assert isinstance(result, CommunitySummary)
        assert result.community_id == "L0_C0"
        assert result.title == "Alice & Bob Network"
        assert result.rating == 8.5

    @patch("builtins.open", create=True)
    @patch("boto3.client")
    def test_summarize_fallback_on_invalid_json(self, mock_boto_client, mock_open, settings):
        """When LLM returns non-JSON, fallback summary should be used."""
        mock_bedrock = MagicMock()
        mock_boto_client.return_value = mock_bedrock

        # Mock Bedrock returning invalid JSON
        mock_bedrock.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "This is not JSON at all"}]
                }
            }
        }

        summarizer = CommunitySummarizer(settings)
        community = _make_community()

        result = summarizer.summarize_community(community)

        # Should fall back to minimal summary
        assert isinstance(result, CommunitySummary)
        assert "Community L0_C0" in result.title
        assert result.rating == 0.0


class TestCommunitySummarizerParseResponse:
    """Test _parse_llm_response method."""

    def test_parse_llm_response_valid_json(self, settings):
        """Valid JSON should be parsed correctly."""
        summarizer = CommunitySummarizer(settings)
        community = _make_community()

        json_str = json.dumps({
            "title": "Test Community",
            "summary": "A test community.",
            "findings": ["Finding 1", "Finding 2"],
            "rating": 7.0,
            "rating_explanation": "Good community.",
        })

        result = summarizer._parse_llm_response(json_str, community)

        assert result.title == "Test Community"
        assert result.summary == "A test community."
        assert result.findings == ["Finding 1", "Finding 2"]
        assert result.rating == 7.0

    def test_parse_llm_response_strips_code_fence(self, settings):
        """JSON wrapped in ```json ... ``` should be parsed."""
        summarizer = CommunitySummarizer(settings)
        community = _make_community()

        json_obj = {
            "title": "Fenced Community",
            "summary": "JSON in code fence.",
            "findings": [],
            "rating": 5.0,
            "rating_explanation": "Fenced.",
        }
        json_str = f"```json\n{json.dumps(json_obj)}\n```"

        result = summarizer._parse_llm_response(json_str, community)

        assert result.title == "Fenced Community"
        assert result.rating == 5.0

    def test_parse_llm_response_invalid_json_uses_fallback(self, settings):
        """Invalid JSON should trigger fallback summary."""
        summarizer = CommunitySummarizer(settings)
        community = _make_community()

        invalid_json = "{ this is not valid json }"

        result = summarizer._parse_llm_response(invalid_json, community)

        # Should be fallback summary
        assert "Community L0_C0" in result.title
        assert result.rating == 0.0


# ---------------------------------------------------------------------------
# CommunityStore Tests
# ---------------------------------------------------------------------------


class TestCommunityStoreInit:
    """Test CommunityStore initialization."""

    def test_store_stores_repo_and_settings(self):
        """CommunityStore should store repo and settings."""
        mock_repo = MagicMock()
        settings = Settings(enable_neo4j=True)
        store = CommunityStore(mock_repo, settings)
        assert store._repo is mock_repo
        assert store._settings is settings


class TestCommunityStoreStoreCommunities:
    """Test store_communities method."""

    @patch("neo4j.GraphDatabase.driver")
    def test_store_communities_calls_upsert(self, mock_gd_driver):
        """store_communities should call Cypher upsert for each summary."""
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        call_count = [0]

        def side_effect(work_fn):
            call_count[0] += 1
            tx = MagicMock()
            tx.run.return_value = MagicMock()
            return work_fn(tx)

        mock_session.execute_write.side_effect = side_effect

        mock_repo = MagicMock()
        mock_repo._get_driver.return_value = mock_driver
        mock_repo._database = "neo4j"

        settings = Settings(enable_neo4j=True)
        store = CommunityStore(mock_repo, settings)

        summary = _make_community_summary()
        result = store.store_communities([summary])

        assert result == 1

    def test_store_communities_empty_list_returns_zero(self):
        """Empty input should return 0 with no Neo4j calls."""
        mock_repo = MagicMock()
        settings = Settings(enable_neo4j=True)
        store = CommunityStore(mock_repo, settings)

        result = store.store_communities([])

        assert result == 0


class TestCommunityStoreGetCommunities:
    """Test get_community_summaries method."""

    @patch("neo4j.GraphDatabase.driver")
    def test_get_community_summaries_returns_list(self, mock_gd_driver):
        """get_community_summaries should return list of CommunitySummary."""
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        now_iso = datetime.now(tz=timezone.utc).isoformat()
        row_data = {
            "community_id": "L0_C0",
            "level": 0,
            "parent_id": None,
            "title": "Test Community",
            "summary": "Test summary.",
            "findings": ["Finding 1"],
            "rating": 7.5,
            "rating_explanation": "Good.",
            "entity_count": 5,
            "updated_at": now_iso,
        }

        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            result.__iter__ = lambda self: iter([row_data])
            tx.run.return_value = result
            return work_fn(tx)

        mock_session.execute_read.side_effect = side_effect

        mock_repo = MagicMock()
        mock_repo._get_driver.return_value = mock_driver
        mock_repo._database = "neo4j"

        settings = Settings(enable_neo4j=True)
        store = CommunityStore(mock_repo, settings)

        result = store.get_community_summaries(level=0)

        assert len(result) == 1
        assert isinstance(result[0], CommunitySummary)
        assert result[0].community_id == "L0_C0"
        assert result[0].title == "Test Community"


# ---------------------------------------------------------------------------
# answer_generator integration Tests
# ---------------------------------------------------------------------------


class TestBuildCommunityEvidenceBlock:
    """Test _build_community_evidence_block function from answer_generator."""

    def test_build_community_evidence_block_none(self):
        """None input should return empty string."""
        result = _build_community_evidence_block(None)
        assert result == ""

    def test_build_community_evidence_block_empty_list(self):
        """Empty list should return empty string."""
        result = _build_community_evidence_block([])
        assert result == ""

    def test_build_community_evidence_block_formats_correctly(self):
        """Community data should be formatted with correct structure."""
        communities = [
            {
                "title": "Community 1",
                "summary": "Summary 1.",
                "findings": ["Finding A", "Finding B"],
                "entity_count": 3,
                "rating": 7.5,
            }
        ]

        result = _build_community_evidence_block(communities)

        assert "--- Community [1] ---" in result
        assert "Topic: Community 1" in result
        assert "Summary: Summary 1." in result
        assert "Key Findings: Finding A; Finding B" in result
        assert "Entities: 3 | Rating: 7.5/10" in result

    def test_build_community_evidence_block_multiple_communities(self):
        """Multiple communities should be formatted with correct numbering."""
        communities = [
            {
                "title": "Community A",
                "summary": "Summary A.",
                "findings": ["Finding 1"],
                "entity_count": 2,
                "rating": 5.0,
            },
            {
                "title": "Community B",
                "summary": "Summary B.",
                "findings": ["Finding 2"],
                "entity_count": 4,
                "rating": 8.0,
            },
        ]

        result = _build_community_evidence_block(communities)

        assert "--- Community [1] ---" in result
        assert "--- Community [2] ---" in result
        assert "Community A" in result
        assert "Community B" in result
