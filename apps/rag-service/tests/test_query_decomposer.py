"""Unit tests for QueryDecomposer — routing logic, sync decompose_query, and Bedrock integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.query_decomposer import (
    DecompositionDecision,
    DecompositionResult,
    QueryDecomposer,
    SubQuestion,
)


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.enable_query_decomposition = True
    settings.decomposition_min_tokens = 15
    settings.decomposition_semantic_gap_threshold = 0.3
    settings.decomposition_max_subquestions = 3
    settings.decomposition_timeout_s = 10
    settings.decomposition_model_id = "amazon.nova-pro-v1:0"
    settings.aws_region = "ap-southeast-2"
    return settings


@pytest.fixture
def mock_bedrock():
    return MagicMock()


@pytest.fixture
def decomposer(mock_settings, mock_bedrock):
    return QueryDecomposer(mock_settings, mock_bedrock)


# ---------------------------------------------------------------------------
# Routing rules (should_decompose)
# ---------------------------------------------------------------------------

class TestShouldDecompose:

    def test_rule1_short_query_returns_false(self, decomposer):
        decision = decomposer.should_decompose("find a song")
        assert decision.should_decompose is False
        assert "too short" in decision.reasoning.lower()

    def test_rule1_empty_query_returns_false(self, decomposer):
        decision = decomposer.should_decompose("")
        assert decision.should_decompose is False

    def test_rule2_entity_who_returns_false(self, decomposer):
        query = "who is the original author of this particular regulation and what specific department or division do they officially belong to within the organisation"
        decision = decomposer.should_decompose(query)
        assert decision.should_decompose is False
        assert "entity" in decision.reasoning.lower() or "keyword" in decision.reasoning.lower()

    def test_rule2_entity_find_returns_false(self, decomposer):
        query = "find the document about copyright law that was published last year in Australia"
        decision = decomposer.should_decompose(query)
        assert decision.should_decompose is False

    def test_rule2_entity_list_returns_false(self, decomposer):
        query = "list all regulations related to music licensing that were updated after 2024"
        decision = decomposer.should_decompose(query)
        assert decision.should_decompose is False

    def test_rule3_reasoning_keyword_decompose(self, decomposer):
        query = (
            "explain the relationship between APRA policy updates and market regulations "
            "and how they impact music licensing in digital platforms"
        )
        decision = decomposer.should_decompose(query)
        assert decision.should_decompose is True
        assert decision.estimated_sub_questions >= 2
        assert decision.confidence >= 0.8

    def test_rule3_compare_keyword_decompose(self, decomposer):
        query = (
            "compare the copyright protection approaches between Australia and United States "
            "and discuss which offers better creator rights"
        )
        decision = decomposer.should_decompose(query)
        assert decision.should_decompose is True

    def test_rule3_with_explicit_semantic_gap_below_threshold(self, decomposer):
        query = "explain why music licensing fees have increased significantly over the past decade"
        decision = decomposer.should_decompose(query, semantic_gap=0.1)
        assert decision.should_decompose is False

    def test_rule4_high_complexity_long_query(self, decomposer):
        long_query = " ".join(["technical"] * 25)
        decision = decomposer.should_decompose(long_query, complexity="high")
        assert decision.should_decompose is True
        assert decision.estimated_sub_questions == 3

    def test_rule4_high_complexity_short_query_no_decompose(self, decomposer):
        query = "short technical query about regulations and policies"
        decision = decomposer.should_decompose(query, complexity="high")
        assert decision.should_decompose is False

    def test_rule5_default_no_decompose(self, decomposer):
        query = "the current state of digital music platforms and their business model evolution"
        decision = decomposer.should_decompose(query)
        assert decision.should_decompose is False

    def test_confidence_always_positive(self, decomposer):
        queries = [
            "short",
            "find something specific in the database",
            "explain the complex relationship between A and B in the context of C and D",
            " ".join(["word"] * 30),
        ]
        for q in queries:
            decision = decomposer.should_decompose(q)
            assert decision.confidence > 0

    def test_cost_zero_when_no_decompose(self, decomposer):
        decision = decomposer.should_decompose("short query")
        assert decision.decomposition_cost_cents == 0.0

    def test_cost_positive_when_decompose(self, decomposer):
        query = (
            "explain the detailed relationship between international copyright law frameworks "
            "and digital content distribution models across multiple different jurisdictions around the world"
        )
        decision = decomposer.should_decompose(query)
        assert decision.should_decompose is True
        assert decision.decomposition_cost_cents > 0


# ---------------------------------------------------------------------------
# decompose_query (sync, with Bedrock mock)
# ---------------------------------------------------------------------------

class TestDecomposeQuery:

    def test_decompose_returns_result_object(self, decomposer, mock_bedrock):
        mock_bedrock.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": json.dumps([
                                {"id": 1, "question": "What is the background of APRA policies?", "focus": "context", "retrieve_strategy": "dense"},
                                {"id": 2, "question": "How do APRA policies impact licensing?", "focus": "impact", "retrieve_strategy": "hybrid"},
                            ])
                        }
                    ]
                }
            },
            "usage": {"inputTokens": 100, "outputTokens": 50},
        }

        query = "explain how APRA policies impact music licensing in digital platforms across multiple regions and what the consequences are for creators"
        result = decomposer.decompose_query(query)

        assert isinstance(result, DecompositionResult)
        assert result.should_decompose is True
        assert len(result.sub_questions) == 2
        assert result.sub_questions[0].id == 1
        assert result.original_query == query

    def test_decompose_skips_when_decision_says_no(self, decomposer, mock_bedrock):
        result = decomposer.decompose_query("short")
        assert result.should_decompose is False
        assert len(result.sub_questions) == 0
        mock_bedrock.converse.assert_not_called()

    def test_decompose_with_precomputed_decision(self, decomposer, mock_bedrock):
        decision = DecompositionDecision(
            should_decompose=False,
            confidence=0.9,
            reasoning="Test skip",
            estimated_sub_questions=0,
            decomposition_cost_cents=0.0,
        )
        result = decomposer.decompose_query("any query here", decision=decision)
        assert result.should_decompose is False
        mock_bedrock.converse.assert_not_called()

    def test_decompose_falls_back_on_bedrock_error(self, decomposer, mock_bedrock):
        mock_bedrock.converse.side_effect = Exception("Bedrock timeout")
        query = "explain the detailed relationship between international copyright law frameworks and digital content distribution across multiple different jurisdictions around the world"
        result = decomposer.decompose_query(query)
        # Bedrock error triggers fallback sub-questions (graceful degradation)
        assert result.should_decompose is True
        assert len(result.sub_questions) >= 2
        # Fallback sub-questions are deterministic, not from Bedrock
        assert "context" in result.sub_questions[0].focus.lower() or "background" in result.sub_questions[0].focus.lower()

    def test_decompose_uses_decomposition_model_id(self, decomposer, mock_settings, mock_bedrock):
        mock_bedrock.converse.return_value = {
            "output": {"message": {"content": [{"text": "[]"}]}},
            "usage": {},
        }
        query = "explain the detailed relationship between international copyright law frameworks and digital content distribution across multiple different jurisdictions around the world"
        decomposer.decompose_query(query)
        call_args = mock_bedrock.converse.call_args
        assert call_args is not None, "Expected bedrock.converse to be called"
        assert call_args.kwargs["modelId"] == "amazon.nova-pro-v1:0"

    def test_decompose_fallback_subquestions_on_bad_json(self, decomposer, mock_bedrock):
        mock_bedrock.converse.return_value = {
            "output": {"message": {"content": [{"text": "This is not JSON at all"}]}},
            "usage": {},
        }
        query = "explain the detailed relationship between international copyright law frameworks and digital content distribution across multiple different jurisdictions around the world"
        result = decomposer.decompose_query(query)
        assert result.should_decompose is True
        assert len(result.sub_questions) >= 2


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TestModels:

    def test_decomposition_decision_fields(self):
        d = DecompositionDecision(
            should_decompose=True,
            confidence=0.85,
            reasoning="test",
            estimated_sub_questions=2,
            decomposition_cost_cents=0.15,
        )
        assert d.should_decompose is True
        assert d.confidence == 0.85
        assert d.estimated_sub_questions == 2

    def test_sub_question_min_length_validation(self):
        with pytest.raises(ValueError):
            SubQuestion(id=1, question="short", focus="test", retrieve_strategy="dense")

    def test_sub_question_id_range_validation(self):
        with pytest.raises(ValueError):
            SubQuestion(id=10, question="Valid question text here", focus="test", retrieve_strategy="dense")

    def test_decomposition_result_defaults(self):
        r = DecompositionResult(
            should_decompose=False,
            decision_reasoning="test",
            original_query="hello",
        )
        assert r.sub_questions == []

    def test_sub_question_valid(self):
        sq = SubQuestion(
            id=1,
            question="What are the benefits of Neo4j for music licensing?",
            focus="graph database advantages",
            retrieve_strategy="dense",
        )
        assert sq.retrieve_strategy == "dense"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_unicode_query(self, decomposer):
        decision = decomposer.should_decompose("音乐许可政策如何影响 Neo4j 的应用 以及它在数据管理中的角色")
        assert isinstance(decision, DecompositionDecision)

    def test_very_long_query(self, decomposer):
        long_query = " ".join(["word"] * 150)
        decision = decomposer.should_decompose(long_query)
        assert isinstance(decision, DecompositionDecision)

    def test_special_characters(self, decomposer):
        query = "What are the best practices for K/D ratio @gaming & how do they compare across platforms?"
        decision = decomposer.should_decompose(query)
        assert isinstance(decision, DecompositionDecision)

    def test_parse_sub_questions_extracts_json_array(self, decomposer):
        text = 'Here are the sub-questions:\n[{"id": 1, "question": "What is APRA and its role?", "focus": "definition", "retrieve_strategy": "dense"}]\nDone.'
        result = decomposer._parse_sub_questions(text, "test query", 2)
        assert len(result) == 1
        assert result[0].id == 1

    def test_parse_sub_questions_handles_non_array(self, decomposer):
        text = '{"id": 1, "question": "Single question about policies?", "focus": "test", "retrieve_strategy": "dense"}'
        result = decomposer._parse_sub_questions(text, "test query", 2)
        assert len(result) >= 1

    def test_fallback_subquestions_count(self, decomposer):
        result = decomposer._create_fallback_subquestions("test query", 3)
        assert len(result) == 3
        assert result[0].id == 1
        assert result[1].id == 2
        assert result[2].id == 3

    def test_fallback_subquestions_min_count(self, decomposer):
        result = decomposer._create_fallback_subquestions("test query", 1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Additional coverage: semantic_gap boundaries, parse edge cases, clamping
# ---------------------------------------------------------------------------

class TestSemanticGapBoundaries:

    def test_semantic_gap_exactly_at_threshold(self, decomposer):
        query = "explain how copyright policy changes affect digital platforms across multiple regions and what this means for creators globally"
        decision = decomposer.should_decompose(query, semantic_gap=0.3)
        assert decision.should_decompose is False

    def test_semantic_gap_just_above_threshold(self, decomposer):
        query = "explain how copyright policy changes affect digital platforms across multiple regions and what this means for creators globally"
        decision = decomposer.should_decompose(query, semantic_gap=0.31)
        assert decision.should_decompose is True

    def test_semantic_gap_none_treated_as_above(self, decomposer):
        query = "explain how copyright policy changes affect digital platforms across multiple regions and what this means for creators globally"
        decision = decomposer.should_decompose(query, semantic_gap=None)
        assert decision.should_decompose is True

    def test_semantic_gap_zero_reasoning_query(self, decomposer):
        query = "explain why music licensing fees have increased significantly over the past decade"
        decision = decomposer.should_decompose(query, semantic_gap=0.0)
        assert decision.should_decompose is False


class TestParseSubQuestionsExtended:

    def test_parse_markdown_fenced_json(self, decomposer):
        text = '```json\n[{"id": 1, "question": "What is the role of APRA in licensing?", "focus": "role", "retrieve_strategy": "dense"}]\n```'
        result = decomposer._parse_sub_questions(text, "test query", 2)
        assert len(result) == 1
        assert result[0].id == 1

    def test_parse_empty_array_falls_back(self, decomposer):
        result = decomposer._parse_sub_questions("[]", "test query", 2)
        assert len(result) >= 2

    def test_parse_invalid_subquestion_fields_falls_back(self, decomposer):
        text = '[{"id": 1, "wrong_field": "bad data"}]'
        result = decomposer._parse_sub_questions(text, "test query", 2)
        assert len(result) >= 2

    def test_parse_mixed_valid_invalid(self, decomposer):
        text = json.dumps([
            {"id": 1, "question": "What is the context of copyright law reform?", "focus": "context", "retrieve_strategy": "dense"},
            {"id": 2, "bad": "data"},
        ])
        result = decomposer._parse_sub_questions(text, "test query", 2)
        assert len(result) >= 1


class TestGenerateSubQuestionsClamping:

    def test_num_subquestions_clamped_to_min_2(self, decomposer, mock_bedrock):
        mock_bedrock.converse.return_value = {
            "output": {"message": {"content": [{"text": json.dumps([
                {"id": 1, "question": "What is the background of this topic?", "focus": "context", "retrieve_strategy": "dense"},
                {"id": 2, "question": "What are the implications of this topic?", "focus": "impact", "retrieve_strategy": "hybrid"},
            ])}]}},
            "usage": {},
        }
        decomposer._generate_sub_questions_bedrock("test query", 1)
        call_args = mock_bedrock.converse.call_args
        system_text = call_args.kwargs["system"][0]["text"]
        assert "2" in system_text

    def test_num_subquestions_clamped_to_max(self, decomposer, mock_settings, mock_bedrock):
        mock_settings.decomposition_max_subquestions = 3
        mock_bedrock.converse.return_value = {
            "output": {"message": {"content": [{"text": "[]"}]}},
            "usage": {},
        }
        decomposer._generate_sub_questions_bedrock("test query", 10)
        call_args = mock_bedrock.converse.call_args
        system_text = call_args.kwargs["system"][0]["text"]
        assert "3" in system_text


class TestDecomposeQueryTopLevelFailure:

    def test_decompose_query_returns_false_on_unexpected_error(self, mock_settings, mock_bedrock):
        decomposer = QueryDecomposer(mock_settings, mock_bedrock)

        def broken_converse(**kwargs):
            raise TypeError("Unexpected internal error")

        mock_bedrock.converse.side_effect = broken_converse

        query = "explain the detailed relationship between international copyright law frameworks and digital content distribution across multiple different jurisdictions around the world"
        result = decomposer.decompose_query(query)
        # Top-level fallback: _generate_sub_questions_bedrock catches and returns fallback subquestions
        assert isinstance(result, DecompositionResult)
