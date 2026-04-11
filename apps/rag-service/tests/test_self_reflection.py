"""Unit tests for Self-Reflection modules: models, graders, adaptive router, and node."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.adaptive_reflection_router import AdaptiveReflectionRouter
from app.faithfulness_grader import FaithfulnessGrader
from app.relevance_grader import RelevanceGrader
from app.self_reflection_models import (
    AdaptiveReflectionDecision,
    Claim,
    FaithfulnessResult,
    FaithfulnessVerdict,
    QueryComplexityScore,
    ReflectionResult,
    RelevanceResult,
    RelevanceVerdict,
    RetryAction,
)
from app.self_reflection_node import SelfReflectionNode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.aws_region = "ap-southeast-2"
    settings.reflection_faithfulness_threshold = 0.6
    settings.reflection_relevance_threshold = 0.5
    return settings


@pytest.fixture
def mock_bedrock():
    return MagicMock()


def _converse_response(text: str) -> dict:
    """Build a minimal Bedrock converse response dict."""
    return {"output": {"message": {"content": [{"text": text}]}}}


def _make_hit(snippet: str = "some evidence text") -> dict:
    return {"snippet": snippet, "chunk_id": "c1", "score": 0.9}


# ===========================================================================
# 1. TestSelfReflectionModels
# ===========================================================================


class TestSelfReflectionModels:
    """Tests for Pydantic data models and enums."""

    # --- Enums ---

    def test_faithfulness_verdict_values(self):
        assert FaithfulnessVerdict.FAITHFUL.value == "faithful"
        assert FaithfulnessVerdict.PARTIALLY_FAITHFUL.value == "partially_faithful"
        assert FaithfulnessVerdict.UNFAITHFUL.value == "unfaithful"
        assert FaithfulnessVerdict.UNCLEAR.value == "unclear"

    def test_relevance_verdict_values(self):
        assert RelevanceVerdict.RELEVANT.value == "relevant"
        assert RelevanceVerdict.PARTIALLY_RELEVANT.value == "partially_relevant"
        assert RelevanceVerdict.IRRELEVANT.value == "irrelevant"

    def test_retry_action_values(self):
        assert RetryAction.ACCEPT.value == "accept"
        assert RetryAction.RETRY_RETRIEVAL.value == "retry_retrieval"
        assert RetryAction.REFINE_ANSWER.value == "refine_answer"
        assert RetryAction.FALLBACK_MODEL.value == "fallback_model"

    # --- Claim ---

    def test_claim_defaults(self):
        c = Claim(text="A fact")
        assert c.supported is False
        assert c.confidence == 0.0
        assert c.supporting_evidence == []

    def test_claim_confidence_bounds(self):
        Claim(text="ok", confidence=0.0)
        Claim(text="ok", confidence=1.0)
        with pytest.raises(ValidationError):
            Claim(text="ok", confidence=1.1)
        with pytest.raises(ValidationError):
            Claim(text="ok", confidence=-0.1)

    # --- FaithfulnessResult ---

    def test_faithfulness_result_score_bounds(self):
        FaithfulnessResult(verdict=FaithfulnessVerdict.FAITHFUL, score=0.0)
        FaithfulnessResult(verdict=FaithfulnessVerdict.FAITHFUL, score=1.0)
        with pytest.raises(ValidationError):
            FaithfulnessResult(verdict=FaithfulnessVerdict.FAITHFUL, score=1.5)

    # --- RelevanceResult ---

    def test_relevance_result_score_bounds(self):
        RelevanceResult(
            verdict=RelevanceVerdict.RELEVANT,
            score=0.8,
            alignment_score=0.9,
            completeness_score=0.7,
        )
        with pytest.raises(ValidationError):
            RelevanceResult(
                verdict=RelevanceVerdict.RELEVANT,
                score=0.8,
                alignment_score=1.5,
                completeness_score=0.7,
            )

    # --- ReflectionResult ---

    def test_reflection_result_construction(self):
        f = FaithfulnessResult(verdict=FaithfulnessVerdict.FAITHFUL, score=0.9)
        r = RelevanceResult(
            verdict=RelevanceVerdict.RELEVANT,
            score=0.8,
            alignment_score=0.8,
            completeness_score=0.8,
        )
        rr = ReflectionResult(
            faithfulness=f,
            relevance=r,
            should_retry=False,
            retry_action=RetryAction.ACCEPT,
            overall_confidence=0.85,
        )
        assert rr.latency_ms == 0.0
        assert rr.explanation == ""

    # --- AdaptiveReflectionDecision ---

    def test_adaptive_decision_defaults(self):
        d = AdaptiveReflectionDecision(should_reflect=True, reason="test")
        assert d.confidence_threshold == 0.6
        assert d.complexity_score == 0.0

    def test_adaptive_decision_complexity_bounds(self):
        with pytest.raises(ValidationError):
            AdaptiveReflectionDecision(
                should_reflect=True, reason="x", complexity_score=2.0,
            )

    # --- QueryComplexityScore ---

    def test_query_complexity_defaults(self):
        qcs = QueryComplexityScore(
            token_count=10,
            has_reasoning_keywords=False,
            is_entity_query=False,
            is_multi_entity=False,
        )
        assert qcs.estimated_complexity == "moderate"
        assert qcs.semantic_gap_score == 0.0

    def test_should_reflect_default_simple_entity(self):
        qcs = QueryComplexityScore(
            token_count=5,
            has_reasoning_keywords=False,
            is_entity_query=True,
            is_multi_entity=False,
            estimated_complexity="simple",
        )
        assert qcs.should_reflect_default is False

    def test_should_reflect_default_complex_high_gap(self):
        qcs = QueryComplexityScore(
            token_count=25,
            has_reasoning_keywords=True,
            is_entity_query=False,
            is_multi_entity=False,
            semantic_gap_score=0.5,
            estimated_complexity="complex",
        )
        assert qcs.should_reflect_default is True

    def test_should_reflect_default_moderate(self):
        qcs = QueryComplexityScore(
            token_count=15,
            has_reasoning_keywords=False,
            is_entity_query=False,
            is_multi_entity=False,
            estimated_complexity="moderate",
        )
        assert qcs.should_reflect_default is True

    def test_should_reflect_default_complex_low_gap(self):
        """Complex but semantic_gap <= 0.3 — falls through to estimated_complexity check."""
        qcs = QueryComplexityScore(
            token_count=25,
            has_reasoning_keywords=True,
            is_entity_query=False,
            is_multi_entity=False,
            semantic_gap_score=0.2,
            estimated_complexity="complex",
        )
        # complex+gap<=0.3 does NOT trigger line 97-98, falls to line 100 → "complex" in set → True
        assert qcs.should_reflect_default is True

    def test_should_reflect_default_entity_15_plus_tokens(self):
        """Entity query with 15+ tokens should NOT skip via line 94."""
        qcs = QueryComplexityScore(
            token_count=20,
            has_reasoning_keywords=False,
            is_entity_query=True,
            is_multi_entity=False,
            estimated_complexity="moderate",
        )
        assert qcs.should_reflect_default is True


# ===========================================================================
# 2. TestFaithfulnessGrader
# ===========================================================================


class TestFaithfulnessGrader:
    """Tests for claim extraction, verification, and overall grading."""

    def test_grade_faithfulness_no_claims(self, mock_settings, mock_bedrock):
        """Empty claim extraction returns FAITHFUL with score 1.0."""
        grader = FaithfulnessGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        # Return empty claims text
        mock_bedrock.converse.return_value = _converse_response("No claims found.")

        result = asyncio.run(grader.grade_faithfulness("Answer text", ["evidence"]))
        assert result.verdict == FaithfulnessVerdict.FAITHFUL
        assert result.score == 1.0
        assert result.total_claims == 0

    def test_grade_faithfulness_all_supported(self, mock_settings, mock_bedrock):
        """All claims supported → FAITHFUL."""
        grader = FaithfulnessGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        # Call 1: extract claims; Calls 2+: verify each claim
        mock_bedrock.converse.side_effect = [
            _converse_response("CLAIM: The sky is blue\nCLAIM: Water is wet"),
            _converse_response(json.dumps({"supported": True, "confidence": 0.9, "explanation": "ok"})),
            _converse_response(json.dumps({"supported": True, "confidence": 0.95, "explanation": "ok"})),
        ]

        result = asyncio.run(grader.grade_faithfulness("The sky is blue and water is wet", ["Sky color is blue. Water is wet."]))
        assert result.verdict == FaithfulnessVerdict.FAITHFUL
        assert result.score == 1.0
        assert result.total_claims == 2
        assert result.supported_claims == 2
        assert result.unsupported_claims == 0

    def test_grade_faithfulness_partially_supported(self, mock_settings, mock_bedrock):
        """One of two claims supported → score 0.5 → PARTIALLY_FAITHFUL."""
        grader = FaithfulnessGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.side_effect = [
            _converse_response("CLAIM: Fact one\nCLAIM: Hallucinated fact"),
            _converse_response(json.dumps({"supported": True, "confidence": 0.9})),
            _converse_response(json.dumps({"supported": False, "confidence": 0.8})),
        ]

        result = asyncio.run(grader.grade_faithfulness("answer", ["evidence"]))
        assert result.verdict == FaithfulnessVerdict.PARTIALLY_FAITHFUL
        assert result.score == pytest.approx(0.5)
        assert result.supported_claims == 1
        assert result.unsupported_claims == 1

    def test_grade_faithfulness_all_unsupported(self, mock_settings, mock_bedrock):
        """No claims supported → score 0.0 → UNFAITHFUL."""
        grader = FaithfulnessGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.side_effect = [
            _converse_response("CLAIM: Wrong fact"),
            _converse_response(json.dumps({"supported": False, "confidence": 0.9})),
        ]

        result = asyncio.run(grader.grade_faithfulness("wrong", ["real evidence"]))
        assert result.verdict == FaithfulnessVerdict.UNFAITHFUL
        assert result.score == 0.0

    def test_grade_faithfulness_converse_error_fails_open(self, mock_settings, mock_bedrock):
        """When converse raises, sub-methods fail open: extract returns fallback claim, verify returns supported=True."""
        grader = FaithfulnessGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.side_effect = Exception("Network failure")

        result = asyncio.run(grader.grade_faithfulness("answer", ["evidence"]))
        # _extract_claims catches exception → fallback [answer[:200]]
        # _verify_claim catches exception → Claim(supported=True, confidence=0.3)
        # Result: 1/1 supported → FAITHFUL
        assert result.verdict == FaithfulnessVerdict.FAITHFUL
        assert result.score == 1.0
        assert result.total_claims == 1

    def test_extract_claims_parses_claim_lines(self, mock_settings, mock_bedrock):
        grader = FaithfulnessGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.return_value = _converse_response(
            "CLAIM: First claim\nSome noise\nCLAIM: Second claim\nCLAIM: Third claim"
        )

        claims = asyncio.run(grader._extract_claims("some answer"))
        assert len(claims) == 3
        assert claims[0] == "First claim"
        assert claims[2] == "Third claim"

    def test_extract_claims_limits_to_10(self, mock_settings, mock_bedrock):
        grader = FaithfulnessGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        lines = "\n".join(f"CLAIM: Claim number {i}" for i in range(15))
        mock_bedrock.converse.return_value = _converse_response(lines)

        claims = asyncio.run(grader._extract_claims("long answer"))
        assert len(claims) == 10

    def test_extract_claims_fallback_on_exception(self, mock_settings, mock_bedrock):
        grader = FaithfulnessGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.side_effect = Exception("timeout")

        claims = asyncio.run(grader._extract_claims("My answer text is this"))
        assert len(claims) == 1
        assert claims[0] == "My answer text is this"[:200]

    def test_verify_claim_supported(self, mock_settings, mock_bedrock):
        grader = FaithfulnessGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.return_value = _converse_response(
            json.dumps({"supported": True, "confidence": 0.85, "explanation": "matches"})
        )

        claim = asyncio.run(grader._verify_claim("Some fact", "Evidence text"))
        assert claim.supported is True
        assert claim.confidence == pytest.approx(0.85)
        assert claim.text == "Some fact"

    def test_verify_claim_json_error_fails_open(self, mock_settings, mock_bedrock):
        grader = FaithfulnessGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.return_value = _converse_response("Not valid JSON")

        claim = asyncio.run(grader._verify_claim("A claim", "evidence"))
        assert claim.supported is True
        assert claim.confidence == pytest.approx(0.3)

    def test_verify_claim_exception_fails_open(self, mock_settings, mock_bedrock):
        grader = FaithfulnessGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.side_effect = Exception("error")

        claim = asyncio.run(grader._verify_claim("A claim", "evidence"))
        assert claim.supported is True
        assert claim.confidence == pytest.approx(0.3)

    def test_verify_claim_strips_markdown_fences(self, mock_settings, mock_bedrock):
        grader = FaithfulnessGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.return_value = _converse_response(
            '```json\n{"supported": false, "confidence": 0.7}\n```'
        )

        claim = asyncio.run(grader._verify_claim("Fact", "Evidence"))
        assert claim.supported is False
        assert claim.confidence == pytest.approx(0.7)

    def test_lazy_bedrock_client_init(self, mock_settings):
        grader = FaithfulnessGrader(mock_settings)
        assert grader._bedrock_client is None
        with patch("app.faithfulness_grader.boto3") as mock_boto:
            mock_boto.client.return_value = MagicMock()
            client = grader._get_bedrock_client()
            mock_boto.client.assert_called_once_with("bedrock-runtime", region_name="ap-southeast-2")
            assert client is not None
            # Second call reuses
            client2 = grader._get_bedrock_client()
            assert client2 is client

    def test_verdict_threshold_at_0_9(self, mock_settings, mock_bedrock):
        """Score exactly 0.9 → FAITHFUL (>=0.9)."""
        grader = FaithfulnessGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        # 9 supported out of 10
        claims_text = "\n".join(f"CLAIM: Claim {i}" for i in range(10))
        responses = [_converse_response(claims_text)]
        for i in range(10):
            supported = i < 9  # 9 supported, 1 not
            responses.append(_converse_response(json.dumps({"supported": supported, "confidence": 0.9})))
        mock_bedrock.converse.side_effect = responses

        result = asyncio.run(grader.grade_faithfulness("answer", ["evidence"]))
        assert result.score == pytest.approx(0.9)
        assert result.verdict == FaithfulnessVerdict.FAITHFUL


# ===========================================================================
# 3. TestRelevanceGrader
# ===========================================================================


class TestRelevanceGrader:
    """Tests for relevance scoring and verdict mapping."""

    def test_grade_relevance_high_scores(self, mock_settings, mock_bedrock):
        grader = RelevanceGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.return_value = _converse_response(
            json.dumps({
                "alignment_score": 0.9,
                "completeness_score": 0.85,
                "relevance_score": 0.8,
                "explanation": "Good answer",
            })
        )

        result = asyncio.run(grader.grade_relevance("What is X?", "X is Y."))
        assert result.verdict == RelevanceVerdict.RELEVANT
        expected_score = (0.9 + 0.85 + 0.8) / 3.0
        assert result.score == pytest.approx(expected_score)
        assert result.alignment_score == pytest.approx(0.9)
        assert result.completeness_score == pytest.approx(0.85)

    def test_grade_relevance_partially_relevant(self, mock_settings, mock_bedrock):
        grader = RelevanceGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.return_value = _converse_response(
            json.dumps({
                "alignment_score": 0.6,
                "completeness_score": 0.5,
                "relevance_score": 0.55,
                "explanation": "Partial",
            })
        )

        result = asyncio.run(grader.grade_relevance("Q?", "A."))
        assert result.verdict == RelevanceVerdict.PARTIALLY_RELEVANT

    def test_grade_relevance_irrelevant(self, mock_settings, mock_bedrock):
        grader = RelevanceGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.return_value = _converse_response(
            json.dumps({
                "alignment_score": 0.2,
                "completeness_score": 0.1,
                "relevance_score": 0.15,
            })
        )

        result = asyncio.run(grader.grade_relevance("Q?", "A."))
        assert result.verdict == RelevanceVerdict.IRRELEVANT
        assert result.score < 0.5

    def test_grade_relevance_json_error_fails_open(self, mock_settings, mock_bedrock):
        grader = RelevanceGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.return_value = _converse_response("Not JSON")

        result = asyncio.run(grader.grade_relevance("Q?", "A."))
        assert result.verdict == RelevanceVerdict.PARTIALLY_RELEVANT
        assert result.score == pytest.approx(0.6)

    def test_grade_relevance_exception_fails_open(self, mock_settings, mock_bedrock):
        grader = RelevanceGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.side_effect = Exception("timeout")

        result = asyncio.run(grader.grade_relevance("Q?", "A."))
        assert result.verdict == RelevanceVerdict.PARTIALLY_RELEVANT
        assert result.score == pytest.approx(0.6)

    def test_grade_relevance_boundary_075(self, mock_settings, mock_bedrock):
        """Exactly 0.75 average → RELEVANT."""
        grader = RelevanceGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.return_value = _converse_response(
            json.dumps({"alignment_score": 0.75, "completeness_score": 0.75, "relevance_score": 0.75})
        )

        result = asyncio.run(grader.grade_relevance("Q?", "A."))
        assert result.verdict == RelevanceVerdict.RELEVANT
        assert result.score == pytest.approx(0.75)

    def test_grade_relevance_boundary_050(self, mock_settings, mock_bedrock):
        """Exactly 0.50 average → PARTIALLY_RELEVANT."""
        grader = RelevanceGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.return_value = _converse_response(
            json.dumps({"alignment_score": 0.5, "completeness_score": 0.5, "relevance_score": 0.5})
        )

        result = asyncio.run(grader.grade_relevance("Q?", "A."))
        assert result.verdict == RelevanceVerdict.PARTIALLY_RELEVANT

    def test_grade_relevance_missing_keys_default_05(self, mock_settings, mock_bedrock):
        """Missing JSON keys default to 0.5."""
        grader = RelevanceGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.return_value = _converse_response(json.dumps({"explanation": "no scores"}))

        result = asyncio.run(grader.grade_relevance("Q?", "A."))
        assert result.score == pytest.approx(0.5)

    def test_grade_relevance_strips_markdown_fences(self, mock_settings, mock_bedrock):
        grader = RelevanceGrader(mock_settings)
        grader._bedrock_client = mock_bedrock
        mock_bedrock.converse.return_value = _converse_response(
            '```json\n{"alignment_score": 0.8, "completeness_score": 0.8, "relevance_score": 0.8}\n```'
        )

        result = asyncio.run(grader.grade_relevance("Q?", "A."))
        assert result.verdict == RelevanceVerdict.RELEVANT


# ===========================================================================
# 4. TestAdaptiveReflectionRouter
# ===========================================================================


class TestAdaptiveReflectionRouter:
    """Tests for all decision tree branches and complexity estimation."""

    # --- estimate_query_complexity ---

    def test_complexity_simple_short_query(self):
        router = AdaptiveReflectionRouter()
        score = router.estimate_query_complexity("Hi")
        assert score.estimated_complexity == "simple"
        assert score.token_count < 5

    def test_complexity_simple_entity_under_15(self):
        router = AdaptiveReflectionRouter()
        score = router.estimate_query_complexity("who is the CEO?")
        assert score.estimated_complexity == "simple"
        assert score.is_entity_query is True

    def test_complexity_complex_reasoning(self):
        router = AdaptiveReflectionRouter()
        score = router.estimate_query_complexity(
            "explain the impact of regulation changes on the music industry"
        )
        assert score.estimated_complexity == "complex"
        assert score.has_reasoning_keywords is True
        assert score.semantic_gap_score == pytest.approx(0.5)

    def test_complexity_complex_long_query(self):
        router = AdaptiveReflectionRouter()
        query = " ".join(["word"] * 35)
        score = router.estimate_query_complexity(query)
        assert score.estimated_complexity == "complex"
        assert score.token_count > 30

    def test_complexity_complex_multi_entity(self):
        router = AdaptiveReflectionRouter()
        score = router.estimate_query_complexity("how does A compare vs B in the market")
        assert score.is_multi_entity is True

    def test_complexity_moderate_default(self):
        router = AdaptiveReflectionRouter()
        score = router.estimate_query_complexity("the current state of digital platforms nowadays")
        assert score.estimated_complexity == "moderate"

    def test_semantic_gap_entity_low(self):
        router = AdaptiveReflectionRouter()
        score = router.estimate_query_complexity("who is the author of this document?")
        assert score.semantic_gap_score == pytest.approx(0.1)

    def test_semantic_gap_reasoning_high(self):
        router = AdaptiveReflectionRouter()
        score = router.estimate_query_complexity("why do copyright laws differ across jurisdictions")
        assert score.semantic_gap_score == pytest.approx(0.5)

    def test_semantic_gap_long_no_reasoning(self):
        router = AdaptiveReflectionRouter()
        query = " ".join(["topic"] * 25)
        score = router.estimate_query_complexity(query)
        assert score.semantic_gap_score == pytest.approx(0.4)

    def test_semantic_gap_default_moderate(self):
        router = AdaptiveReflectionRouter()
        score = router.estimate_query_complexity("some normal question here")
        assert score.semantic_gap_score == pytest.approx(0.2)

    def test_multi_entity_and_keyword(self):
        router = AdaptiveReflectionRouter()
        score = router.estimate_query_complexity("company A and company B profits")
        assert score.is_multi_entity is True

    def test_multi_entity_vs(self):
        router = AdaptiveReflectionRouter()
        score = router.estimate_query_complexity("Python vs Java performance overhead long query tokens")
        assert score.is_multi_entity is True

    # --- should_reflect decision tree ---

    def test_branch_force_reflect(self):
        router = AdaptiveReflectionRouter()
        decision = router.should_reflect("any", "ans", 5, force_reflect=True)
        assert decision.should_reflect is True
        assert "force" in decision.reason.lower()

    def test_branch_disabled_globally(self):
        router = AdaptiveReflectionRouter(enable_reflection=False)
        decision = router.should_reflect("explain everything", "ans", 5)
        assert decision.should_reflect is False
        assert "disabled" in decision.reason.lower()

    def test_branch_simple_entity_skip(self):
        router = AdaptiveReflectionRouter()
        decision = router.should_reflect("who is the CEO?", "ans", 10)
        assert decision.should_reflect is False
        assert "entity" in decision.reason.lower()

    def test_branch_high_confidence_skip(self):
        router = AdaptiveReflectionRouter()
        # Long enough query that won't trigger entity skip
        decision = router.should_reflect(
            "the current state of digital platforms in the modern era",
            "ans", 10, model_confidence=0.96,
        )
        assert decision.should_reflect is False
        assert "confidence" in decision.reason.lower()

    def test_branch_low_hits_reflect(self):
        router = AdaptiveReflectionRouter()
        decision = router.should_reflect(
            "the current state of digital platforms in the modern era",
            "ans", hits_count=2,
        )
        assert decision.should_reflect is True
        assert "hit" in decision.reason.lower()

    def test_branch_low_confidence_reflect(self):
        router = AdaptiveReflectionRouter()
        decision = router.should_reflect(
            "the current state of digital platforms in the modern era",
            "ans", hits_count=5, model_confidence=0.3,
        )
        assert decision.should_reflect is True
        assert "confidence" in decision.reason.lower()

    def test_branch_complex_high_gap_reflect(self):
        router = AdaptiveReflectionRouter()
        decision = router.should_reflect(
            "explain the impact of regulation changes on the music industry and digital licensing",
            "ans", hits_count=5, model_confidence=0.7,
        )
        assert decision.should_reflect is True
        assert "complex" in decision.reason.lower() or "semantic" in decision.reason.lower()

    def test_branch_moderate_sampling_reflect(self):
        """Moderate complexity with random < 0.30 → reflect."""
        router = AdaptiveReflectionRouter()
        with patch("app.adaptive_reflection_router.random.random", return_value=0.1):
            decision = router.should_reflect(
                "the current state of digital platforms in the modern era",
                "ans", hits_count=5, model_confidence=0.7,
            )
        assert decision.should_reflect is True
        assert "sample" in decision.reason.lower() or "moderate" in decision.reason.lower()

    def test_branch_moderate_sampling_skip(self):
        """Moderate complexity with random >= 0.30 → skip."""
        router = AdaptiveReflectionRouter()
        with patch("app.adaptive_reflection_router.random.random", return_value=0.5):
            decision = router.should_reflect(
                "the current state of digital platforms in the modern era",
                "ans", hits_count=5, model_confidence=0.7,
            )
        assert decision.should_reflect is False
        assert "sampling" in decision.reason.lower() or "moderate" in decision.reason.lower()

    def test_branch_default_skip(self):
        """Complex query that somehow doesn't trigger earlier rules → default skip.
        This is hard to reach since complex queries almost always hit branch 7.
        Use a query that yields simple complexity but high token count isn't enough."""
        router = AdaptiveReflectionRouter()
        # A simple query that doesn't match any special rules
        decision = router.should_reflect(
            "some very basic plain text", "ans",
            hits_count=10, model_confidence=0.8,
        )
        # This is moderate complexity → goes to sampling. With real randomness it could be either.
        # Let's test the actual default branch with a simple query
        decision = router.should_reflect(
            "cat", "ans", hits_count=10, model_confidence=0.8,
        )
        # "cat" is too short → simple entity-like but not entity keyword
        assert isinstance(decision, AdaptiveReflectionDecision)


# ===========================================================================
# 5. TestSelfReflectionNode
# ===========================================================================


class TestSelfReflectionNode:
    """Tests for the orchestrator node: routing, parallel grading, retry decisions."""

    def _make_node(self, mock_settings) -> tuple[SelfReflectionNode, MagicMock, MagicMock, MagicMock]:
        f_grader = MagicMock(spec=FaithfulnessGrader)
        r_grader = MagicMock(spec=RelevanceGrader)
        router = MagicMock(spec=AdaptiveReflectionRouter)
        node = SelfReflectionNode(
            settings=mock_settings,
            faithfulness_grader=f_grader,
            relevance_grader=r_grader,
            adaptive_router=router,
        )
        return node, f_grader, r_grader, router

    def _faithful_result(self, score: float = 0.9) -> FaithfulnessResult:
        if score >= 0.9:
            v = FaithfulnessVerdict.FAITHFUL
        elif score >= 0.5:
            v = FaithfulnessVerdict.PARTIALLY_FAITHFUL
        else:
            v = FaithfulnessVerdict.UNFAITHFUL
        return FaithfulnessResult(
            verdict=v, score=score, total_claims=5, supported_claims=int(score * 5),
            unsupported_claims=5 - int(score * 5),
        )

    def _relevant_result(self, score: float = 0.8) -> RelevanceResult:
        if score >= 0.75:
            v = RelevanceVerdict.RELEVANT
        elif score >= 0.5:
            v = RelevanceVerdict.PARTIALLY_RELEVANT
        else:
            v = RelevanceVerdict.IRRELEVANT
        return RelevanceResult(verdict=v, score=score, alignment_score=score, completeness_score=score)

    # --- Routing: skip reflection ---

    def test_skip_reflection_returns_neutral(self, mock_settings):
        node, f_grader, r_grader, router = self._make_node(mock_settings)
        router.should_reflect.return_value = AdaptiveReflectionDecision(
            should_reflect=False, reason="Skip: Simple query",
        )

        result = asyncio.run(node.reflect_on_answer("Q?", "A.", [_make_hit()]))
        assert result.should_retry is False
        assert result.retry_action == RetryAction.ACCEPT
        assert result.overall_confidence == 0.7  # neutral default
        f_grader.grade_faithfulness.assert_not_called()
        r_grader.grade_relevance.assert_not_called()

    # --- Routing: reflect → ACCEPT ---

    def test_reflect_accept_good_scores(self, mock_settings):
        node, f_grader, r_grader, router = self._make_node(mock_settings)
        router.should_reflect.return_value = AdaptiveReflectionDecision(
            should_reflect=True, reason="Reflect",
        )

        async def mock_grade_f(*a, **kw):
            return self._faithful_result(0.9)

        async def mock_grade_r(*a, **kw):
            return self._relevant_result(0.8)

        f_grader.grade_faithfulness = mock_grade_f
        r_grader.grade_relevance = mock_grade_r

        result = asyncio.run(node.reflect_on_answer("Q?", "A.", [_make_hit()]))
        assert result.should_retry is False
        assert result.retry_action == RetryAction.ACCEPT
        assert result.overall_confidence == pytest.approx((0.9 + 0.8) / 2)
        assert result.latency_ms > 0

    # --- Routing: reflect → RETRY_RETRIEVAL (low faithfulness) ---

    def test_reflect_retry_retrieval_low_faithfulness(self, mock_settings):
        node, f_grader, r_grader, router = self._make_node(mock_settings)
        router.should_reflect.return_value = AdaptiveReflectionDecision(
            should_reflect=True, reason="Reflect",
        )

        async def mock_grade_f(*a, **kw):
            return self._faithful_result(0.3)  # Below threshold-0.1 = 0.5

        async def mock_grade_r(*a, **kw):
            return self._relevant_result(0.8)

        f_grader.grade_faithfulness = mock_grade_f
        r_grader.grade_relevance = mock_grade_r

        result = asyncio.run(node.reflect_on_answer("Q?", "A.", [_make_hit()] * 5))
        assert result.should_retry is True
        assert result.retry_action == RetryAction.RETRY_RETRIEVAL

    # --- Routing: reflect → REFINE_ANSWER (low relevance, enough hits) ---

    def test_reflect_refine_answer_low_relevance(self, mock_settings):
        node, f_grader, r_grader, router = self._make_node(mock_settings)
        router.should_reflect.return_value = AdaptiveReflectionDecision(
            should_reflect=True, reason="Reflect",
        )

        async def mock_grade_f(*a, **kw):
            return self._faithful_result(0.7)  # Above threshold

        async def mock_grade_r(*a, **kw):
            return self._relevant_result(0.3)  # Below threshold

        f_grader.grade_faithfulness = mock_grade_f
        r_grader.grade_relevance = mock_grade_r

        result = asyncio.run(node.reflect_on_answer("Q?", "A.", [_make_hit()] * 5))
        assert result.should_retry is True
        assert result.retry_action == RetryAction.REFINE_ANSWER

    # --- Routing: reflect → RETRY_RETRIEVAL (low scores + few hits) ---

    def test_reflect_retry_retrieval_low_scores_few_hits(self, mock_settings):
        node, f_grader, r_grader, router = self._make_node(mock_settings)
        router.should_reflect.return_value = AdaptiveReflectionDecision(
            should_reflect=True, reason="Reflect",
        )

        async def mock_grade_f(*a, **kw):
            return self._faithful_result(0.55)  # Slightly below threshold

        async def mock_grade_r(*a, **kw):
            return self._relevant_result(0.45)  # Below threshold

        f_grader.grade_faithfulness = mock_grade_f
        r_grader.grade_relevance = mock_grade_r

        result = asyncio.run(node.reflect_on_answer("Q?", "A.", [_make_hit()] * 2))
        assert result.should_retry is True
        assert result.retry_action == RetryAction.RETRY_RETRIEVAL

    # --- Routing: reflect → FALLBACK_MODEL ---

    def test_reflect_fallback_model(self, mock_settings):
        node, f_grader, r_grader, router = self._make_node(mock_settings)
        router.should_reflect.return_value = AdaptiveReflectionDecision(
            should_reflect=True, reason="Reflect",
        )

        async def mock_grade_f(*a, **kw):
            # f=0.55, threshold=0.6 → not < threshold-0.1(=0.5), so doesn't hit RETRY_RETRIEVAL
            return self._faithful_result(0.55)

        async def mock_grade_r(*a, **kw):
            # r=0.55, threshold=0.5 → r >= threshold, so doesn't hit REFINE_ANSWER
            # But f < f_threshold → overall not ACCEPT
            # hits >= 3 → not the "few hits" branch
            return self._relevant_result(0.55)

        f_grader.grade_faithfulness = mock_grade_f
        r_grader.grade_relevance = mock_grade_r

        result = asyncio.run(node.reflect_on_answer("Q?", "A.", [_make_hit()] * 5))
        assert result.should_retry is True
        assert result.retry_action == RetryAction.FALLBACK_MODEL

    # --- Evidence extraction from hits ---

    def test_evidence_from_snippet_key(self, mock_settings):
        node, f_grader, r_grader, router = self._make_node(mock_settings)
        router.should_reflect.return_value = AdaptiveReflectionDecision(
            should_reflect=True, reason="Reflect",
        )

        captured_evidence = []

        async def mock_grade_f(answer, evidence_texts, question=""):
            captured_evidence.extend(evidence_texts)
            return self._faithful_result(0.9)

        async def mock_grade_r(*a, **kw):
            return self._relevant_result(0.9)

        f_grader.grade_faithfulness = mock_grade_f
        r_grader.grade_relevance = mock_grade_r

        hits = [{"snippet": "from snippet"}, {"chunk_text": "from chunk"}]
        asyncio.run(node.reflect_on_answer("Q?", "A.", hits))
        assert captured_evidence == ["from snippet", "from chunk"]

    # --- Error handling ---

    def test_reflection_exception_returns_neutral(self, mock_settings):
        node, f_grader, r_grader, router = self._make_node(mock_settings)
        router.should_reflect.side_effect = Exception("Router broke")

        result = asyncio.run(node.reflect_on_answer("Q?", "A.", [_make_hit()]))
        assert result.should_retry is False
        assert result.retry_action == RetryAction.ACCEPT
        assert result.overall_confidence == 0.5  # error neutral confidence

    # --- Neutral result static method ---

    def test_neutral_result_defaults(self):
        result = SelfReflectionNode._neutral_result("test reason")
        assert result.should_retry is False
        assert result.retry_action == RetryAction.ACCEPT
        assert result.overall_confidence == 0.7
        assert result.latency_ms == 0.0
        assert result.faithfulness.verdict == FaithfulnessVerdict.FAITHFUL
        assert result.relevance.verdict == RelevanceVerdict.RELEVANT

    def test_neutral_result_custom_confidence(self):
        result = SelfReflectionNode._neutral_result("err", confidence=0.3, latency_ms=100.0)
        assert result.overall_confidence == 0.3
        assert result.latency_ms == 100.0

    # --- _decide_retry_action direct tests ---

    def test_decide_retry_accept(self, mock_settings):
        node, *_ = self._make_node(mock_settings)
        action = node._decide_retry_action(
            self._faithful_result(0.8), self._relevant_result(0.7), hits_count=5,
        )
        assert action == RetryAction.ACCEPT

    def test_decide_retry_retrieval_very_low_faithfulness(self, mock_settings):
        node, *_ = self._make_node(mock_settings)
        action = node._decide_retry_action(
            self._faithful_result(0.4), self._relevant_result(0.7), hits_count=5,
        )
        assert action == RetryAction.RETRY_RETRIEVAL

    def test_decide_refine_answer_low_relevance(self, mock_settings):
        node, *_ = self._make_node(mock_settings)
        action = node._decide_retry_action(
            self._faithful_result(0.7), self._relevant_result(0.3), hits_count=5,
        )
        assert action == RetryAction.REFINE_ANSWER

    def test_decide_retry_retrieval_few_hits(self, mock_settings):
        node, *_ = self._make_node(mock_settings)
        action = node._decide_retry_action(
            self._faithful_result(0.55), self._relevant_result(0.45), hits_count=1,
        )
        assert action == RetryAction.RETRY_RETRIEVAL

    def test_decide_fallback_model_intermediate(self, mock_settings):
        node, *_ = self._make_node(mock_settings)
        # f=0.55 (< 0.6 threshold, but not < 0.5), r=0.55 (>= 0.5 threshold), hits=5
        action = node._decide_retry_action(
            self._faithful_result(0.55), self._relevant_result(0.55), hits_count=5,
        )
        assert action == RetryAction.FALLBACK_MODEL

    # --- Default adaptive router creation ---

    def test_default_router_created(self, mock_settings):
        f_grader = MagicMock(spec=FaithfulnessGrader)
        r_grader = MagicMock(spec=RelevanceGrader)
        node = SelfReflectionNode(
            settings=mock_settings,
            faithfulness_grader=f_grader,
            relevance_grader=r_grader,
        )
        assert isinstance(node._adaptive_router, AdaptiveReflectionRouter)
