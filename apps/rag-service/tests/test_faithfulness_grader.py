from __future__ import annotations

import asyncio
import json
import unittest.mock as mock
from unittest.mock import MagicMock

import pytest

from app.faithfulness_grader import FaithfulnessGrader
from app.self_reflection_models import FaithfulnessVerdict


def _converse_response(text: str) -> dict:
    return {"output": {"message": {"content": [{"text": text}]}}}


def _make_grader() -> tuple[FaithfulnessGrader, MagicMock]:
    settings = MagicMock()
    settings.aws_region = "ap-southeast-2"
    grader = FaithfulnessGrader(settings)
    mock_client = MagicMock()
    grader._bedrock_client = mock_client
    return grader, mock_client


class TestFaithfulnessGrader:

    def test_grade_all_claims_supported(self):
        grader, mock_client = _make_grader()
        mock_client.converse.side_effect = [
            _converse_response("CLAIM: The sky is blue\nCLAIM: Water is wet"),
            _converse_response(json.dumps({"supported": True, "confidence": 0.95, "explanation": "yes"})),
            _converse_response(json.dumps({"supported": True, "confidence": 0.90, "explanation": "yes"})),
        ]

        result = asyncio.run(
            grader.grade_faithfulness(
                "The sky is blue and water is wet",
                ["The sky has a blue color. Water is a wet substance."],
            )
        )

        assert result.verdict == FaithfulnessVerdict.FAITHFUL
        assert result.score == pytest.approx(1.0)
        assert result.total_claims == 2
        assert result.supported_claims == 2
        assert result.unsupported_claims == 0

    def test_grade_no_claims_supported(self):
        grader, mock_client = _make_grader()
        mock_client.converse.side_effect = [
            _converse_response("CLAIM: First hallucination\nCLAIM: Second hallucination"),
            _converse_response(json.dumps({"supported": False, "confidence": 0.9, "explanation": "no"})),
            _converse_response(json.dumps({"supported": False, "confidence": 0.85, "explanation": "no"})),
        ]

        result = asyncio.run(
            grader.grade_faithfulness("Some wrong claim", ["Unrelated evidence text."])
        )

        assert result.verdict == FaithfulnessVerdict.UNFAITHFUL
        assert result.score == pytest.approx(0.0)
        assert result.supported_claims == 0
        assert result.unsupported_claims == 2

    def test_grade_partial_claims(self):
        grader, mock_client = _make_grader()
        mock_client.converse.side_effect = [
            _converse_response("CLAIM: True claim\nCLAIM: Hallucinated claim"),
            _converse_response(json.dumps({"supported": True, "confidence": 0.9, "explanation": "yes"})),
            _converse_response(json.dumps({"supported": False, "confidence": 0.8, "explanation": "no"})),
        ]

        result = asyncio.run(
            grader.grade_faithfulness("Mixed truth and hallucination", ["Some evidence."])
        )

        assert result.verdict == FaithfulnessVerdict.PARTIALLY_FAITHFUL
        assert 0.5 <= result.score < 0.9
        assert result.supported_claims == 1
        assert result.unsupported_claims == 1

    def test_grade_empty_answer(self):
        grader, mock_client = _make_grader()
        mock_client.converse.return_value = _converse_response(
            "No claims could be extracted from the empty answer."
        )

        result = asyncio.run(grader.grade_faithfulness("", ["Some evidence."]))

        assert result.verdict == FaithfulnessVerdict.FAITHFUL
        assert result.score == pytest.approx(1.0)
        assert result.total_claims == 0
        assert result.claims == []

    def test_grade_bedrock_error_fails_open(self):
        grader, mock_client = _make_grader()

        async def _raise_outer(*a, **kw):
            raise RuntimeError("Outer failure")

        with mock.patch.object(grader, "_extract_claims", side_effect=_raise_outer):
            result = asyncio.run(
                grader.grade_faithfulness("some answer", ["some evidence"])
            )

        assert result.verdict == FaithfulnessVerdict.UNCLEAR
        assert result.score == pytest.approx(0.5)
        assert result.total_claims == 0

    def test_extract_claims_parse_error_fallback(self):
        grader, mock_client = _make_grader()
        mock_client.converse.side_effect = Exception("network failure")

        answer = "This is a longer answer that exceeds two hundred characters. " * 5
        claims = asyncio.run(grader._extract_claims(answer))

        assert len(claims) == 1
        assert claims[0] == answer[:200]

    def test_verify_claim_json_parse_error_fails_open(self):
        grader, mock_client = _make_grader()
        mock_client.converse.return_value = _converse_response(
            "This is not valid JSON at all! {broken"
        )

        claim = asyncio.run(grader._verify_claim("Some claim text", "Some evidence"))

        assert claim.supported is True
        assert claim.confidence == pytest.approx(0.3)
        assert claim.text == "Some claim text"

    def test_claims_limited_to_10(self):
        grader, mock_client = _make_grader()
        claim_lines = "\n".join(f"CLAIM: Claim number {i}" for i in range(15))
        mock_client.converse.return_value = _converse_response(claim_lines)

        claims = asyncio.run(grader._extract_claims("long answer with many claims"))

        assert len(claims) == 10
        assert claims[0] == "Claim number 0"
        assert claims[9] == "Claim number 9"

    def test_grade_score_exactly_09_is_faithful(self):
        grader, mock_client = _make_grader()
        claims_text = "\n".join(f"CLAIM: Claim {i}" for i in range(10))
        side_effects = [_converse_response(claims_text)]
        for i in range(10):
            supported = i < 9
            side_effects.append(
                _converse_response(json.dumps({"supported": supported, "confidence": 0.9}))
            )
        mock_client.converse.side_effect = side_effects

        result = asyncio.run(grader.grade_faithfulness("answer", ["evidence"]))

        assert result.score == pytest.approx(0.9)
        assert result.verdict == FaithfulnessVerdict.FAITHFUL

    def test_verify_claim_strips_markdown_fences(self):
        grader, mock_client = _make_grader()
        mock_client.converse.return_value = _converse_response(
            '```json\n{"supported": false, "confidence": 0.75, "explanation": "nope"}\n```'
        )

        claim = asyncio.run(grader._verify_claim("A fact", "Evidence"))

        assert claim.supported is False
        assert claim.confidence == pytest.approx(0.75)
