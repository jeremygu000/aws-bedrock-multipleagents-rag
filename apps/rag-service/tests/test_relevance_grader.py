from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from app.relevance_grader import RelevanceGrader
from app.self_reflection_models import RelevanceVerdict


def _converse_response(text: str) -> dict:
    return {"output": {"message": {"content": [{"text": text}]}}}


def _make_grader() -> tuple[RelevanceGrader, MagicMock]:
    settings = MagicMock()
    settings.aws_region = "ap-southeast-2"
    grader = RelevanceGrader(settings)
    mock_client = MagicMock()
    grader._bedrock_client = mock_client
    return grader, mock_client


class TestRelevanceGrader:

    def test_grade_high_relevance(self):
        grader, mock_client = _make_grader()
        mock_client.converse.return_value = _converse_response(
            json.dumps({
                "alignment_score": 0.9,
                "completeness_score": 0.85,
                "relevance_score": 0.8,
                "explanation": "Good answer",
            })
        )

        result = asyncio.run(grader.grade_relevance("What is APRA?", "APRA is a royalty body."))

        assert result.verdict == RelevanceVerdict.RELEVANT
        assert result.score >= 0.75
        assert result.alignment_score == pytest.approx(0.9)
        assert result.completeness_score == pytest.approx(0.85)

    def test_grade_low_relevance(self):
        grader, mock_client = _make_grader()
        mock_client.converse.return_value = _converse_response(
            json.dumps({
                "alignment_score": 0.2,
                "completeness_score": 0.1,
                "relevance_score": 0.15,
                "explanation": "Completely off topic",
            })
        )

        result = asyncio.run(grader.grade_relevance("What is licensing?", "Cats are mammals."))

        assert result.verdict == RelevanceVerdict.IRRELEVANT
        assert result.score < 0.5

    def test_grade_partial_relevance(self):
        grader, mock_client = _make_grader()
        mock_client.converse.return_value = _converse_response(
            json.dumps({
                "alignment_score": 0.6,
                "completeness_score": 0.5,
                "relevance_score": 0.55,
                "explanation": "Partially answers",
            })
        )

        result = asyncio.run(grader.grade_relevance("How does licensing work?", "Licensing varies."))

        assert result.verdict == RelevanceVerdict.PARTIALLY_RELEVANT
        assert 0.5 <= result.score < 0.75

    def test_grade_bedrock_error_fails_open(self):
        grader, mock_client = _make_grader()
        mock_client.converse.side_effect = Exception("Connection refused")

        result = asyncio.run(grader.grade_relevance("Any question?", "Any answer."))

        assert result.verdict == RelevanceVerdict.PARTIALLY_RELEVANT
        assert result.score == pytest.approx(0.6)

    def test_grade_json_parse_error_fails_open(self):
        grader, mock_client = _make_grader()
        mock_client.converse.return_value = _converse_response("Not JSON { at all }")

        result = asyncio.run(grader.grade_relevance("Q?", "A."))

        assert result.verdict == RelevanceVerdict.PARTIALLY_RELEVANT
        assert result.score == pytest.approx(0.6)

    def test_input_truncation(self):
        grader, mock_client = _make_grader()
        mock_client.converse.return_value = _converse_response(
            json.dumps({"alignment_score": 0.8, "completeness_score": 0.8, "relevance_score": 0.8})
        )

        long_question = "Q" * 600
        long_answer = "A" * 2000

        result = asyncio.run(grader.grade_relevance(long_question, long_answer))

        assert result.verdict == RelevanceVerdict.RELEVANT
        call_args = mock_client.converse.call_args
        prompt_text = call_args[1]["messages"][0]["content"][0]["text"]
        assert "Q" * 501 not in prompt_text
        assert "A" * 1501 not in prompt_text

    def test_score_computation(self):
        grader, mock_client = _make_grader()
        alignment = 0.7
        completeness = 0.8
        relevance = 0.6
        mock_client.converse.return_value = _converse_response(
            json.dumps({
                "alignment_score": alignment,
                "completeness_score": completeness,
                "relevance_score": relevance,
                "explanation": "computed",
            })
        )

        result = asyncio.run(grader.grade_relevance("Q?", "A."))

        expected = (alignment + completeness + relevance) / 3.0
        assert result.score == pytest.approx(expected)
        assert result.alignment_score == pytest.approx(alignment)
        assert result.completeness_score == pytest.approx(completeness)

    def test_markdown_fences_stripped(self):
        grader, mock_client = _make_grader()
        mock_client.converse.return_value = _converse_response(
            '```json\n{"alignment_score": 0.9, "completeness_score": 0.9, "relevance_score": 0.9}\n```'
        )

        result = asyncio.run(grader.grade_relevance("Q?", "A."))

        assert result.verdict == RelevanceVerdict.RELEVANT
        assert result.score == pytest.approx(0.9)

    def test_grade_boundary_075_is_relevant(self):
        grader, mock_client = _make_grader()
        mock_client.converse.return_value = _converse_response(
            json.dumps({"alignment_score": 0.75, "completeness_score": 0.75, "relevance_score": 0.75})
        )

        result = asyncio.run(grader.grade_relevance("Q?", "A."))

        assert result.verdict == RelevanceVerdict.RELEVANT
        assert result.score == pytest.approx(0.75)

    def test_grade_boundary_050_is_partially_relevant(self):
        grader, mock_client = _make_grader()
        mock_client.converse.return_value = _converse_response(
            json.dumps({"alignment_score": 0.5, "completeness_score": 0.5, "relevance_score": 0.5})
        )

        result = asyncio.run(grader.grade_relevance("Q?", "A."))

        assert result.verdict == RelevanceVerdict.PARTIALLY_RELEVANT
        assert result.score == pytest.approx(0.5)

    def test_missing_keys_default_to_05(self):
        grader, mock_client = _make_grader()
        mock_client.converse.return_value = _converse_response(
            json.dumps({"explanation": "no scores provided"})
        )

        result = asyncio.run(grader.grade_relevance("Q?", "A."))

        assert result.score == pytest.approx(0.5)
