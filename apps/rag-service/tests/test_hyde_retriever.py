"""Unit tests for HyDE retriever."""

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

from app.hyde_retriever import BedrockLLMAdapter, HyDEConfig, HyDERetriever, _LLMResponse

# ---------------------------------------------------------------------------
# _LLMResponse
# ---------------------------------------------------------------------------


class TestLLMResponse:
    """Test the _LLMResponse dataclass."""

    def test_content_attribute(self):
        r = _LLMResponse(content="hello")
        assert r.content == "hello"

    def test_is_dataclass(self):
        assert len(fields(_LLMResponse)) == 1
        assert fields(_LLMResponse)[0].name == "content"

    def test_empty_content(self):
        r = _LLMResponse(content="")
        assert r.content == ""


# ---------------------------------------------------------------------------
# BedrockLLMAdapter
# ---------------------------------------------------------------------------


def _converse_response(text: str) -> dict:
    """Build a minimal Bedrock converse response."""
    return {"output": {"message": {"content": [{"text": text}]}}}


class TestBedrockLLMAdapter:
    """Test BedrockLLMAdapter wrapping boto3 converse."""

    @patch("app.hyde_retriever.boto3")
    def test_init_creates_client(self, mock_boto3):
        adapter = BedrockLLMAdapter("model-x", "us-west-2", temperature=0.5, max_tokens=256)
        mock_boto3.client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")
        assert adapter._model_id == "model-x"
        assert adapter._temperature == 0.5
        assert adapter._max_tokens == 256

    # -- _convert_messages -----------------------------------------------

    @patch("app.hyde_retriever.boto3")
    def test_convert_dict_messages(self, mock_boto3):
        adapter = BedrockLLMAdapter("m", "r")
        system, msgs = adapter._convert_messages([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        assert system == "sys"
        assert msgs == [
            {"role": "user", "content": [{"text": "hi"}]},
            {"role": "assistant", "content": [{"text": "hello"}]},
        ]

    @patch("app.hyde_retriever.boto3")
    def test_convert_langchain_message_objects(self, mock_boto3):
        """Simulate LangChain message objects using simple namespace."""
        from dataclasses import dataclass

        @dataclass
        class FakeMsg:
            type: str
            content: str

        adapter = BedrockLLMAdapter("m", "r")
        system, msgs = adapter._convert_messages([
            FakeMsg(type="system", content="sys prompt"),
            FakeMsg(type="human", content="question"),
            FakeMsg(type="ai", content="answer"),
        ])
        assert system == "sys prompt"
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    @patch("app.hyde_retriever.boto3")
    def test_convert_unknown_role_maps_to_user(self, mock_boto3):
        adapter = BedrockLLMAdapter("m", "r")
        system, msgs = adapter._convert_messages([{"role": "tool", "content": "data"}])
        assert system is None
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == [{"text": "data"}]

    @patch("app.hyde_retriever.boto3")
    def test_convert_empty_messages(self, mock_boto3):
        adapter = BedrockLLMAdapter("m", "r")
        system, msgs = adapter._convert_messages([])
        assert system is None
        assert msgs == []

    # -- invoke ----------------------------------------------------------

    @patch("app.hyde_retriever.boto3")
    def test_invoke_returns_llm_response(self, mock_boto3):
        client_mock = MagicMock()
        mock_boto3.client.return_value = client_mock
        client_mock.converse.return_value = _converse_response("result text")

        adapter = BedrockLLMAdapter("m", "r")
        resp = adapter.invoke([{"role": "user", "content": "hi"}])

        assert isinstance(resp, _LLMResponse)
        assert resp.content == "result text"

    @patch("app.hyde_retriever.boto3")
    def test_invoke_passes_system_prompt(self, mock_boto3):
        client_mock = MagicMock()
        mock_boto3.client.return_value = client_mock
        client_mock.converse.return_value = _converse_response("ok")

        adapter = BedrockLLMAdapter("m", "r")
        adapter.invoke([
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ])

        call_kwargs = client_mock.converse.call_args[1]
        assert call_kwargs["system"] == [{"text": "be helpful"}]

    @patch("app.hyde_retriever.boto3")
    def test_invoke_without_system_prompt(self, mock_boto3):
        client_mock = MagicMock()
        mock_boto3.client.return_value = client_mock
        client_mock.converse.return_value = _converse_response("ok")

        adapter = BedrockLLMAdapter("m", "r")
        adapter.invoke([{"role": "user", "content": "hi"}])

        call_kwargs = client_mock.converse.call_args[1]
        assert "system" not in call_kwargs

    @patch("app.hyde_retriever.boto3")
    def test_invoke_passes_inference_config(self, mock_boto3):
        client_mock = MagicMock()
        mock_boto3.client.return_value = client_mock
        client_mock.converse.return_value = _converse_response("ok")

        adapter = BedrockLLMAdapter("m", "r", temperature=0.3, max_tokens=100)
        adapter.invoke([{"role": "user", "content": "hi"}])

        call_kwargs = client_mock.converse.call_args[1]
        assert call_kwargs["inferenceConfig"] == {"temperature": 0.3, "maxTokens": 100}

    @patch("app.hyde_retriever.boto3")
    def test_invoke_reraises_on_error(self, mock_boto3):
        client_mock = MagicMock()
        mock_boto3.client.return_value = client_mock
        client_mock.converse.side_effect = RuntimeError("network error")

        adapter = BedrockLLMAdapter("m", "r")
        with pytest.raises(RuntimeError, match="network error"):
            adapter.invoke([{"role": "user", "content": "hi"}])


class TestHyDEConfig:

    def test_default_config(self):
        config = HyDEConfig()
        assert config.enabled is True
        assert config.min_query_length == 5
        assert config.temperature == 0.7
        assert config.include_original is True
        assert config.max_hypothesis_tokens == 500
        assert config.num_hypotheses == 1

    def test_custom_config(self):
        config = HyDEConfig(enabled=False, min_query_length=10, num_hypotheses=3)
        assert config.enabled is False
        assert config.min_query_length == 10
        assert config.num_hypotheses == 3


# ---------------------------------------------------------------------------
# HyDERetriever helpers
# ---------------------------------------------------------------------------


def _make_retriever(config: HyDEConfig | None = None):
    llm_mock = MagicMock()
    llm_mock.invoke.return_value = _LLMResponse(content="  hypothetical doc  ")
    embeddings_mock = MagicMock()
    embeddings_mock.embed_query.return_value = [0.1, 0.2, 0.3]
    retriever = HyDERetriever(llm_mock, embeddings_mock, config or HyDEConfig())
    return retriever, llm_mock, embeddings_mock


# ---------------------------------------------------------------------------
# _should_use_hyde
# ---------------------------------------------------------------------------


class TestShouldUseHyde:

    def test_disabled_config(self):
        r, _, _ = _make_retriever(HyDEConfig(enabled=False))
        assert r._should_use_hyde("explain machine learning concepts in detail") is False

    def test_short_query_below_min_length(self):
        r, _, _ = _make_retriever(HyDEConfig(min_query_length=5))
        assert r._should_use_hyde("hello world") is False  # 2 tokens < 5

    def test_short_query_exact_min_length(self):
        r, _, _ = _make_retriever(HyDEConfig(min_query_length=5))
        assert r._should_use_hyde("one two three four five") is True  # exactly 5 tokens

    def test_bypass_query_router(self):
        r, _, _ = _make_retriever()
        assert r._should_use_hyde("a sufficiently long query for testing purposes", use_query_router=False) is True

    def test_entity_query_skipped(self):
        r, _, _ = _make_retriever()
        assert r._should_use_hyde("Tell me about John Smith born in 2000") is False

    def test_reasoning_query_enabled(self):
        r, _, _ = _make_retriever()
        assert r._should_use_hyde("Explain why machine learning works better than traditional methods") is True

    def test_default_path_long_query(self):
        r, _, _ = _make_retriever()
        # Long query, no entities, no reasoning keywords → default path returns True
        assert r._should_use_hyde("the quick brown fox jumps over the lazy dog repeatedly") is True

    def test_date_pattern_detected_as_entity(self):
        r, _, _ = _make_retriever()
        assert r._should_use_hyde("what happened on 2024 in the world of technology and science") is False

    def test_winf_id_detected_as_entity(self):
        r, _, _ = _make_retriever()
        assert r._should_use_hyde("find the record for WINFabc123 in the database please") is False


# ---------------------------------------------------------------------------
# _has_entities
# ---------------------------------------------------------------------------


class TestHasEntities:

    def test_caps_below_threshold_not_detected(self):
        r, _, _ = _make_retriever()
        # Only 2 capitalized words — threshold is >2
        assert r._has_entities("John visited Paris") is False

    def test_caps_above_threshold(self):
        r, _, _ = _make_retriever()
        assert r._has_entities("John Smith visited Google Headquarters") is True  # 4 caps words

    def test_date_yyyy(self):
        r, _, _ = _make_retriever()
        assert r._has_entities("events in 2023") is True

    def test_date_mm_dd(self):
        r, _, _ = _make_retriever()
        assert r._has_entities("meeting on 3/15") is True

    def test_id_pattern_with_hyphen(self):
        r, _, _ = _make_retriever()
        assert r._has_entities("case AB-1234") is True

    def test_id_colon_pattern(self):
        r, _, _ = _make_retriever()
        assert r._has_entities("ID: xyz789") is True

    def test_winf_pattern(self):
        r, _, _ = _make_retriever()
        assert r._has_entities("song WINFabc") is True

    def test_no_entities(self):
        r, _, _ = _make_retriever()
        assert r._has_entities("what is machine learning") is False

    def test_lowercase_no_patterns(self):
        r, _, _ = _make_retriever()
        assert r._has_entities("how to build a better system quickly") is False


# ---------------------------------------------------------------------------
# _is_reasoning_query
# ---------------------------------------------------------------------------


class TestIsReasoningQuery:

    @pytest.mark.parametrize("kw", [
        "explain", "why", "how", "compare", "contrast",
        "analyze", "discuss", "evaluate", "summarize", "what are",
    ])
    def test_each_reasoning_keyword(self, kw):
        r, _, _ = _make_retriever()
        query = f"{kw} the impact of technology on modern education systems"
        assert r._is_reasoning_query(query) is True

    def test_case_insensitive(self):
        r, _, _ = _make_retriever()
        assert r._is_reasoning_query("EXPLAIN the difference between these two approaches") is True

    def test_no_reasoning_keywords(self):
        r, _, _ = _make_retriever()
        assert r._is_reasoning_query("find a document about cats") is False

    def test_substring_match_in_word(self):
        r, _, _ = _make_retriever()
        # "show" contains "how" as substring → `in` operator matches
        assert r._is_reasoning_query("show me the data") is True

    def test_no_keyword_at_all(self):
        r, _, _ = _make_retriever()
        assert r._is_reasoning_query("list all records from database") is False


# ---------------------------------------------------------------------------
# generate_hypothesis
# ---------------------------------------------------------------------------


class TestGenerateHypothesis:

    def test_success_strips_whitespace(self):
        r, llm, _ = _make_retriever()
        llm.invoke.return_value = _LLMResponse(content="  hypothesis text  ")
        result = r.generate_hypothesis("some query about topics")
        assert result == "hypothesis text"
        llm.invoke.assert_called_once()

    def test_error_fallback_returns_query(self):
        r, llm, _ = _make_retriever()
        llm.invoke.side_effect = RuntimeError("boom")
        result = r.generate_hypothesis("my query")
        assert result == "my query"

    def test_invokes_with_langchain_messages(self):
        r, llm, _ = _make_retriever()
        r.generate_hypothesis("test query for message types")
        call_args = llm.invoke.call_args[0][0]
        # Should be a list of LangChain message objects
        assert len(call_args) == 2
        assert call_args[0].type == "system"
        assert call_args[1].type == "human"
        assert "test query for message types" in call_args[1].content


# ---------------------------------------------------------------------------
# generate_multi_hypotheses
# ---------------------------------------------------------------------------


class TestGenerateMultiHypotheses:

    def test_success_returns_n_hypotheses(self):
        r, llm, _ = _make_retriever()
        counter = [0]

        def side_effect(msgs):
            counter[0] += 1
            return _LLMResponse(content=f"hypothesis {counter[0]}")

        llm.invoke.side_effect = side_effect
        result = r.generate_multi_hypotheses("complex query", num=3)
        assert len(result) == 3
        assert result[0] == "hypothesis 1"
        assert result[2] == "hypothesis 3"

    def test_partial_failure_uses_query_fallback(self):
        r, llm, _ = _make_retriever()
        call_count = [0]

        def side_effect(msgs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("fail")
            return _LLMResponse(content=f"ok {call_count[0]}")

        llm.invoke.side_effect = side_effect
        result = r.generate_multi_hypotheses("query", num=3)
        assert len(result) == 3
        assert result[0] == "ok 1"
        assert result[1] == "query"  # fallback
        assert result[2] == "ok 3"

    def test_capped_by_perspectives_max_5(self):
        r, llm, _ = _make_retriever()
        llm.invoke.return_value = _LLMResponse(content="h")
        result = r.generate_multi_hypotheses("query", num=10)
        # perspectives list only has 5 entries, so capped at 5
        assert len(result) == 5

    def test_prompts_contain_perspective_text(self):
        r, llm, _ = _make_retriever()
        llm.invoke.return_value = _LLMResponse(content="h")
        r.generate_multi_hypotheses("query", num=2)
        # 2nd call should contain "practical and actionable"
        second_call = llm.invoke.call_args_list[1][0][0]
        assert "practical and actionable" in second_call[1].content


# ---------------------------------------------------------------------------
# get_query_embeddings
# ---------------------------------------------------------------------------


class TestGetQueryEmbeddings:

    def test_disabled_returns_original(self):
        r, _, emb = _make_retriever(HyDEConfig(enabled=False))
        emb.embed_query.return_value = [1.0, 2.0]
        result = r.get_query_embeddings("short")
        assert result["strategy"] == "original"
        assert result["sources"] == ["original_query"]
        assert result["embeddings"] == [[1.0, 2.0]]

    def test_short_query_returns_original(self):
        r, _, emb = _make_retriever(HyDEConfig(min_query_length=10))
        emb.embed_query.return_value = [0.5]
        result = r.get_query_embeddings("hi there")
        assert result["strategy"] == "original"

    def test_single_hypothesis_with_original(self):
        r, llm, emb = _make_retriever(HyDEConfig(include_original=True, num_hypotheses=1))
        llm.invoke.return_value = _LLMResponse(content="hyp text")
        call_count = [0]

        def embed_side(text):
            call_count[0] += 1
            return [float(call_count[0])]

        emb.embed_query.side_effect = embed_side

        result = r.get_query_embeddings("explain machine learning concepts in detail")
        assert result["strategy"] == "single_hypothesis"
        assert len(result["embeddings"]) == 2  # hypothesis + original
        assert "original_query" in result["sources"]

    def test_single_hypothesis_without_original(self):
        r, llm, emb = _make_retriever(HyDEConfig(include_original=False, num_hypotheses=1))
        llm.invoke.return_value = _LLMResponse(content="hyp text")
        emb.embed_query.return_value = [0.5, 0.5]

        result = r.get_query_embeddings("explain machine learning concepts in detail")
        assert result["strategy"] == "single_hypothesis"
        assert len(result["embeddings"]) == 1
        assert "original_query" not in result["sources"]

    def test_multi_hypothesis_averages_embeddings(self):
        r, llm, emb = _make_retriever(HyDEConfig(include_original=False, num_hypotheses=3))
        counter = [0]

        def llm_side(msgs):
            counter[0] += 1
            return _LLMResponse(content=f"hyp {counter[0]}")

        llm.invoke.side_effect = llm_side

        # Return different embeddings for each hypothesis
        embed_values = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        emb.embed_query.side_effect = embed_values

        result = r.get_query_embeddings("explain how machine learning and AI work together")
        assert result["strategy"] == "multi_hypothesis_averaged"
        assert result["sources"] == ["averaged_hypotheses"]
        # Average of [1,0], [0,1], [0,0] = [0.333, 0.333]
        avg = result["embeddings"][0]
        assert abs(avg[0] - 1 / 3) < 0.01
        assert abs(avg[1] - 1 / 3) < 0.01

    def test_embed_failure_skips_hypothesis(self):
        r, llm, emb = _make_retriever(HyDEConfig(include_original=True, num_hypotheses=1))
        llm.invoke.return_value = _LLMResponse(content="hyp")
        call_count = [0]

        def embed_side(text):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("embed fail")
            return [0.9]

        emb.embed_query.side_effect = embed_side

        result = r.get_query_embeddings("explain machine learning concepts in detail")
        # Hypothesis embed failed, but original should succeed
        assert len(result["embeddings"]) >= 1
        assert "original_query" in result["sources"]

    def test_llm_failure_returns_fallback_embeddings(self):
        r, llm, emb = _make_retriever(HyDEConfig(include_original=True, num_hypotheses=1))
        llm.invoke.side_effect = RuntimeError("LLM down")
        emb.embed_query.return_value = [0.5, 0.5]

        result = r.get_query_embeddings("explain the relationship between retrieval and generation in RAG systems")
        # generate_hypothesis falls back to query text, which still gets embedded
        assert len(result["embeddings"]) > 0


# ---------------------------------------------------------------------------
# rerank_with_original
# ---------------------------------------------------------------------------


class TestRerankWithOriginal:

    def test_no_reranker_returns_original(self):
        r, _, _ = _make_retriever()
        results = [{"id": 1}, {"id": 2}]
        out = r.rerank_with_original(results, "query")
        assert out == results

    def test_with_reranker_calls_and_returns(self):
        r, _, _ = _make_retriever()
        reranker = MagicMock(return_value=[{"id": 2}, {"id": 1}])
        results = [{"id": 1}, {"id": 2}]
        out = r.rerank_with_original(results, "query", reranker=reranker)
        reranker.assert_called_once_with("query", results)
        assert out == [{"id": 2}, {"id": 1}]

    def test_reranker_failure_returns_original(self):
        r, _, _ = _make_retriever()
        reranker = MagicMock(side_effect=RuntimeError("fail"))
        results = [{"id": 1}]
        out = r.rerank_with_original(results, "query", reranker=reranker)
        assert out == results


# ---------------------------------------------------------------------------
# HyDERetriever init
# ---------------------------------------------------------------------------


class TestHyDERetrieverInit:

    def test_default_config_when_none(self):
        llm, emb = MagicMock(), MagicMock()
        r = HyDERetriever(llm, emb)
        assert r.config.enabled is True
        assert r.config.min_query_length == 5

    def test_custom_config_applied(self):
        llm, emb = MagicMock(), MagicMock()
        cfg = HyDEConfig(enabled=False, temperature=0.3)
        r = HyDERetriever(llm, emb, cfg)
        assert r.config.enabled is False
        assert r.config.temperature == 0.3

    def test_system_prompt_set(self):
        llm, emb = MagicMock(), MagicMock()
        r = HyDERetriever(llm, emb)
        assert "expert" in r.system_prompt.lower()
        assert "passage" in r.user_prompt_template.lower()
