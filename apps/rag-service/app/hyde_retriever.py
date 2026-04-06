"""
HyDE (Hypothetical Document Embeddings) Retriever for Production RAG.

Generates hypothetical documents from queries to improve retrieval accuracy.
Compatible with AWS Bedrock Nova Pro + pgvector embeddings.

Based on: Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels" (arXiv:2212.10496)
"""

import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HyDEConfig(BaseModel):
    """HyDE configuration."""

    enabled: bool = Field(default=True, description="Enable HyDE retrieval")
    min_query_length: int = Field(default=5, description="Minimum tokens to enable HyDE")
    temperature: float = Field(default=0.7, description="LLM temperature for hypothesis generation")
    max_hypothesis_tokens: int = Field(default=500, description="Max tokens for hypothesis document")
    include_original: bool = Field(
        default=True, description="Include original query in embedding (dual strategy)"
    )
    num_hypotheses: int = Field(default=1, description="Number of hypotheses to generate (1=simple, 5=multi)")


class HyDERetriever:
    """HyDE retriever for query expansion via hypothetical documents."""

    def __init__(self, llm_client, embedding_model, config: Optional[HyDEConfig] = None):
        """Initialize HyDE retriever.

        Args:
            llm_client: LLM client for hypothesis generation (e.g., ChatBedrock)
            embedding_model: Embedding model for vector generation
            config: HyDE configuration
        """
        self.llm = llm_client
        self.embeddings = embedding_model
        self.config = config or HyDEConfig()

        self.system_prompt = """You are an expert at understanding information retrieval tasks.
You will be given a query and asked to write a passage to answer it.
Your goal is to capture key details and semantic patterns that would appear in relevant documents.
Write the passage as if it were an excerpt from an actual document."""

        self.user_prompt_template = """Please write a passage to answer the following question.
Try to include as many key details as possible.
Write naturally, as if from a real document.

Question: {query}

Passage:"""

    def _should_use_hyde(self, query: str, use_query_router: bool = True) -> bool:
        """Determine if HyDE should be applied to this query.

        Args:
            query: User query
            use_query_router: Whether to use advanced routing logic

        Returns:
            True if HyDE should be enabled for this query
        """
        if not self.config.enabled:
            return False

        # Length check: short queries (< 5 tokens) - cost > benefit
        tokens = len(query.split())
        if tokens < self.config.min_query_length:
            logger.debug(f"Skipping HyDE: query too short ({tokens} tokens < {self.config.min_query_length})")
            return False

        if not use_query_router:
            return True

        # Entity detection: queries with named entities often better served by BM25
        if self._has_entities(query):
            logger.debug("Skipping HyDE: query contains named entities")
            return False

        # Reasoning intent: favor HyDE for complex reasoning
        if self._is_reasoning_query(query):
            logger.debug("Using HyDE: detected reasoning query")
            return True

        # Default: enable for other queries
        return True

    def _has_entities(self, query: str) -> bool:
        """Heuristic entity detection (names, dates, IDs).

        Args:
            query: Query string

        Returns:
            True if query contains likely named entities
        """
        # Pattern: Capitalized words (likely proper nouns)
        caps_words = len(re.findall(r"\b[A-Z][a-z]+\b", query))
        if caps_words > 2:
            return True

        # Pattern: dates (YYYY or MM/DD)
        if re.search(r"\b(19|20)\d{2}\b|\b(0?[1-9]|1[0-2])/\d{1,2}\b", query):
            return True

        # Pattern: IDs or codes (alphanumeric with hyphens)
        if re.search(r"\b[A-Z]{2,}-\d+\b|\bID:\s*\d+\b", query):
            return True

        return False

    def _is_reasoning_query(self, query: str) -> bool:
        """Detect reasoning-heavy queries.

        Args:
            query: Query string

        Returns:
            True if query requires reasoning
        """
        reasoning_keywords = [
            "explain",
            "why",
            "how",
            "compare",
            "contrast",
            "analyze",
            "discuss",
            "evaluate",
            "summarize",
            "what are",
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in reasoning_keywords)

    def generate_hypothesis(self, query: str) -> str:
        """Generate a single hypothetical document from query.

        Args:
            query: User query

        Returns:
            Hypothetical document text
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self.user_prompt_template.format(query=query)),
            ]

            response = self.llm.invoke(messages)
            hypothesis = response.content.strip()

            logger.debug(f"Generated hypothesis: {hypothesis[:100]}...")
            return hypothesis

        except Exception as e:
            logger.error(f"Failed to generate hypothesis: {e}")
            # Fallback: return original query
            return query

    def generate_multi_hypotheses(self, query: str, num: int = 5) -> list[str]:
        """Generate multiple hypothetical documents from different perspectives.

        Args:
            query: User query
            num: Number of hypotheses to generate

        Returns:
            List of hypothetical documents
        """
        perspectives = [
            "technical and detailed",
            "practical and actionable",
            "historical and contextual",
            "comparative and relational",
            "foundational and definitional",
        ][:num]

        hypotheses = []
        for i, perspective in enumerate(perspectives):
            prompt = self.user_prompt_template.format(query=query)
            prompt += f"\n\nWrite from a {perspective} perspective."

            try:
                from langchain_core.messages import HumanMessage, SystemMessage

                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=prompt),
                ]
                response = self.llm.invoke(messages)
                hypothesis = response.content.strip()
                hypotheses.append(hypothesis)
            except Exception as e:
                logger.warning(f"Failed to generate hypothesis {i+1}: {e}")
                hypotheses.append(query)  # Fallback

        return hypotheses

    def get_query_embeddings(self, query: str) -> dict:
        """Generate embeddings for HyDE retrieval.

        Strategy:
        - Single hypothesis mode (default): embed only hypothesis
        - Multi-hypothesis mode: average multiple hypotheses
        - include_original=True: dual strategy (hypothesis + original query)

        Args:
            query: User query

        Returns:
            Dict with 'embeddings', 'sources', and 'strategy'
        """
        if not self._should_use_hyde(query, use_query_router=True):
            logger.debug("HyDE disabled for query, using original embedding")
            return {
                "embeddings": [self.embeddings.embed_query(query)],
                "sources": ["original_query"],
                "strategy": "original",
            }

        embeddings_list = []
        sources = []

        # Generate hypothesis/hypotheses
        if self.config.num_hypotheses == 1:
            hypothesis = self.generate_hypothesis(query)
            hypotheses = [hypothesis]
        else:
            hypotheses = self.generate_multi_hypotheses(query, num=self.config.num_hypotheses)

        # Embed hypotheses
        for i, hyp in enumerate(hypotheses):
            try:
                emb = self.embeddings.embed_query(hyp)
                embeddings_list.append(emb)
                sources.append(f"hypothesis_{i+1}")
            except Exception as e:
                logger.warning(f"Failed to embed hypothesis {i+1}: {e}")

        # Average if multiple hypotheses
        if len(embeddings_list) > 1:
            import numpy as np

            avg_embedding = np.mean(embeddings_list, axis=0).tolist()
            embeddings_list = [avg_embedding]
            sources = ["averaged_hypotheses"]

        # include_original: dual strategy
        if self.config.include_original:
            try:
                orig_emb = self.embeddings.embed_query(query)
                embeddings_list.append(orig_emb)
                sources.append("original_query")
                logger.debug("Using dual strategy: hypothesis + original")
            except Exception as e:
                logger.warning(f"Failed to embed original query: {e}")

        return {
            "embeddings": embeddings_list,
            "sources": sources,
            "strategy": "multi_hypothesis_averaged" if self.config.num_hypotheses > 1 else "single_hypothesis",
        }

    def rerank_with_original(self, results: list, query: str, reranker=None) -> list:
        """Optional: rerank results using original query for safety.

        When using HyDE, hypothesis may introduce bias. Reranking with original
        query + reranker provides additional safety layer.

        Args:
            results: Retrieved results
            query: Original query
            reranker: Optional reranker callable (takes query, results)

        Returns:
            Reranked results
        """
        if reranker is None:
            return results

        try:
            reranked = reranker(query, results)
            logger.debug(f"Reranked {len(results)} results using original query")
            return reranked
        except Exception as e:
            logger.warning(f"Reranking failed, returning original order: {e}")
            return results
