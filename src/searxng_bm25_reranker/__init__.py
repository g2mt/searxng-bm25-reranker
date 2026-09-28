"""BM25 reranking plugin for SearXNG.

Uses sparse_search (zerodep BM25 implementation) to rerank search results
by text relevance, with RRF fusion to preserve engine ranking signals.
"""

from __future__ import annotations

import logging
import re
import typing as t

import searx
from searx.plugins import Plugin, PluginInfo  # ty: ignore[unresolved-import]

from ._tokenizer import _has_cjk, cjk_tokenize
from ._vendor.sparse_search import Result as SparseResult
from ._vendor.sparse_search import SparseIndex, rrf

if t.TYPE_CHECKING:
    from searx.extended_types import SXNG_Request  # ty: ignore[unresolved-import]
    from searx.plugins import PluginCfg  # ty: ignore[unresolved-import]
    from searx.search import SearchWithPlugins  # ty: ignore[unresolved-import]

import math
import urllib.parse

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standalone ranking functions
# ---------------------------------------------------------------------------


def _compute_bm25_ranking(
    query: str,
    results: list[t.Any],
    *,
    field_weights: dict[str, float] | None = None,
) -> list[SparseResult] | None:
    """Compute BM25F ranking for search results.

    Builds a temporary BM25F index from result title+content fields and
    retrieves the top results matching *query*.  The returned list can be
    passed directly to :func:`rrf` for fusion with other rankings.

    Args:
        query: Original search query.
        results: List of result objects (supporting ``[]`` access for
            ``"title"`` and ``"content"``).
        field_weights: Optional BM25F field weights.  Defaults to
            ``{"title": 2.0, "content": 1.0}``.

    Returns:
        List of :class:`SparseResult` sorted by descending BM25 score,
        or ``None`` when fewer than two results have usable text.
    """
    if field_weights is None:
        field_weights = {"title": 2.0, "content": 1.0}

    idx = SparseIndex(
        variant="bm25",
        field_weights=field_weights,
        tokenize=cjk_tokenize,
    )

    valid_indices: list[int] = []
    for i, r in enumerate(results):
        title = _get_text(r, "title")
        content = _get_text(r, "content")
        if not title and not content:
            continue
        idx.add(str(i), {"title": title, "content": content})
        valid_indices.append(i)

    if len(valid_indices) < 2:
        return None

    return idx.search(query, top_k=len(valid_indices))


def _compute_lm_embedding_ranking(
    query: str,
    results: list[t.Any],
    *,
    lm_host: str,
    lm_query_prefix: str = "",
    lm_doc_prefix: str = "",
) -> list[SparseResult] | None:
    """Compute embedding-similarity ranking via an OpenAI-compatible API.

    Sends *query* and the text of each result to the embedding endpoint,
    then ranks results by cosine similarity to the query embedding.
    Documents are truncated to roughly 512 tokens before embedding.
    The returned list can be passed directly to :func:`rrf` for fusion.

    Args:
        query: Original search query.
        results: List of result objects (supporting ``[]`` access for
            ``"title"`` and ``"content"``).
        lm_host: Base URL of the embedding API (e.g. ``http://localhost:11434``).
            The endpoint ``{lm_host}/v1/embeddings`` is called.

    Returns:
        List of :class:`SparseResult` sorted by descending cosine
        similarity, or ``None`` on failure / insufficient results.
    """
    # Collect texts: query first, then result title+content pairs
    # (documents truncated to ~512 tokens to bound payload size)
    texts: list[str] = [_truncate_tokens(f"{lm_query_prefix}{query}".strip())]
    valid_indices: list[int] = []
    for i, r in enumerate(results):
        title = _get_text(r, "title")
        content = _get_text(r, "content")
        url = _get_text(r, "url")
        host = _url_host(url)
        text = _truncate_tokens(f"{lm_doc_prefix}{title} {host}\n{content}".strip())
        if text:
            texts.append(text)
            valid_indices.append(i)

    if len(valid_indices) < 2:
        return None

    # Obtain embeddings from the API
    try:
        embeddings = _fetch_embeddings(lm_host, texts)
    except Exception:
        logger.exception("LM embedding API call failed")
        return None

    if embeddings is None or len(embeddings) != len(texts):
        logger.warning(
            "Embedding API returned %d vectors, expected %d",
            len(embeddings) if embeddings else 0,
            len(texts),
        )
        return None

    query_emb = embeddings[0]
    result_embs = embeddings[1:]

    # Rank by cosine similarity
    scored: list[SparseResult] = []
    for idx, emb in enumerate(result_embs):
        dot = sum(a * b for a, b in zip(query_emb, emb))
        norm_q = math.sqrt(sum(a * a for a in query_emb))
        norm_e = math.sqrt(sum(b * b for b in emb))
        sim = dot / (norm_q * norm_e) if norm_q and norm_e else 0.0
        scored.append(SparseResult(doc_id=str(valid_indices[idx]), score=sim))

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Helpers for the embedding pipeline
# ---------------------------------------------------------------------------


_SPACE_RE = re.compile(r"[ \t]+")


def _url_host(url: str) -> str:
    """Extract the hostname from a URL, without scheme, path, or port."""
    try:
        return urllib.parse.urlparse(url).hostname or ""
    except ValueError:
        return ""


def _truncate_tokens(text: str, max_tokens: int = 512) -> str:
    """Normalize and truncate text to roughly *max_tokens* tokens.

    Runs of spaces/tabs are collapsed to a single space, then the text is
    cut down using a lightweight heuristic (about 4 characters per token
    for Latin text, 1 character per token for CJK) so the full text never
    needs to be tokenized.  The cut lands on the nearest whitespace
    boundary to avoid splitting mid-word.
    """
    if not text:
        return text

    text = _SPACE_RE.sub(" ", text)

    # CJK text is roughly 1 token per character; Latin ~4 chars per token
    budget = max_tokens if _has_cjk(text) else max_tokens * 4
    if len(text) <= budget:
        return text

    cut = text[:budget]
    # Prefer a whitespace boundary so we don't split mid-word
    ws = max(cut.rfind(" "), cut.rfind("\n"), cut.rfind("\t"))
    if ws > budget // 2:
        cut = cut[:ws]
    return cut


def _fetch_embeddings(host: str, texts: list[str]) -> list[list[float]] | None:
    """Call an OpenAI-compatible ``/v1/embeddings`` endpoint."""
    import json
    import urllib.request

    url = f"{host.rstrip('/')}/v1/embeddings"
    payload = json.dumps({"model": "default", "input": texts}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    # OpenAI shape: {"data": [{"embedding": [...], "index": 0}, ...]}
    items = sorted(body["data"], key=lambda e: e["index"])
    return [e["embedding"] for e in items]


class SXNGPlugin(Plugin):
    """Rerank search results using BM25 scoring with RRF fusion."""

    id = "bm25_reranker"

    def __init__(self, plg_cfg: PluginCfg) -> None:
        super().__init__(plg_cfg)
        self._settings_prefix: str = self.id  # reads from searx.settings["bm25_reranker"]
        self.info = PluginInfo(
            id=self.id,
            name="BM25 Reranker",
            description="Reranks search results using BM25 text relevance scoring with RRF fusion.",
            preference_section="general",
        )

    def post_search(self, request: SXNG_Request, search: SearchWithPlugins) -> None:
        """Rerank results by BM25 relevance fused with original engine ranking.

        Accesses main_results_map before close() calculates scores,
        builds a temporary BM25F index, and rewrites positions to
        influence the final scoring formula.
        """
        results_map = search.result_container.main_results_map
        if len(results_map) < 2:
            return None

        query = search.search_query.query
        if not query or not query.strip():
            return None

        try:
            self._rerank(query, results_map)
        except Exception:
            logger.exception("BM25 reranking failed, keeping original order")

        logger.debug(
            "BM25 reranker processed %d results for: %s", len(results_map), query[:50]
        )

        return None

    def _rerank(self, query: str, results_map: dict) -> None:
        """Core reranking logic.

        Combines the original engine ranking with BM25 text relevance and
        optionally LM embedding similarity via weighted RRF (Reciprocal
        Rank Fusion).

        Args:
            query: Original search query.
            results_map: Dict of result hash -> MainResult/LegacyResult objects.
        """
        # Discard results that do not contain any query word in their title
        # or description before applying any ranking.  Keeping these results
        # in the engine ranking would allow them to survive RRF even though
        # they are not textually relevant.
        query_words = set(cjk_tokenize(query))
        for result_id, result in list(results_map.items()):
            if not _result_matches_query(result, query_words):
                del results_map[result_id]

        if len(results_map) < 2:
            return

        results = list(results_map.values())

        # ---- BM25 ranking -------------------------------------------------
        bm25_results = _compute_bm25_ranking(query, results)

        # ---- LM embedding ranking (only when lm_weight > 0) ---------------
        cfg = searx.settings.get(self._settings_prefix, {})
        lm_weight: float = float(cfg.get("lm_weight", 0))
        lm_results: list[SparseResult] | None = None
        if lm_weight > 0:
            lm_host: str = str(cfg.get("lm_host", ""))
            if lm_host:
                lm_query_prefix: str = str(cfg.get("lm_query_prefix", ""))
                lm_doc_prefix: str = str(cfg.get("lm_doc_prefix", ""))
                lm_results = _compute_lm_embedding_ranking(
                    query,
                    results,
                    lm_host=lm_host,
                    lm_query_prefix=lm_query_prefix,
                    lm_doc_prefix=lm_doc_prefix,
                )

        # ---- Determine valid indices (results with usable text) -----------
        valid_indices: list[int] = []
        for i, r in enumerate(results):
            title = _get_text(r, "title")
            content = _get_text(r, "content")
            if title or content:
                valid_indices.append(i)

        if len(valid_indices) < 2:
            return

        # ---- Engine ranking (original positions → reciprocal scores) ------
        engine_ranking: list[SparseResult] = [
            SparseResult(doc_id=str(i), score=1.0 / (rank + 1))
            for rank, i in enumerate(valid_indices)
        ]

        # ---- Weighted RRF fusion ------------------------------------------
        bm25_weight: float = float(cfg.get("bm25_weight", 1.0))
        result_lists: list[list[SparseResult]] = [engine_ranking]
        rrf_weights: list[float] = [1.0]

        if bm25_results:
            result_lists.append(bm25_results)
            rrf_weights.append(bm25_weight)

        if lm_results:
            result_lists.append(lm_results)
            rrf_weights.append(lm_weight)

        fused = rrf(*result_lists, k=60, weights=rrf_weights)

        # ---- URL priority: apply per-URL weight multipliers ---------------
        # Applied after all ranking (BM25 / LM / RRF fusion) is done; the
        # multiplied scores decide the final order rewritten below.
        url_priority: dict = cfg.get("url_priority") or {}
        fused = _apply_url_priority(fused, results, url_priority)

        # ---- Rewrite positions to influence calculate_score() -------------
        for new_pos, fused_r in enumerate(fused, start=1):
            idx_int = int(fused_r.doc_id)
            r = results[idx_int]
            # Preserve positions list length (multi-engine boost) but update values
            n_positions = len(r["positions"]) if r["positions"] else 1
            r["positions"] = [new_pos] * max(n_positions, 1)

        logger.debug(
            "Reranked %d results (bm25_weight=%.2f, lm_weight=%.2f, url_priority=%d patterns) for: %s",
            len(fused),
            bm25_weight,
            lm_weight,
            len(url_priority),
            query[:50],
        )


def _apply_url_priority(
    fused: list[SparseResult],
    results: list[t.Any],
    url_priority: t.Mapping[str, t.Any] | None,
) -> list[SparseResult]:
    """Apply per-URL weight multipliers on top of the fused ranking.

    *url_priority* maps a regex pattern (matched against each result's URL)
    to a float weight multiplier.  Every result whose URL matches a pattern
    has its fused score multiplied by the corresponding multiplier; when
    several patterns match, all multipliers are applied.  The adjusted
    ranking is re-sorted and returned, so the multipliers take effect
    *after* all other ranking (BM25 / LM / RRF fusion) is done.

    Args:
        fused: RRF-fused ranking (each ``doc_id`` indexes *results*).
        results: List of result objects (supporting ``[]`` access for
            ``"url"``).
        url_priority: Mapping of regex string -> weight multiplier, or
            ``None``/empty to leave the ranking unchanged.

    Returns:
        A new list of :class:`SparseResult` sorted by descending
        (fused score x matched URL multipliers).
    """
    if not url_priority:
        return fused

    # Compile the patterns once per request; drop invalid entries.
    patterns: list[tuple[re.Pattern[str], float]] = []
    for pattern, multiplier in url_priority.items():
        try:
            multiplier = float(multiplier)
        except (TypeError, ValueError):
            logger.warning(
                "url_priority: ignoring non-numeric multiplier %r for %r", multiplier, pattern
            )
            continue
        if multiplier <= 0:
            logger.warning(
                "url_priority: ignoring non-positive multiplier %r for %r", multiplier, pattern
            )
            continue
        try:
            patterns.append((re.compile(pattern), multiplier))
        except re.error:
            logger.warning("url_priority: ignoring invalid regex %r", pattern)

    if not patterns:
        return fused

    adjusted: list[SparseResult] = []
    for fused_r in fused:
        r = results[int(fused_r.doc_id)]
        url = _get_text(r, "url")
        multiplier = 1.0
        if url:
            for regex, m in patterns:
                if regex.search(url):
                    multiplier *= m
        adjusted.append(
            SparseResult(doc_id=fused_r.doc_id, score=fused_r.score * multiplier)
        )

    adjusted.sort(key=lambda r: r.score, reverse=True)
    return adjusted


def _get_text(result: t.Any, field: str) -> str:
    """Safely extract text field from a result object supporting [] access."""
    try:
        val = result[field]
    except (KeyError, TypeError):
        val = getattr(result, field, "")
    return val or ""


def _result_matches_query(result: t.Any, query_words: set[str]) -> bool:
    """Return whether a result title or description contains a query word."""
    if not query_words:
        return False

    title = _get_text(result, "title")
    description = _get_text(result, "description") or _get_text(result, "content")
    return any(
        token in query_words
        for token in cjk_tokenize(f"{title} {description}")
    )
