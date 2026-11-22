# src/evaluation/retrieval_metrics.py
"""
Retrieval evaluation metrics.

Metrics:

- Recall@K
- Precision@K
- Hit Rate@K
- MRR
- MAP
- NDCG@K
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


@dataclass
class RetrievalMetricResult:
    metric: str
    value: Optional[float]
    evaluated_count: int
    skipped_count: int


def _identifier_from_result(result: Any) -> Optional[str]:
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        return (
            result.get("chunk_id")
            or result.get("id")
            or result.get("document_id")
        )

    return (
        getattr(result, "chunk_id", None)
        or getattr(result, "id", None)
        or getattr(result, "document_id", None)
    )


def _relevant_ids(record: Any) -> Set[str]:
    chunk_ids = set(getattr(record, "relevant_chunk_ids", []) or [])
    if chunk_ids:
        return chunk_ids

    doc_ids = set(getattr(record, "relevant_document_ids", []) or [])
    return doc_ids


def _graded_relevance(record: Any, ranked_ids: Sequence[str]) -> List[float]:
    relevance_map: Dict[str, float] = getattr(record, "relevance", {}) or {}
    relevant_ids = _relevant_ids(record)

    scores: List[float] = []
    for chunk_id in ranked_ids:
        if chunk_id in relevance_map:
            scores.append(float(relevance_map[chunk_id]))
        elif chunk_id in relevant_ids:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> Optional[float]:
    if not relevant_ids:
        return None
    top = set(ranked_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> Optional[float]:
    if k <= 0:
        return None
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = len(set(top) & relevant_ids)
    return hits / len(top)


def hit_rate_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> Optional[float]:
    if not relevant_ids:
        return None
    top = set(ranked_ids[:k])
    return 1.0 if top & relevant_ids else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> Optional[float]:
    if not relevant_ids:
        return None
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> Optional[float]:
    if not relevant_ids:
        return None

    hits = 0
    score_sum = 0.0

    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            hits += 1
            score_sum += hits / rank

    if hits == 0:
        return 0.0

    return score_sum / len(relevant_ids)


def dcg_at_k(scores: Sequence[float], k: int) -> float:
    value = 0.0
    for idx, rel in enumerate(scores[:k], start=1):
        value += ((2.0 ** float(rel)) - 1.0) / math.log2(idx + 1.0)
    return value


def ndcg_at_k(scores: Sequence[float], k: int) -> Optional[float]:
    if not scores:
        return None

    ideal_scores = sorted(scores, reverse=True)
    actual = dcg_at_k(scores, k)
    ideal = dcg_at_k(ideal_scores, k)

    if ideal == 0:
        return None

    return actual / ideal


def compute_retrieval_metrics(
    records: Iterable[Any],
    retrieval_results: Dict[str, List[Any]],
    k: int = 10,
) -> Dict[str, RetrievalMetricResult]:
    records = list(records)

    recall_values: List[float] = []
    precision_values: List[float] = []
    hit_values: List[float] = []
    mrr_values: List[float] = []
    ap_values: List[float] = []
    ndcg_values: List[float] = []

    skipped = 0

    for record in records:
        result_list = retrieval_results.get(record.id, [])
        ranked_ids = [
            identifier
            for identifier in (_identifier_from_result(item) for item in result_list)
            if identifier
        ]

        relevant_ids = _relevant_ids(record)
        if not relevant_ids:
            skipped += 1
            continue

        recall = recall_at_k(ranked_ids, relevant_ids, k)
        precision = precision_at_k(ranked_ids, relevant_ids, k)
        hit = hit_rate_at_k(ranked_ids, relevant_ids, k)
        rr = reciprocal_rank(ranked_ids, relevant_ids)
        ap = average_precision(ranked_ids, relevant_ids)

        if recall is not None:
            recall_values.append(recall)
        if precision is not None:
            precision_values.append(precision)
        if hit is not None:
            hit_values.append(hit)
        if rr is not None:
            mrr_values.append(rr)
        if ap is not None:
            ap_values.append(ap)

        graded = _graded_relevance(record, ranked_ids)
        ndcg = ndcg_at_k(graded, k)
        if ndcg is not None:
            ndcg_values.append(ndcg)

    def _mean(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    return {
        f"recall@{k}": RetrievalMetricResult(
            metric=f"recall@{k}",
            value=_mean(recall_values),
            evaluated_count=len(recall_values),
            skipped_count=skipped,
        ),
        f"precision@{k}": RetrievalMetricResult(
            metric=f"precision@{k}",
            value=_mean(precision_values),
            evaluated_count=len(precision_values),
            skipped_count=skipped,
        ),
        f"hit_rate@{k}": RetrievalMetricResult(
            metric=f"hit_rate@{k}",
            value=_mean(hit_values),
            evaluated_count=len(hit_values),
            skipped_count=skipped,
        ),
        "mrr": RetrievalMetricResult(
            metric="mrr",
            value=_mean(mrr_values),
            evaluated_count=len(mrr_values),
            skipped_count=skipped,
        ),
        "map": RetrievalMetricResult(
            metric="map",
            value=_mean(ap_values),
            evaluated_count=len(ap_values),
            skipped_count=skipped,
        ),
        f"ndcg@{k}": RetrievalMetricResult(
            metric=f"ndcg@{k}",
            value=_mean(ndcg_values),
            evaluated_count=len(ndcg_values),
            skipped_count=skipped,
        ),
    }