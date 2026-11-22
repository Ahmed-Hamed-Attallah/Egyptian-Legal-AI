# src/evaluation/evaluator.py
"""
RAG evaluation runner.

Uses actual application services:

- SearchService for retrieval evaluation.
- ChatService for generation/citation evaluation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.arabic_evaluation import classify_query, group_records_by_category
from src.evaluation.citation_metrics import compute_citation_metrics
from src.evaluation.dataset import EvalRecord
from src.evaluation.generation_metrics import compute_generation_metrics
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.monitoring.metrics_collector import get_monitor

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    dataset_path: str
    mode: str
    top_k: int
    generated_at: datetime = field(default_factory=datetime.utcnow)
    record_count: int = 0
    overall: Dict[str, Any] = field(default_factory=dict)
    by_category: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "mode": self.mode,
            "top_k": self.top_k,
            "generated_at": self.generated_at.isoformat(),
            "record_count": self.record_count,
            "overall": self.overall,
            "by_category": self.by_category,
            "errors": self.errors,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class RAGEvaluator:
    def __init__(
        self,
        search_service: Optional[Any] = None,
        chat_service: Optional[Any] = None,
    ) -> None:
        self.search_service = search_service
        self.chat_service = chat_service
        self.monitor = get_monitor()

    def evaluate_retrieval(
        self,
        records: List[EvalRecord],
        top_k: int = 10,
        rerank: bool = True,
    ) -> EvaluationReport:
        if self.search_service is None:
            raise RuntimeError("SearchService is required for retrieval evaluation.")

        report = EvaluationReport(
            dataset_path="",
            mode="retrieval",
            top_k=top_k,
            record_count=len(records),
        )

        retrieval_results: Dict[str, List[Any]] = {}

        for record in records:
            try:
                results = self.search_service.search(
                    query=record.question,
                    top_k=top_k,
                    use_reranking=rerank,
                    tenant_id=None,
                )
                retrieval_results[record.id] = results
            except Exception as exc:
                logger.exception("Retrieval evaluation failed for record_id=%s", record.id)
                report.errors.append(f"{record.id}: {exc}")

        overall_metrics = compute_retrieval_metrics(records, retrieval_results, k=top_k)
        report.overall = {
            name: {
                "metric": result.metric,
                "value": result.value,
                "evaluated_count": result.evaluated_count,
                "skipped_count": result.skipped_count,
            }
            for name, result in overall_metrics.items()
        }

        grouped = group_records_by_category(records)
        for category, category_records in grouped.items():
            category_results = {
                record.id: retrieval_results.get(record.id, [])
                for record in category_records
            }
            category_metrics = compute_retrieval_metrics(category_records, category_results, k=top_k)
            report.by_category[category] = {
                name: {
                    "metric": result.metric,
                    "value": result.value,
                    "evaluated_count": result.evaluated_count,
                    "skipped_count": result.skipped_count,
                }
                for name, result in category_metrics.items()
            }

        self.monitor.record_event(
            event="evaluation.retrieval.completed",
            tags={"record_count": len(records), "top_k": top_k, "rerank": rerank},
            values=report.overall,
        )

        return report

    def evaluate_generation(
        self,
        records: List[EvalRecord],
        top_k: int = 8,
    ) -> EvaluationReport:
        if self.chat_service is None:
            raise RuntimeError("ChatService is required for generation evaluation.")

        report = EvaluationReport(
            dataset_path="",
            mode="generation",
            top_k=top_k,
            record_count=len(records),
        )

        generation_outputs: Dict[str, Dict[str, Any]] = {}

        for record in records:
            try:
                result = self.chat_service.ask(
                    question=record.question,
                    tenant_id="evaluation",
                    conversation_id=None,
                    top_k=top_k,
                )
                generation_outputs[record.id] = result
            except Exception as exc:
                logger.exception("Generation evaluation failed for record_id=%s", record.id)
                report.errors.append(f"{record.id}: {exc}")

        generation_metrics = compute_generation_metrics(records, generation_outputs)
        citation_metrics = compute_citation_metrics(records, generation_outputs)

        report.overall = {
            "generation": {
                name: {
                    "metric": result.metric,
                    "value": result.value,
                    "evaluated_count": result.evaluated_count,
                    "skipped_count": result.skipped_count,
                }
                for name, result in generation_metrics.items()
            },
            "citations": {
                name: {
                    "metric": result.metric,
                    "value": result.value,
                    "evaluated_count": result.evaluated_count,
                    "skipped_count": result.skipped_count,
                }
                for name, result in citation_metrics.items()
            },
        }

        grouped = group_records_by_category(records)
        for category, category_records in grouped.items():
            category_outputs = {
                record.id: generation_outputs.get(record.id, {})
                for record in category_records
            }

            category_generation = compute_generation_metrics(category_records, category_outputs)
            category_citations = compute_citation_metrics(category_records, category_outputs)

            report.by_category[category] = {
                "generation": {
                    name: {
                        "metric": result.metric,
                        "value": result.value,
                        "evaluated_count": result.evaluated_count,
                        "skipped_count": result.skipped_count,
                    }
                    for name, result in category_generation.items()
                },
                "citations": {
                    name: {
                        "metric": result.metric,
                        "value": result.value,
                        "evaluated_count": result.evaluated_count,
                        "skipped_count": result.skipped_count,
                    }
                    for name, result in category_citations.items()
                },
            }

        self.monitor.record_event(
            event="evaluation.generation.completed",
            tags={"record_count": len(records), "top_k": top_k},
            values=report.overall,
        )

        return report

    def evaluate_full(
        self,
        records: List[EvalRecord],
        top_k: int = 10,
        rerank: bool = True,
    ) -> EvaluationReport:
        retrieval_report = self.evaluate_retrieval(records, top_k=top_k, rerank=rerank)
        generation_report = self.evaluate_generation(records, top_k=min(top_k, 8))

        combined = EvaluationReport(
            dataset_path=retrieval_report.dataset_path,
            mode="full",
            top_k=top_k,
            record_count=len(records),
            overall={
                "retrieval": retrieval_report.overall,
                "generation": generation_report.overall,
            },
            by_category={},
            errors=retrieval_report.errors + generation_report.errors,
        )

        categories = set(retrieval_report.by_category) | set(generation_report.by_category)
        for category in categories:
            combined.by_category[category] = {
                "retrieval": retrieval_report.by_category.get(category, {}),
                "generation": generation_report.by_category.get(category, {}),
            }

        return combined