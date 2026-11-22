# src/evaluation/evaluation_runner.py
"""
CLI runner for RAG evaluation.

Examples:

    python -m src.evaluation.evaluation_runner \
        --dataset data/evaluation/retrieval_sample.jsonl \
        --mode retrieval \
        --top-k 10 \
        --output artifacts/evaluation/retrieval_report.json

    python -m src.evaluation.evaluation_runner \
        --dataset data/evaluation/full_sample.jsonl \
        --mode full \
        --top-k 10 \
        --output artifacts/evaluation/full_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config.logging_config import setup_logging
from src.evaluation.dataset import load_dataset
from src.evaluation.evaluator import RAGEvaluator
from src.services.factory import build_application_services

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Egyptian Legal AI evaluation runner.")

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to evaluation dataset: .json, .jsonl, .csv, or .tsv.",
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "generation", "full"],
        default="retrieval",
        help="Evaluation mode.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top-k.")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking during retrieval evaluation.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save JSON report.",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    dataset_path = Path(args.dataset)
    records = load_dataset(dataset_path)

    if not records:
        raise SystemExit("Evaluation dataset contains no records.")

    services = build_application_services()

    evaluator = RAGEvaluator(
        search_service=services.search_service,
        chat_service=services.chat_service,
    )

    if args.mode == "retrieval":
        report = evaluator.evaluate_retrieval(
            records,
            top_k=args.top_k,
            rerank=not args.no_rerank,
        )
    elif args.mode == "generation":
        report = evaluator.evaluate_generation(
            records,
            top_k=args.top_k,
        )
    else:
        report = evaluator.evaluate_full(
            records,
            top_k=args.top_k,
            rerank=not args.no_rerank,
        )

    report.dataset_path = str(dataset_path)

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    if args.output:
        report.save(Path(args.output))
        logger.info("Evaluation report saved to %s", args.output)


if __name__ == "__main__":
    main()