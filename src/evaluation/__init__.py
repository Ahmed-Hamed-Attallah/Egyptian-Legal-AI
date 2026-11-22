# src/evaluation/__init__.py
"""
Evaluation framework for Egyptian Legal AI.

This package measures retrieval quality, generation faithfulness, citation
accuracy, and Arabic legal query behavior using the actual application services.
"""

from src.evaluation.dataset import EvalRecord, load_dataset
from src.evaluation.evaluator import RAGEvaluator, EvaluationReport

__all__ = [
    "EvalRecord",
    "load_dataset",
    "RAGEvaluator",
    "EvaluationReport",
]