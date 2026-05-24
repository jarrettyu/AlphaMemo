from __future__ import annotations

from abc import ABC, abstractmethod

from sspm.core.types import Candidate, EvaluationResult
from sspm.evaluation.formula_eval import FormulaEvaluator


class SearchStrategy(ABC):
    name: str

    def initialize(self, evaluator: FormulaEvaluator, seed_formulas: list[str]) -> None:
        return None

    @abstractmethod
    def propose(self, n: int, step: int) -> list[Candidate]:
        raise NotImplementedError

    @abstractmethod
    def update(self, candidate: Candidate, result: EvaluationResult, success: bool, step: int) -> None:
        raise NotImplementedError

    def diagnostics(self) -> dict:
        return {}

