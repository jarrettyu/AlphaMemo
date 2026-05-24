from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvaluationResult:
    formula: str
    ok: bool
    ic: float = 0.0
    icir: float = 0.0
    ric: float = 0.0
    ricir: float = 0.0
    abs_ic: float = 0.0
    abs_icir: float = 0.0
    abs_ric: float = 0.0
    abs_ricir: float = 0.0
    n_days: int = 0
    error: str | None = None
    max_corr: float = 0.0

    @property
    def quality(self) -> float:
        return self.abs_icir


@dataclass(slots=True)
class Candidate:
    formula: str
    parent_id: int | None
    parent_formula: str | None
    parent_quality: float
    category: str
    motif: str
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchEvent:
    step: int
    strategy: str
    candidate: Candidate
    result: EvaluationResult
    success: bool
    n_discovered: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "strategy": self.strategy,
            "formula": self.candidate.formula,
            "parent_id": self.candidate.parent_id,
            "category": self.candidate.category,
            "motif": self.candidate.motif,
            "score": self.candidate.score,
            "ok": self.result.ok,
            "ic": self.result.ic,
            "icir": self.result.icir,
            "ric": self.result.ric,
            "ricir": self.result.ricir,
            "abs_ic": self.result.abs_ic,
            "abs_icir": self.result.abs_icir,
            "abs_ric": self.result.abs_ric,
            "abs_ricir": self.result.abs_ricir,
            "success": self.success,
            "n_discovered": self.n_discovered,
            "error": self.result.error,
            "max_corr": self.result.max_corr,
        }
