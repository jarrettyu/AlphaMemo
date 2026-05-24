from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class ResidualCell:
    residuals: list[float] = field(default_factory=list)
    successes: int = 0
    failures: int = 0

    @property
    def n(self) -> int:
        return self.successes + self.failures

    @property
    def fail_rate(self) -> float:
        return self.failures / max(self.n, 1)

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.n, 1)


class ResidualMemory:
    """Online residual memory over (factor category, edit motif)."""

    def __init__(self, n_conf: int = 12) -> None:
        self.n_conf = n_conf
        self.cells: dict[tuple[str, str], ResidualCell] = defaultdict(ResidualCell)
        self.parent_bucket_baselines: dict[str, list[float]] = defaultdict(list)

    def parent_bucket(self, parent_quality: float) -> str:
        if parent_quality >= 0.06:
            return "high"
        if parent_quality >= 0.03:
            return "medium"
        return "low"

    def update(self, category: str, motif: str, child_quality: float, parent_quality: float, success: bool) -> None:
        bucket = self.parent_bucket(parent_quality)
        history = self.parent_bucket_baselines[bucket]
        baseline = sum(history) / len(history) if history else parent_quality
        residual = child_quality - baseline
        history.append(child_quality)

        cell = self.cells[(category, motif)]
        cell.residuals.append(float(residual))
        if success:
            cell.successes += 1
        else:
            cell.failures += 1

    def query(self, category: str, motif: str) -> tuple[float, float]:
        cell = self.cells.get((category, motif))
        if cell is None or cell.n < 2:
            return 0.0, 0.0
        mean_residual = sum(cell.residuals) / len(cell.residuals)
        p = (cell.successes + 1.0) / (cell.n + 2.0)
        entropy = 0.0
        if 0.0 < p < 1.0:
            entropy = -(p * math.log(p, 2) + (1 - p) * math.log(1 - p, 2))
        certainty = max(0.0, 1.0 - entropy)
        count_gate = min(1.0, cell.n / max(self.n_conf, 1))
        if len(cell.residuals) > 1:
            mean_abs = abs(mean_residual) + 1e-6
            variance_penalty = max(0.0, 1.0 - _std(cell.residuals) / (mean_abs + 0.03))
        else:
            variance_penalty = 1.0
        confidence = count_gate * (0.5 + 0.5 * certainty) * variance_penalty
        return float(mean_residual), float(max(0.0, min(1.0, confidence)))

    def top_cells(self, k: int = 8) -> list[dict]:
        rows = []
        for (category, motif), cell in self.cells.items():
            if not cell.residuals:
                continue
            delta, conf = self.query(category, motif)
            rows.append(
                {
                    "category": category,
                    "motif": motif,
                    "n": cell.n,
                    "successes": cell.successes,
                    "failures": cell.failures,
                    "mean_residual": delta,
                    "confidence": conf,
                }
            )
        return sorted(rows, key=lambda row: (row["confidence"] * row["mean_residual"]), reverse=True)[:k]

    def to_dict(self) -> dict:
        return {
            "cells": {f"{cat}/{motif}": asdict(cell) for (cat, motif), cell in self.cells.items()},
            "parent_bucket_baselines": dict(self.parent_bucket_baselines),
            "top_cells": self.top_cells(k=20),
        }


class AsymmetricAPV:
    """Veto-first memory for high-confidence failure cells."""

    def __init__(self, veto_threshold: float = 0.80, min_observations: int = 5) -> None:
        self.veto_threshold = veto_threshold
        self.min_observations = min_observations
        self.counts: dict[tuple[str, str], ResidualCell] = defaultdict(ResidualCell)

    def update(self, category: str, motif: str, success: bool) -> None:
        cell = self.counts[(category, motif)]
        if success:
            cell.successes += 1
        else:
            cell.failures += 1

    def check(self, category: str, motif: str) -> tuple[bool, float]:
        cell = self.counts.get((category, motif))
        if cell is None or cell.n < self.min_observations:
            return False, 0.0
        if cell.fail_rate >= self.veto_threshold:
            return True, cell.fail_rate
        return False, 0.5 * cell.fail_rate

    def to_dict(self) -> dict:
        return {f"{cat}/{motif}": asdict(cell) for (cat, motif), cell in self.counts.items()}


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

