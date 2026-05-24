from __future__ import annotations

import numpy as np

from sspm.core.math_utils import mean_ic_icir, rowwise_pearson, rowwise_spearman
from sspm.core.operators import evaluate_formula
from sspm.core.types import EvaluationResult

from .synthetic import MarketData


class FormulaEvaluator:
    def __init__(self, market: MarketData, min_days: int = 80) -> None:
        self.market = market
        self.min_days = min_days
        self.library_values: list[np.ndarray] = []

    def evaluate(self, formula: str) -> EvaluationResult:
        try:
            values = evaluate_formula(formula, self.market.features)
            daily_ic = rowwise_pearson(values, self.market.target)
            daily_ric = rowwise_spearman(values, self.market.target)
            ic, icir, n_days = mean_ic_icir(daily_ic)
            ric, ricir, n_rank_days = mean_ic_icir(daily_ric)
            n_days = min(n_days, n_rank_days)
            if n_days < self.min_days:
                return EvaluationResult(formula=formula, ok=False, error=f"too few valid days: {n_days}")
            max_corr = self.max_library_corr(values)
            return EvaluationResult(
                formula=formula,
                ok=True,
                ic=ic,
                icir=icir,
                ric=ric,
                ricir=ricir,
                abs_ic=abs(ic),
                abs_icir=abs(icir),
                abs_ric=abs(ric),
                abs_ricir=abs(ricir),
                n_days=n_days,
                max_corr=max_corr,
            )
        except Exception as exc:
            return EvaluationResult(formula=formula, ok=False, error=str(exc))

    def add_to_library(self, formula: str) -> None:
        try:
            self.library_values.append(evaluate_formula(formula, self.market.features))
        except Exception:
            return

    def max_library_corr(self, values: np.ndarray) -> float:
        if not self.library_values:
            return 0.0
        flat = np.nan_to_num(values.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        flat = flat - flat.mean()
        denom_a = float(np.dot(flat, flat))
        if denom_a <= 1e-12:
            return 1.0
        max_corr = 0.0
        for other in self.library_values:
            rhs = np.nan_to_num(other.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
            rhs = rhs - rhs.mean()
            denom = (denom_a * float(np.dot(rhs, rhs))) ** 0.5
            corr = abs(float(np.dot(flat, rhs) / denom)) if denom > 1e-12 else 1.0
            max_corr = max(max_corr, corr)
        return max_corr
