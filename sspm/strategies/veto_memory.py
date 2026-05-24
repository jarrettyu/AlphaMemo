from __future__ import annotations

import numpy as np

from sspm.core.dag import FactorDAG
from sspm.core.motifs import CATEGORIES, MOTIFS, classify_formula, extract_edit_motif
from sspm.core.types import Candidate, EvaluationResult
from sspm.evaluation.formula_eval import FormulaEvaluator
from sspm.generation.heuristic import GenerationRequest, HeuristicGenerator
from sspm.memory.residual import AsymmetricAPV, ResidualMemory

from .base import SearchStrategy


class VetoMemoryStrategy(SearchStrategy):
    """Structured search with memory used only as a conservative failure veto."""

    name = "veto"

    def __init__(
        self,
        generator: HeuristicGenerator,
        rng: np.random.Generator,
        warmup: int = 120,
        max_resample: int = 4,
    ) -> None:
        self.generator = generator
        self.rng = rng
        self.warmup = warmup
        self.max_resample = max_resample
        self.dag = FactorDAG()
        self.memory = ResidualMemory()
        self.apv = AsymmetricAPV()

    def initialize(self, evaluator: FormulaEvaluator, seed_formulas: list[str]) -> None:
        for formula in seed_formulas:
            result = evaluator.evaluate(formula)
            if result.ok:
                category = classify_formula(formula)
                self.dag.add(formula, result.ic, result.icir, result.quality, category, "seed", None, 0)

    def propose(self, n: int, step: int) -> list[Candidate]:
        if len(self.dag) == 0:
            return self._cold_start(n)

        parents = self.dag.select(k=min(max(n, 3), len(self.dag.nodes)))
        candidates: list[Candidate] = []
        for i in range(n):
            parent_id, node, dag_score = parents[i % len(parents)]
            motif, meta = self._choose_motif(node.category, step)
            formula = self.generator.generate(
                GenerationRequest(category=node.category, motif=motif, parent_formula=node.formula)
            )
            candidates.append(
                Candidate(
                    formula=formula,
                    parent_id=parent_id,
                    parent_formula=node.formula,
                    parent_quality=node.quality,
                    category=node.category,
                    motif=motif,
                    score=float(dag_score - meta.get("apv_severity", 0.0)),
                    meta={"dag_score": float(dag_score), **meta},
                )
            )
        return candidates

    def update(self, candidate: Candidate, result: EvaluationResult, success: bool, step: int) -> None:
        category = classify_formula(candidate.formula)
        motif = extract_edit_motif(candidate.parent_formula, candidate.formula)
        self.memory.update(category, motif, result.quality, candidate.parent_quality, success)
        self.apv.update(category, motif, success)
        if success and result.ok:
            self.dag.add(
                candidate.formula,
                result.ic,
                result.icir,
                result.quality,
                category,
                motif,
                candidate.parent_id,
                step,
            )

    def diagnostics(self) -> dict:
        return {
            "dag": self.dag.to_dict(),
            "memory": self.memory.to_dict(),
            "apv": self.apv.to_dict(),
        }

    def _choose_motif(self, category: str, step: int) -> tuple[str, dict]:
        if step < self.warmup:
            return str(self.rng.choice(MOTIFS)), {"memory_mode": "warmup", "apv_severity": 0.0}

        veto_count = 0
        last_severity = 0.0
        for _ in range(max(self.max_resample, 1)):
            motif = str(self.rng.choice(MOTIFS))
            vetoed, severity = self.apv.check(category, motif)
            last_severity = severity
            if not vetoed:
                return motif, {
                    "memory_mode": "apv_resample",
                    "apv_vetoed": False,
                    "apv_severity": float(severity),
                    "apv_resamples": veto_count,
                }
            veto_count += 1

        motif = str(self.rng.choice(MOTIFS))
        return motif, {
            "memory_mode": "apv_fallback",
            "apv_vetoed": True,
            "apv_severity": float(last_severity),
            "apv_resamples": veto_count,
        }

    def _cold_start(self, n: int) -> list[Candidate]:
        candidates: list[Candidate] = []
        for _ in range(n):
            category = str(self.rng.choice(CATEGORIES[:-1]))
            formula = self.generator.generate(GenerationRequest(category=category))
            candidates.append(
                Candidate(
                    formula=formula,
                    parent_id=None,
                    parent_formula=None,
                    parent_quality=0.0,
                    category=category,
                    motif="seed",
                    score=0.0,
                )
            )
        return candidates
