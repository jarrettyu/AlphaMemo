from __future__ import annotations

import math

import numpy as np

from sspm.core.dag import FactorDAG
from sspm.core.motifs import CATEGORIES, MOTIFS, classify_formula, extract_edit_motif
from sspm.core.types import Candidate, EvaluationResult
from sspm.evaluation.formula_eval import FormulaEvaluator
from sspm.generation.heuristic import GenerationRequest, HeuristicGenerator
from sspm.memory.residual import AsymmetricAPV, ResidualMemory

from .base import SearchStrategy


class GraphMemoryStrategy(SearchStrategy):
    """Graph-led evolution with memory as a small calibrated edit prior."""

    name = "alphamemo"

    def __init__(
        self,
        generator: HeuristicGenerator,
        rng: np.random.Generator,
        warmup: int = 120,
        memory_weight: float = 0.20,
        motif_sample_size: int = 4,
        random_motif_prob: float = 0.35,
    ) -> None:
        self.generator = generator
        self.rng = rng
        self.warmup = warmup
        self.memory_weight = memory_weight
        self.motif_sample_size = motif_sample_size
        self.random_motif_prob = random_motif_prob
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
            motif, motif_meta = self._choose_motif(node.category, step)
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
                    score=float(math.log(max(dag_score, 1e-6)) + motif_meta.get("memory_score", 0.0)),
                    meta={"dag_score": float(dag_score), **motif_meta},
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
        if step < self.warmup or float(self.rng.random()) < self.random_motif_prob:
            return str(self.rng.choice(MOTIFS)), {"memory_score": 0.0, "memory_mode": "random_or_warmup"}

        sample_size = min(self.motif_sample_size, len(MOTIFS))
        motif_pool = [str(m) for m in self.rng.choice(MOTIFS, size=sample_size, replace=False)]
        scored: list[tuple[float, str, float, float, float, bool]] = []
        for motif in motif_pool:
            vetoed, severity = self.apv.check(category, motif)
            delta, conf = self.memory.query(category, motif)
            score = -severity if vetoed else self.memory_weight * conf * delta - severity
            score += float(self.rng.normal(0.0, 0.01))
            scored.append((score, motif, delta, conf, severity, vetoed))

        score, motif, delta, conf, severity, vetoed = max(scored, key=lambda item: item[0])
        return motif, {
            "memory_score": float(score),
            "memory_delta": float(delta),
            "memory_confidence": float(conf),
            "apv_severity": float(severity),
            "apv_vetoed": bool(vetoed),
            "memory_mode": "motif_prior",
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
