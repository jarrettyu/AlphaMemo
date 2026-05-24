from __future__ import annotations

import numpy as np

from sspm.core.dag import FactorDAG
from sspm.core.motifs import CATEGORIES, MOTIFS, classify_formula, extract_edit_motif
from sspm.core.types import Candidate, EvaluationResult
from sspm.evaluation.formula_eval import FormulaEvaluator
from sspm.generation.heuristic import GenerationRequest, HeuristicGenerator

from .base import SearchStrategy


class StructuredSearchStrategy(SearchStrategy):
    """Search-ledger evolution without process-memory correction."""

    name = "structured"

    def __init__(self, generator: HeuristicGenerator, rng: np.random.Generator, name: str = "structured") -> None:
        self.name = name
        self.generator = generator
        self.rng = rng
        self.dag = FactorDAG()

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
            motif = str(self.rng.choice(MOTIFS))
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
                    score=float(dag_score),
                    meta={"dag_score": float(dag_score)},
                )
            )
        return candidates

    def update(self, candidate: Candidate, result: EvaluationResult, success: bool, step: int) -> None:
        if success and result.ok:
            category = classify_formula(candidate.formula)
            motif = extract_edit_motif(candidate.parent_formula, candidate.formula)
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
        return {"dag": self.dag.to_dict()}

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
                )
            )
        return candidates
