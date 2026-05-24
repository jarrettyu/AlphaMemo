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


class SSPMStrategy(SearchStrategy):
    name = "sspm"

    def __init__(
        self,
        generator: HeuristicGenerator,
        rng: np.random.Generator,
        warmup: int = 30,
        memory_weight: float = 1.0,
    ) -> None:
        self.generator = generator
        self.rng = rng
        self.warmup = warmup
        self.memory_weight = memory_weight
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
        lam = min(1.0, step / max(self.warmup, 1))
        parents = self.dag.select(k=min(max(n, 5), len(self.dag.nodes)))
        dag_scores = self.dag.scores()

        scored: list[Candidate] = []
        for parent_id, node, dag_score in parents:
            log_dag = math.log(max(float(dag_score), 1e-6))
            for motif in MOTIFS:
                vetoed, severity = self.apv.check(node.category, motif)
                if vetoed:
                    continue
                delta, conf = self.memory.query(node.category, motif)
                score = log_dag + self.memory_weight * lam * conf * delta - severity
                scored.append(
                    Candidate(
                        formula="",
                        parent_id=parent_id,
                        parent_formula=node.formula,
                        parent_quality=node.quality,
                        category=node.category,
                        motif=motif,
                        score=float(score),
                        meta={
                            "dag_score": float(dag_scores[parent_id]),
                            "lambda": lam,
                            "delta_mem": delta,
                            "confidence": conf,
                            "apv_severity": severity,
                        },
                    )
                )

        if not scored:
            return self._cold_start(n)
        scored.sort(key=lambda cand: -cand.score)
        out: list[Candidate] = []
        for base in scored[:n]:
            formula = self.generator.generate(
                GenerationRequest(
                    category=base.category,
                    motif=base.motif,
                    parent_formula=base.parent_formula,
                )
            )
            out.append(
                Candidate(
                    formula=formula,
                    parent_id=base.parent_id,
                    parent_formula=base.parent_formula,
                    parent_quality=base.parent_quality,
                    category=base.category,
                    motif=base.motif,
                    score=base.score,
                    meta=base.meta,
                )
            )
        return out

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

