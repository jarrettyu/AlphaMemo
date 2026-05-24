from __future__ import annotations

import numpy as np

from sspm.core.motifs import CATEGORIES, MOTIFS
from sspm.core.types import Candidate, EvaluationResult
from sspm.evaluation.formula_eval import FormulaEvaluator
from sspm.generation.heuristic import GenerationRequest, HeuristicGenerator

from .base import SearchStrategy


class GeneticProgrammingStrategy(SearchStrategy):
    """Population-based formula baseline with mutation and crossover."""

    name = "gp"

    def __init__(
        self,
        generator: HeuristicGenerator,
        rng: np.random.Generator,
        population_size: int = 128,
        crossover_rate: float = 0.35,
        random_rate: float = 0.20,
    ) -> None:
        self.generator = generator
        self.rng = rng
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.random_rate = random_rate
        self.population: list[dict] = []
        self.n_crossover = 0
        self.n_mutation = 0
        self.n_random = 0

    def initialize(self, evaluator: FormulaEvaluator, seed_formulas: list[str]) -> None:
        self.population = [
            {"formula": formula, "quality": 0.0, "category": "Other"}
            for formula in seed_formulas
            if formula.strip()
        ]

    def propose(self, n: int, step: int) -> list[Candidate]:
        candidates: list[Candidate] = []
        for _ in range(n):
            category = str(self.rng.choice(CATEGORIES[:-1]))
            motif = str(self.rng.choice(MOTIFS))
            mode = self._mode()
            if mode == "random" or not self.population:
                formula = self.generator.generate(GenerationRequest(category=category, motif=motif))
                parent_id = None
                parent_formula = None
                parent_quality = 0.0
                self.n_random += 1
            elif mode == "crossover" and len(self.population) >= 2:
                p1_id, p1 = self._select_parent()
                p2_id, p2 = self._select_parent(exclude=p1_id)
                formula = self._crossover(p1["formula"], p2["formula"])
                parent_id = p1_id
                parent_formula = p1["formula"]
                parent_quality = float(p1["quality"])
                self.n_crossover += 1
            else:
                parent_id, parent = self._select_parent()
                formula = self.generator.generate(
                    GenerationRequest(
                        category=str(parent.get("category", category)),
                        motif=motif,
                        parent_formula=str(parent["formula"]),
                    )
                )
                parent_formula = str(parent["formula"])
                parent_quality = float(parent["quality"])
                self.n_mutation += 1

            candidates.append(
                Candidate(
                    formula=formula,
                    parent_id=parent_id,
                    parent_formula=parent_formula,
                    parent_quality=parent_quality,
                    category=category,
                    motif=motif,
                    score=parent_quality,
                    meta={"mode": mode, "step": step},
                )
            )
        return candidates

    def update(self, candidate: Candidate, result: EvaluationResult, success: bool, step: int) -> None:
        if not result.ok:
            return
        quality = float(result.abs_icir + 0.25 * result.abs_ic)
        self.population.append(
            {
                "formula": candidate.formula,
                "quality": quality,
                "category": candidate.category,
                "step": step,
                "success": success,
            }
        )
        self.population.sort(key=lambda row: float(row["quality"]), reverse=True)
        del self.population[self.population_size :]

    def diagnostics(self) -> dict:
        best_quality = max((float(row["quality"]) for row in self.population), default=0.0)
        return {
            "population_size": len(self.population),
            "best_quality": best_quality,
            "n_crossover": self.n_crossover,
            "n_mutation": self.n_mutation,
            "n_random": self.n_random,
        }

    def _mode(self) -> str:
        roll = float(self.rng.random())
        if roll < self.random_rate:
            return "random"
        if roll < self.random_rate + self.crossover_rate:
            return "crossover"
        return "mutation"

    def _select_parent(self, exclude: int | None = None) -> tuple[int, dict]:
        choices = [i for i in range(len(self.population)) if i != exclude]
        if not choices:
            idx = 0
            return idx, self.population[idx]
        k = min(4, len(choices))
        sampled = self.rng.choice(choices, size=k, replace=False)
        idx = int(max(sampled, key=lambda i: float(self.population[int(i)]["quality"])))
        return idx, self.population[idx]

    def _crossover(self, left: str, right: str) -> str:
        left_inner = self._strip_rank(left)
        right_inner = self._strip_rank(right)
        op = str(self.rng.choice(("Add", "Sub", "Mul")))
        if op == "Sub" and float(self.rng.random()) < 0.5:
            left_inner, right_inner = right_inner, left_inner
        formula = f"CsRank({op}({left_inner},{right_inner}))"
        if len(formula) > 280:
            parent = left if float(self.rng.random()) < 0.5 else right
            formula = self.generator.generate(
                GenerationRequest(category="Other", motif="operator_substitute", parent_formula=parent)
            )
        return formula

    @staticmethod
    def _strip_rank(formula: str) -> str:
        text = formula.strip()
        if text.startswith("CsRank(") and text.endswith(")"):
            return text[len("CsRank(") : -1]
        return text
