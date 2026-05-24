from __future__ import annotations

import numpy as np

from sspm.core.motifs import CATEGORIES, MOTIFS
from sspm.core.types import Candidate, EvaluationResult
from sspm.generation.heuristic import GenerationRequest, HeuristicGenerator

from .base import SearchStrategy


class RandomSearch(SearchStrategy):
    name = "random"

    def __init__(self, generator: HeuristicGenerator, rng: np.random.Generator) -> None:
        self.generator = generator
        self.rng = rng

    def propose(self, n: int, step: int) -> list[Candidate]:
        candidates: list[Candidate] = []
        for _ in range(n):
            category = str(self.rng.choice(CATEGORIES[:-1]))
            motif = str(self.rng.choice(MOTIFS))
            formula = self.generator.generate(GenerationRequest(category=category, motif=motif))
            candidates.append(
                Candidate(
                    formula=formula,
                    parent_id=None,
                    parent_formula=None,
                    parent_quality=0.0,
                    category=category,
                    motif=motif,
                    score=0.0,
                )
            )
        return candidates

    def update(self, candidate: Candidate, result: EvaluationResult, success: bool, step: int) -> None:
        return None

