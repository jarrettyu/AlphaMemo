from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from sspm.core.motifs import CATEGORIES, MOTIFS


FEATURES = ("$close", "$open", "$high", "$low", "$volume")
WINDOWS = (5, 10, 20, 60)


@dataclass(slots=True)
class GenerationRequest:
    category: str
    motif: str = "other"
    parent_formula: str | None = None
    max_length: int = 280
    context: str = ""


class HeuristicGenerator:
    """Deterministic local stand-in for an LLM alpha generator."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def generate(self, request: GenerationRequest) -> str:
        parent = request.parent_formula
        if not parent:
            return self.random_formula(request.category)
        if len(parent) > request.max_length:
            parent = self.random_formula(request.category)
        formula = self.mutate(parent, request.motif, request.category)
        if len(formula) > request.max_length:
            formula = self.random_formula(request.category)
        return formula

    def random_formula(self, category: str | None = None) -> str:
        cat = category if category in CATEGORIES else str(self.rng.choice(CATEGORIES[:-1]))
        w = int(self.rng.choice(WINDOWS))
        w2 = int(self.rng.choice([x for x in WINDOWS if x != w]))
        feat = str(self.rng.choice(("$close", "$open", "$high", "$low")))
        templates = {
            "Momentum": [
                f"CsRank(Div(Delta($close,{w}),Add(Abs(Delay($close,{w})),1e-6)))",
                f"CsRank(Sub(TsMean($close,{w}),TsMean($close,{w2})))",
                f"CsRank(TsRank($close,{w}))",
            ],
            "MeanReversion": [
                f"CsRank(Neg(Div(Delta($close,{w}),Add(TsStd($close,{w2}),1e-6))))",
                f"CsRank(Neg(Sub($close,TsMean($close,{w}))))",
                f"CsRank(Neg(TsRank($close,{w})))",
            ],
            "Volatility": [
                f"CsRank(Neg(TsStd(Div(Delta($close,1),Add(Delay($close,1),1e-6)),{w})))",
                f"CsRank(Div(TsStd($close,{w}),Add($close,1e-6)))",
            ],
            "Volume": [
                f"CsRank(Delta(Log(Add($volume,1.0)),{w}))",
                f"CsRank(Div($volume,Add(TsMean($volume,{w}),1e-6)))",
                f"CsRank(Mul(Delta(Log(Add($volume,1.0)),{w}),Delta($close,1)))",
            ],
            "Risk": [
                f"CsRank(Neg(Div(Sub(TsMax($high,{w}),$close),Add($close,1e-6))))",
                f"CsRank(Div(Sub($close,TsMin($low,{w})),Add(Sub(TsMax($high,{w}),TsMin($low,{w})),1e-6)))",
            ],
            "Intraday": [
                "CsRank(Div(Sub($close,$open),Add(Sub($high,$low),1e-6)))",
                "CsRank(Div(Sub($high,$close),Add(Sub($high,$low),1e-6)))",
                "CsRank(Div(Sub($close,$low),Add(Sub($high,$low),1e-6)))",
            ],
            "Trend": [
                f"CsRank(Div(Sub(TsMean($close,5),TsMean($close,{w2})),Add(TsStd($close,{w2}),1e-6)))",
                f"CsRank(Sub(TsRank($close,{w2}),TsRank($volume,{w})))",
            ],
            "Other": [
                f"CsRank({feat})",
                f"CsRank(TsMean({feat},{w}))",
            ],
        }
        return str(self.rng.choice(templates.get(cat, templates["Other"])))

    def mutate(self, parent: str, motif: str, category: str) -> str:
        motif = motif if motif in MOTIFS else "other"
        if motif == "window_rescale":
            return self._window_rescale(parent)
        if motif == "feature_swap":
            return self._feature_swap(parent)
        if motif == "interaction_add":
            other = self._strip_outer_rank(self.random_formula(category))
            return f"CsRank(Mul({parent},{other}))"
        if motif == "rank_switch":
            inner = self._strip_outer_rank(parent)
            if "TsRank" in parent:
                return f"CsRank({inner})"
            return f"CsRank(TsRank({inner},{int(self.rng.choice(WINDOWS))}))"
        if motif == "operator_substitute":
            return self._operator_substitute(parent)
        if motif == "condition_gate":
            gate = self._strip_outer_rank(self.random_formula("Volume"))
            return f"Where(Greater({gate},0.5),{parent},Neg({parent}))"
        if motif == "nesting_increase":
            return f"CsRank(TsMean({parent},{int(self.rng.choice(WINDOWS))}))"
        if motif == "temporal_shift":
            return f"CsRank(Delta({parent},{int(self.rng.choice((1,5,10))) }))".replace(" ", "")
        if motif == "normalization_change":
            inner = self._strip_outer_rank(parent)
            return f"CsRank(Div({inner},Add(Abs(TsMean({inner},{int(self.rng.choice(WINDOWS))})),1e-6)))"
        return self._fallback_mutation(parent, category)

    def _fallback_mutation(self, parent: str, category: str) -> str:
        motif = str(self.rng.choice(("window_rescale", "feature_swap", "interaction_add", "operator_substitute")))
        return self.mutate(parent, motif, category)

    def _window_rescale(self, formula: str) -> str:
        windows = re.findall(r",(\d+)\)", formula)
        if not windows:
            return f"CsRank(TsMean({formula},{int(self.rng.choice(WINDOWS))}))"
        old = str(self.rng.choice(windows))
        choices = [str(w) for w in WINDOWS if str(w) != old]
        new = str(self.rng.choice(choices))
        return re.sub(rf",{old}\)", f",{new})", formula, count=1)

    def _feature_swap(self, formula: str) -> str:
        feats = re.findall(r"\$(?:open|close|high|low|volume)\b", formula)
        if not feats:
            return f"CsRank(Mul({formula},$volume))"
        old = str(self.rng.choice(feats))
        choices = [f for f in FEATURES if f != old]
        return formula.replace(old, str(self.rng.choice(choices)), 1)

    def _operator_substitute(self, formula: str) -> str:
        pairs = [
            ("TsMean", "TsStd"),
            ("TsStd", "TsMean"),
            ("TsMax", "TsMin"),
            ("TsMin", "TsMax"),
            ("Delta", "Delay"),
            ("Delay", "Delta"),
            ("Add", "Sub"),
            ("Sub", "Add"),
        ]
        self.rng.shuffle(pairs)
        for old, new in pairs:
            if old in formula:
                return formula.replace(old, new, 1)
        return f"CsRank(Abs({formula}))"

    def _strip_outer_rank(self, formula: str) -> str:
        text = formula.strip()
        if text.startswith("CsRank(") and text.endswith(")"):
            return text[len("CsRank(") : -1]
        return text
