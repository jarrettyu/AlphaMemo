from __future__ import annotations

import re


CATEGORIES = (
    "Momentum",
    "MeanReversion",
    "Volatility",
    "Volume",
    "Risk",
    "Intraday",
    "Trend",
    "Other",
)

MOTIFS = (
    "condition_gate",
    "rank_switch",
    "interaction_add",
    "window_rescale",
    "operator_substitute",
    "feature_swap",
    "nesting_increase",
    "temporal_shift",
    "normalization_change",
    "other",
)


def classify_formula(formula: str | None) -> str:
    if not formula:
        return "Other"
    fl = formula.lower()
    if "$volume" in fl or "volume" in fl:
        return "Volume"
    if "tsstd" in fl or "vol" in fl:
        return "Volatility"
    if "delta" in fl or "delay" in fl:
        return "Momentum"
    if "tsmean" in fl or "tsrank" in fl or "mean" in fl:
        return "MeanReversion"
    if "tsmax" in fl or "tsmin" in fl:
        return "Risk"
    if "$open" in fl or "open" in fl or "$high" in fl or "$low" in fl:
        return "Intraday"
    if "trend" in fl:
        return "Trend"
    return "Other"


def _ops(formula: str) -> set[str]:
    return set(re.findall(r"([A-Za-z][A-Za-z0-9_]*)\(", formula))


def _windows(formula: str) -> set[str]:
    return set(re.findall(r",\s*(\d+)\)", formula))


def _features(formula: str) -> set[str]:
    return set(re.findall(r"\$(?:open|close|high|low|volume|vwap)\b", formula))


def extract_edit_motif(parent: str | None, child: str | None) -> str:
    if not parent or not child:
        return "other"

    p = parent.strip()
    c = child.strip()
    pl = p.lower()
    cl = c.lower()
    p_ops = {op.lower() for op in _ops(p)}
    c_ops = {op.lower() for op in _ops(c)}
    added = c_ops - p_ops
    removed = p_ops - c_ops

    if "where" in added or ("greater" in added and "where" in c_ops):
        return "condition_gate"
    if ("csrank" in p_ops and "tsrank" in c_ops) or ("tsrank" in p_ops and "csrank" in c_ops):
        return "rank_switch"
    if _windows(p) != _windows(c) and p_ops == c_ops:
        return "window_rescale"
    if _features(p) != _features(c):
        return "feature_swap"
    if {"mul", "div", "add", "sub"} & added and len(c) > len(p) * 1.15:
        return "interaction_add"
    if {"delay", "delta"} & added:
        return "temporal_shift"
    if {"csrank", "abs", "log"} & added:
        return "normalization_change"
    if removed and added:
        return "operator_substitute"
    if cl.count("(") > pl.count("(") + 1:
        return "nesting_increase"
    return "other"

