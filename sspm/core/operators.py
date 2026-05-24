from __future__ import annotations

import ast
import re
from typing import Any, Callable

import numpy as np

from .math_utils import EPS, nan_to_num, rank_rows, safe_div


FEATURES = ("$open", "$close", "$high", "$low", "$volume", "$vwap", "$return")
WINDOWS = (5, 10, 20, 60)


def _as_window(w: Any) -> int:
    value = int(float(w))
    return max(1, min(value, 252))


def _rolling(x: np.ndarray, window: int, reducer: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    w = _as_window(window)
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(x.shape[0]):
        chunk = x[max(0, t - w + 1) : t + 1]
        out[t] = reducer(chunk)
    return out


def _nanmean0(z: np.ndarray) -> np.ndarray:
    finite = np.isfinite(z)
    count = finite.sum(axis=0)
    summed = np.nansum(z, axis=0)
    return np.where(count > 0, summed / np.maximum(count, 1), np.nan)


def _nanstd0(z: np.ndarray) -> np.ndarray:
    mean = _nanmean0(z)
    finite = np.isfinite(z)
    count = finite.sum(axis=0)
    centered = np.where(finite, z - mean, 0.0)
    var = np.sum(centered * centered, axis=0) / np.maximum(count, 1)
    return np.where(count > 0, np.sqrt(var), np.nan)


def _nanmax0(z: np.ndarray) -> np.ndarray:
    finite = np.isfinite(z)
    safe = np.where(finite, z, -np.inf)
    out = np.max(safe, axis=0)
    return np.where(finite.any(axis=0), out, np.nan)


def _nanmin0(z: np.ndarray) -> np.ndarray:
    finite = np.isfinite(z)
    safe = np.where(finite, z, np.inf)
    out = np.min(safe, axis=0)
    return np.where(finite.any(axis=0), out, np.nan)


def _nanmedian0(z: np.ndarray) -> np.ndarray:
    finite = np.isfinite(z)
    safe = np.where(finite, z, np.nan)
    return np.where(finite.any(axis=0), np.nanmedian(safe, axis=0), np.nan)


def TsMean(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling(x, w, _nanmean0)


def TsStd(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling(x, w, _nanstd0)


def TsMax(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling(x, w, _nanmax0)


def TsMin(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling(x, w, _nanmin0)


def TsSum(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling(x, w, lambda z: np.nansum(z, axis=0))


def TsMed(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling(x, w, _nanmedian0)


def TsMad(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling(x, w, lambda z: _nanmean0(np.abs(z - _nanmean0(z))))


def TsVar(x: np.ndarray, w: int) -> np.ndarray:
    std = TsStd(x, w)
    return std * std


def TsSkew(x: np.ndarray, w: int) -> np.ndarray:
    def reducer(z: np.ndarray) -> np.ndarray:
        mean = _nanmean0(z)
        std = _nanstd0(z)
        centered = z - mean
        return safe_div(_nanmean0(centered**3), std**3)

    return _rolling(x, w, reducer)


def TsKurt(x: np.ndarray, w: int) -> np.ndarray:
    def reducer(z: np.ndarray) -> np.ndarray:
        mean = _nanmean0(z)
        std = _nanstd0(z)
        centered = z - mean
        return safe_div(_nanmean0(centered**4), std**4)

    return _rolling(x, w, reducer)


def TsRank(x: np.ndarray, w: int) -> np.ndarray:
    window = _as_window(w)
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(x.shape[0]):
        chunk = x[max(0, t - window + 1) : t + 1]
        cur = x[t]
        finite = np.isfinite(chunk)
        counts = finite.sum(axis=0)
        leq = np.nansum(chunk <= cur, axis=0)
        out[t] = np.where(counts > 0, leq / np.maximum(counts, 1), np.nan)
    return out


def TsZScore(x: np.ndarray, w: int) -> np.ndarray:
    return Div(x - TsMean(x, w), TsStd(x, w))


def Delay(x: np.ndarray, d: int) -> np.ndarray:
    lag = _as_window(d)
    out = np.full_like(x, np.nan, dtype=float)
    out[lag:] = x[:-lag]
    return out


def Delta(x: np.ndarray, d: int) -> np.ndarray:
    return x - Delay(x, d)


def TsDelta(x: np.ndarray, d: int) -> np.ndarray:
    return Delta(x, d)


def TsPctChange(x: np.ndarray, d: int) -> np.ndarray:
    return Div(x, Delay(x, d)) - 1.0


def TsDiv(x: np.ndarray, d: int) -> np.ndarray:
    return Div(x, Delay(x, d))


def TsIr(x: np.ndarray, w: int) -> np.ndarray:
    return Div(TsMean(x, w), TsStd(x, w))


def TsMinMaxDiff(x: np.ndarray, w: int) -> np.ndarray:
    return TsMax(x, w) - TsMin(x, w)


def TsMaxDiff(x: np.ndarray, w: int) -> np.ndarray:
    return TsMax(x, w) - x


def TsMinDiff(x: np.ndarray, w: int) -> np.ndarray:
    return x - TsMin(x, w)


def TsWMA(x: np.ndarray, w: int) -> np.ndarray:
    window = _as_window(w)

    def reducer(z: np.ndarray) -> np.ndarray:
        weights = np.arange(1, z.shape[0] + 1, dtype=float).reshape(-1, 1)
        finite = np.isfinite(z)
        weighted = np.where(finite, z * weights, 0.0)
        denom = np.where(finite, weights, 0.0).sum(axis=0)
        return np.where(denom > 0, weighted.sum(axis=0) / denom, np.nan)

    return _rolling(x, window, reducer)


def TsEMA(x: np.ndarray, w: int) -> np.ndarray:
    window = _as_window(w)
    alpha = 2.0 / (window + 1.0)
    out = np.full_like(x, np.nan, dtype=float)
    prev = np.full(x.shape[1], np.nan, dtype=float)
    for t in range(x.shape[0]):
        cur = x[t]
        prev = np.where(np.isfinite(prev), alpha * cur + (1.0 - alpha) * prev, cur)
        out[t] = prev
    return out


def TsCov(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    window = _as_window(w)
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(x.shape[0]):
        xs = x[max(0, t - window + 1) : t + 1]
        ys = y[max(0, t - window + 1) : t + 1]
        finite = np.isfinite(xs) & np.isfinite(ys)
        count = finite.sum(axis=0)
        x_mean = np.where(count > 0, np.where(finite, xs, 0.0).sum(axis=0) / np.maximum(count, 1), np.nan)
        y_mean = np.where(count > 0, np.where(finite, ys, 0.0).sum(axis=0) / np.maximum(count, 1), np.nan)
        cov = np.where(finite, (xs - x_mean) * (ys - y_mean), 0.0).sum(axis=0) / np.maximum(count, 1)
        out[t] = np.where(count > 0, cov, np.nan)
    return out


def TsCorr(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    return safe_div(TsCov(x, y, w), TsStd(x, w) * TsStd(y, w))


def CsRank(x: np.ndarray) -> np.ndarray:
    return rank_rows(x)


def CsZScore(x: np.ndarray) -> np.ndarray:
    row_mean = np.nanmean(x, axis=1, keepdims=True)
    row_std = np.nanstd(x, axis=1, keepdims=True)
    return safe_div(x - row_mean, row_std)


def Add(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.asarray(a) + np.asarray(b)


def Sub(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.asarray(a) - np.asarray(b)


def Mul(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.asarray(a) * np.asarray(b)


def Div(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return safe_div(np.asarray(a), np.asarray(b))


def Neg(x: np.ndarray | float) -> np.ndarray:
    return -np.asarray(x)


def Abs(x: np.ndarray | float) -> np.ndarray:
    return np.abs(np.asarray(x))


def Sign(x: np.ndarray | float) -> np.ndarray:
    return np.sign(np.asarray(x))


def Inv(x: np.ndarray | float) -> np.ndarray:
    return safe_div(1.0, np.asarray(x))


def Log(x: np.ndarray | float) -> np.ndarray:
    return np.log(np.abs(np.asarray(x)) + EPS)


def SLog1p(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x)
    return np.sign(arr) * np.log1p(np.abs(arr))


def Pow(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    base = np.asarray(a)
    exp = np.asarray(b)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return np.sign(base) * (np.abs(base) + EPS) ** exp


def Greater(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return (np.asarray(a) > np.asarray(b)).astype(float)


def Less(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return (np.asarray(a) < np.asarray(b)).astype(float)


def GetGreater(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.maximum(np.asarray(a), np.asarray(b))


def GetLess(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.minimum(np.asarray(a), np.asarray(b))


def Where(cond: np.ndarray | float, a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.where(np.asarray(cond) > 0, np.asarray(a), np.asarray(b))


ALLOWED_FUNCS: dict[str, Callable[..., Any]] = {
    name: obj
    for name, obj in {
        "TsMean": TsMean,
        "TsStd": TsStd,
        "TsMax": TsMax,
        "TsMin": TsMin,
        "TsSum": TsSum,
        "TsMed": TsMed,
        "TsMad": TsMad,
        "TsVar": TsVar,
        "TsSkew": TsSkew,
        "TsKurt": TsKurt,
        "TsRank": TsRank,
        "TsZScore": TsZScore,
        "TsDelta": TsDelta,
        "TsPctChange": TsPctChange,
        "TsDiv": TsDiv,
        "TsIr": TsIr,
        "TsMinMaxDiff": TsMinMaxDiff,
        "TsMaxDiff": TsMaxDiff,
        "TsMinDiff": TsMinDiff,
        "TsWMA": TsWMA,
        "TsEMA": TsEMA,
        "TsCov": TsCov,
        "TsCorr": TsCorr,
        "Delay": Delay,
        "Ref": Delay,
        "Delta": Delta,
        "CsRank": CsRank,
        "CsZScore": CsZScore,
        "Rank": CsRank,
        "ZScore": CsZScore,
        "Mean": TsMean,
        "Std": TsStd,
        "Sum": TsSum,
        "Mad": TsMad,
        "Add": Add,
        "Sub": Sub,
        "Mul": Mul,
        "Div": Div,
        "Neg": Neg,
        "Abs": Abs,
        "Sign": Sign,
        "Inv": Inv,
        "Log": Log,
        "SLog1p": SLog1p,
        "Pow": Pow,
        "Greater": Greater,
        "Less": Less,
        "GetGreater": GetGreater,
        "GetLess": GetLess,
        "Where": Where,
        "TS_MEAN": TsMean,
        "TS_STD": TsStd,
        "TS_MAX": TsMax,
        "TS_MIN": TsMin,
        "TS_SUM": TsSum,
        "TS_MEDIAN": TsMed,
        "TS_MAD": TsMad,
        "TS_VAR": TsVar,
        "TS_RANK": TsRank,
        "TS_ZSCORE": TsZScore,
        "TS_DELTA": TsDelta,
        "TS_PCTCHANGE": TsPctChange,
        "TS_COVARIANCE": TsCov,
        "TS_CORR": TsCorr,
        "DELAY": Delay,
        "DELTA": Delta,
        "RANK": CsRank,
        "ZSCORE": CsZScore,
        "ABS": Abs,
        "SIGN": Sign,
        "INV": Inv,
        "LOG": Log,
        "POW": Pow,
        "MAX": GetGreater,
        "MIN": GetLess,
    }.items()
}


_FEATURE_NAME = {
    "$open": "open_",
    "$close": "close",
    "$high": "high",
    "$low": "low",
    "$volume": "volume",
    "$vwap": "vwap",
    "$return": "ret",
}
_NAME_FEATURE = {v: k for k, v in _FEATURE_NAME.items()}


def normalize_formula(formula: str) -> str:
    text = formula.strip().replace("```", "").strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<=\d)d\b", "", text)
    for src, dst in _FEATURE_NAME.items():
        text = text.replace(src, dst)
    return text


def display_formula(formula: str) -> str:
    text = formula
    for src, dst in _NAME_FEATURE.items():
        text = re.sub(rf"\b{src}\b", dst, text)
    return text


class FormulaSafetyError(ValueError):
    pass


class _SafetyVisitor(ast.NodeVisitor):
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.USub,
        ast.UAdd,
    )

    def __init__(self, allowed_names: set[str]) -> None:
        self.allowed_names = allowed_names

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, self.allowed_nodes):
            raise FormulaSafetyError(f"disallowed syntax: {node.__class__.__name__}")
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
            raise FormulaSafetyError("disallowed function call")
        if node.keywords:
            raise FormulaSafetyError("keyword arguments are not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.allowed_names:
            raise FormulaSafetyError(f"unknown name: {node.id}")


def evaluate_formula(formula: str, features: dict[str, np.ndarray]) -> np.ndarray:
    expr = normalize_formula(formula)
    allowed_names = set(ALLOWED_FUNCS) | set(features)
    tree = ast.parse(expr, mode="eval")
    _SafetyVisitor(allowed_names).visit(tree)
    env: dict[str, Any] = dict(ALLOWED_FUNCS)
    env.update(features)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        value = eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, env)
    arr = np.asarray(value, dtype=float)
    if arr.shape != features["close"].shape:
        raise ValueError(f"formula returned shape {arr.shape}, expected {features['close'].shape}")
    return nan_to_num(arr)
