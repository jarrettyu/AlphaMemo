from __future__ import annotations


SEED_FORMULAS = [
    "CsRank(Div(Delta($close,5),Add(Abs(Delay($close,5)),1e-6)))",
    "CsRank(Neg(Div(Delta($close,3),Add(TsStd($close,20),1e-6))))",
    "CsRank(Delta(Log(Add($volume,1.0)),5))",
    "CsRank(Div(Sub($close,$open),Add(Sub($high,$low),1e-6)))",
    "CsRank(Neg(TsStd(Div(Delta($close,1),Add(Delay($close,1),1e-6)),20)))",
    "CsRank(Div(Sub(TsMean($close,5),TsMean($close,20)),Add(TsStd($close,20),1e-6)))",
]


def default_seed_formulas() -> list[str]:
    return list(SEED_FORMULAS)

