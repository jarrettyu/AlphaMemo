# AlphaMemo

Official implementation of:

> AlphaMemo: Structured Search-Process Memory for Self-Evolving Alpha Mining Agents

AlphaMemo is a self-evolving formulaic alpha mining framework with:

- search-ledger guided evolution
- structured search-process memory (SSPM)
- confidence-gated residual fusion
- asymmetric process veto (APV)

## Setup

```bash
conda env create -f environment.yml
conda activate alphamemo
pip install -e .
````

## Run

```bash
bash scripts/run_main.sh
```

## Data

Experiments are based on Qlib-format CSI500 and S&P500 data.

## Citation

```bibtex
@article{alphamemo2026,
  title={AlphaMemo: Structured Search-Process Memory for Self-Evolving Alpha Mining Agents},
  author={...},
  year={2026}
}
```