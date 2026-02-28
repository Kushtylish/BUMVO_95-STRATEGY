# BUMVO_95-STRATEGY
# BUMVO_95 for XAUUSD – Hybrid Trend Strategy

This repository implements the **Bootstrap‑based Utility Mean‑Variance Optimization (BUMVO_95)** framework, originally described in the paper *“Non‑Parametric Bootstrap Robust Optimization for Portfolios and Trading Strategies”* (Institute of Mathematics and Statistics). The code is adapted for **XAUUSD spot** trading using a hybrid trend‑following strategy.

## Key Features

- **Robust parameter selection** – Uses a dependent bootstrap (stationary bootstrap) to generate many plausible return paths.
- **BUMVO_95** – Chooses the lookback `L` (20,60,180,252 days) and position sizing method (`fixed` or `kelly`) that maximise the **95th percentile** of the bootstrapped Sharpe ratio distribution.
- **Meta‑optimisation of the percentile** – Automatically selects the best percentile (between 60 and 95) using a validation set.
- **Hybrid trading strategy** – Enters a position if **any** of three signals is active:
  1. Time‑series momentum (TSMOM) over `L` days.
  2. 50/200‑day EMA crossover.
  3. Keltner channel breakout (20‑day EMA ± 2×ATR).
- **Transaction costs** – 0.2% spread applied per trade.
- **Out‑of‑sample backtest** – Plots equity curve with drawdown shading, starting from $10,000.
- **Monte Carlo drawdown simulation** – 1000 bootstrapped paths to assess drawdown risk.

## Requirements

Install dependencies with:

```bash
pip install numpy pandas matplotlib arch tqdm
