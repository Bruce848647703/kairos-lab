"""Kairos Lab 的三条端到端流水线。

- :func:`run_equity_pipeline`      A 股：数据 → 因子 → 组合 → 回测 → 风险
- :func:`run_multi_asset_pipeline` 跨资产 ETF：风险平价全天候 → 回测 → 风险
- :func:`run_execution_demo`       执行：父单 → TWAP/VWAP 切片 → 模拟撮合 → TCA

导入本包会连带 import 所需的兄弟包（通过 :mod:`kairos_lab._bootstrap` 定位），
缺少任一兄弟包时给出带安装指引的中文 ``ImportError``。
"""
from __future__ import annotations

from .e2e_equity import (
    backtest_stage,
    build_report,
    factor_stage,
    factor_weights,
    momentum_factor,
    optimized_weights,
    portfolio_stage,
    risk_parity_chain,
    risk_stage,
    run_equity_pipeline,
    solve_weights,
    tradable_assets,
)
from .e2e_equity import summarize as summarize_equity
from .e2e_multi_asset import (
    aggregate_by_class,
    inverse_volatility_weights,
    rolling_inverse_vol,
    rolling_weights,
    run_multi_asset_pipeline,
    static_weights,
)
from .e2e_multi_asset import summarize as summarize_multi_asset
from .execution_demo import (
    execute_once,
    make_algo,
    run_execution_demo,
    tca_frame,
    to_bar_list,
)
from .execution_demo import summarize as summarize_execution

__all__ = [
    # equity
    "run_equity_pipeline", "summarize_equity", "factor_stage", "portfolio_stage",
    "backtest_stage", "risk_stage", "momentum_factor", "factor_weights",
    "optimized_weights", "solve_weights", "risk_parity_chain", "tradable_assets",
    "build_report",
    # multi asset
    "run_multi_asset_pipeline", "summarize_multi_asset", "rolling_weights",
    "rolling_inverse_vol", "inverse_volatility_weights", "static_weights",
    "aggregate_by_class",
    # execution
    "run_execution_demo", "summarize_execution", "execute_once", "to_bar_list",
    "make_algo", "tca_frame",
]
