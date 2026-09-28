"""pytest 公共夹具：离线合成数据（固定 seed，绝不联网、绝不读仓库外文件）。"""
from __future__ import annotations

import os
import sys

import pytest

# 让 tests/ 在任意工作目录下都能 import kairos_lab
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kairos_lab            # noqa: E402,F401  导入即修补兄弟包 sys.path
from kairos_lab import data as kld   # noqa: E402


@pytest.fixture(scope="session")
def equity_prices():
    """6 资产 × 300 日的合成 A 股风格价格面板（带动量结构）。"""
    return kld.synthetic_prices(n_assets=6, n_days=300, seed=7)


@pytest.fixture(scope="session")
def equity_volumes(equity_prices):
    """与 ``equity_prices`` 配套的成交量面板。"""
    return kld.synthetic_volumes(equity_prices, seed=11)


@pytest.fixture(scope="session")
def etf_prices():
    """5 资产 × 420 日的合成跨资产面板（波动差异明显，供风险平价区分）。"""
    return kld.synthetic_prices(n_assets=5, n_days=420, seed=11, prefix="C",
                                mu_low=-0.02, mu_high=0.22, vol_low=0.04,
                                vol_high=0.30, market_vol=0.12, momentum_rho=0.05)


@pytest.fixture(scope="session")
def ohlcv_frame(etf_prices):
    """由合成价格构造的单标的 OHLCV DataFrame（执行演示用）。"""
    return kld.synthetic_ohlcv(etf_prices.iloc[:, [0]], seed=5, symbol="TEST")
