"""数据入口：读取真实行情面板，或生成离线可复现的合成面板。

两类数据来源：

1. **真实数据**（示例默认）：``kairos-data`` 仓库已提交的 CSV 数据集，
   ``data/ashare/``（38 只 A 股前复权日线）与 ``data/etf/``（9 只跨资产 ETF）。
   本模块用 :func:`default_data_dir` 相对定位该目录，用
   ``kairos_data.load_ashare_panel`` 读出对齐的 (prices, volumes) 面板；
   目录缺失且 ``fetch=True`` 时才调用 ``kairos_data.ashare.fetch_universe`` 联网抓取。
2. **合成数据**（测试/演示默认）：:func:`synthetic_prices` 用固定 seed 生成带
   「市场因子 + AR(1) 动量结构」的价格面板，离线、确定性、可复现。

设计原则：测试链路永不联网；真实数据缺失时给出清晰的中文提示而不是静默失败。
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._bootstrap import kairos_root, require

#: 数据集名称 -> 相对 kairos-data 仓库的子目录
DATASETS: Dict[str, str] = {"ashare": "ashare", "etf": "etf"}


# ---------------------------------------------------------------------------
# 真实数据
# ---------------------------------------------------------------------------
def default_data_dir(dataset: str = "ashare") -> str:
    """返回 ``kairos-data`` 仓库中某数据集的默认目录（可能不存在）。

    相对定位：``<kairos_root>/kairos-data/data/<dataset>``。
    """
    if dataset not in DATASETS:
        raise ValueError(f"未知数据集: {dataset!r}，可选 {sorted(DATASETS)}")
    return os.path.join(kairos_root(), "kairos-data", "data", DATASETS[dataset])


def has_local_data(data_dir: Optional[str] = None, dataset: str = "ashare") -> bool:
    """判断本地数据目录是否存在且至少含一个 CSV。"""
    d = data_dir or default_data_dir(dataset)
    if not os.path.isdir(d):
        return False
    return any(f.endswith(".csv") for f in os.listdir(d))


def universe_symbols(dataset: str = "ashare") -> List[str]:
    """返回某数据集对应的官方股票/ETF 池（来自 ``kairos_data.universe``）。"""
    kd = require("kairos_data")
    if dataset == "etf":
        return list(kd.etf_symbols())
    return list(kd.symbols())


def fetch_dataset(dataset: str = "ashare", out_dir: Optional[str] = None,
                  start: str = "2016-01-01", delay: float = 0.3) -> Dict[str, int]:
    """联网抓取某数据集的全部标的到 ``out_dir``，返回 ``{symbol: 行数}``。"""
    kd = require("kairos_data")
    out = out_dir or default_data_dir(dataset)
    syms = universe_symbols(dataset)
    return kd.ashare.fetch_universe(syms, out, start=start, adjust="qfq", delay=delay)


def load_panel(data_dir: Optional[str] = None, dataset: str = "ashare",
               fetch: bool = False, start: Optional[str] = None,
               end: Optional[str] = None, max_assets: Optional[int] = None,
               max_days: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读出对齐的 (prices, volumes) 面板。

    参数
    ----
    data_dir:  CSV 目录；省略时用 :func:`default_data_dir`。
    dataset:   ``"ashare"`` 或 ``"etf"``，仅用于推断默认目录与抓取清单。
    fetch:     目录缺失/为空时是否联网抓取（默认 False，测试必须保持 False）。
    start/end: 可选的日期窗口裁剪（含端点）。
    max_assets: 可选，只保留前 N 个标的（按代码排序），用于快速试跑。
    max_days:  可选，只保留最近 N 个交易日。

    返回 ``(prices, volumes)``：index=DatetimeIndex，columns=symbol。
    """
    d = data_dir or default_data_dir(dataset)
    if not has_local_data(d, dataset):
        if not fetch:
            raise FileNotFoundError(
                f"未找到本地行情数据: {d}\n"
                f"请指定 --data-dir，或把 kairos-data 仓库放到 {kairos_root()}/ 下，"
                f"或用 fetch=True（联网）抓取。"
            )
        fetch_dataset(dataset, d)
    kd = require("kairos_data")
    prices, volumes = kd.load_ashare_panel(d)
    if max_assets is not None and prices.shape[1] > max_assets:
        cols = list(prices.columns)[: int(max_assets)]
        prices, volumes = prices[cols], volumes[cols]
    if start is not None or end is not None:
        prices = prices.loc[start:end]
        volumes = volumes.reindex(prices.index)
    if max_days is not None and len(prices) > max_days:
        prices = prices.iloc[-int(max_days):]
        volumes = volumes.reindex(prices.index)
    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="all")
    volumes = volumes.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    return prices, volumes


def ohlcv_frame(data_dir: Optional[str], symbol: str, dataset: str = "ashare",
                last: Optional[int] = None) -> pd.DataFrame:
    """读出单个标的的完整 OHLCV DataFrame（执行演示需要开高低收与成交量）。

    用 ``kairos_data.CSVSource(root, datetime_col="date")`` 读取；``last`` 非空时
    只返回最后 N 行。
    """
    d = data_dir or default_data_dir(dataset)
    kd = require("kairos_data")
    src = kd.CSVSource(d, datetime_col="date")
    df = src.load(symbol)
    df = df.dropna(subset=[c for c in ("open", "high", "low", "close", "volume")
                           if c in df.columns])
    if last is not None and len(df) > last:
        df = df.iloc[-int(last):]
    return df


# ---------------------------------------------------------------------------
# 合成数据（离线、固定 seed）
# ---------------------------------------------------------------------------
def synthetic_prices(n_assets: int = 6, n_days: int = 300, seed: int = 7,
                     start: str = "2020-01-02", s0: float = 50.0,
                     mu_low: float = -0.10, mu_high: float = 0.60,
                     vol_low: float = 0.10, vol_high: float = 0.22,
                     market_vol: float = 0.13, market_beta: float = 0.85,
                     momentum_rho: float = 0.10, prefix: str = "A") -> pd.DataFrame:
    """生成带「市场因子 + AR(1) 动量结构」的合成价格面板（可复现）。

    收益结构（均为年化参数，按 252 交易日折算到每日）::

        r_it = mu_i/252 + beta_i·m_t + sigma_i/√252·e_it
        m_t  = N(0, market_vol/√252)                        # 共同市场因子
        e_it = rho·e_i,t-1 + √(1−rho²)·N(0,1)               # AR(1) 特质冲击

    ``momentum_rho > 0`` 使特质冲击具备持续性，从而让「过去 lookback 期收益」
    这类动量因子对**其后**的收益具备正的预测力，便于端到端流水线在合成数据上
    也能跑出有意义的 IC。资产按 ``mu`` / ``sigma`` 线性铺开，形成波动与收益的
    横截面差异（风险平价/优化器才有区分度）。

    返回：价格面板 DataFrame（index=工作日，columns=``A00..``），起点约为 ``s0``。
    """
    if n_assets <= 0 or n_days <= 1:
        raise ValueError("n_assets 必须为正且 n_days 必须大于 1")
    rng = np.random.default_rng(seed)
    n, t = int(n_assets), int(n_days)

    mu = np.linspace(mu_low, mu_high, n)
    sigma = np.linspace(vol_low, vol_high, n)
    beta = np.full(n, market_beta) * np.linspace(0.7, 1.3, n)

    idx = pd.bdate_range(start=start, periods=t)
    dt = 1.0 / 252.0

    market = rng.standard_normal(t) * market_vol * math.sqrt(dt)
    eps = rng.standard_normal((t, n))
    if momentum_rho:
        rho = float(momentum_rho)
        scale = math.sqrt(max(1.0 - rho * rho, 1e-12))
        for i in range(1, t):
            eps[i] = rho * eps[i - 1] + scale * eps[i]

    idio = eps * (sigma * math.sqrt(dt))[None, :]
    rets = (mu * dt)[None, :] + beta[None, :] * market[:, None] + idio
    rets[0] = 0.0                                  # 首行作为基准日，无收益
    prices = s0 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"{prefix}{i:02d}" for i in range(n)]
    return pd.DataFrame(prices, index=idx, columns=cols)


def synthetic_volumes(prices: pd.DataFrame, seed: int = 11,
                      base: float = 2.0e6, u_shape: float = 0.5) -> pd.DataFrame:
    """为合成价格面板配一份成交量面板（对数正态噪声，仅用于展示/参与率约束）。"""
    rng = np.random.default_rng(seed)
    t, n = prices.shape
    noise = np.exp(rng.standard_normal((t, n)) * 0.35)
    drift = np.linspace(1.0, 1.0 + u_shape, t)[:, None]
    return pd.DataFrame(base * noise * drift, index=prices.index, columns=prices.columns)


def synthetic_ohlcv(prices: pd.DataFrame, volumes: Optional[pd.DataFrame] = None,
                    seed: int = 5, intraday: float = 0.012,
                    symbol: str = "SIM") -> pd.DataFrame:
    """由收盘价面板（单列或取第一列）构造单标的 OHLCV DataFrame。

    开/高/低围绕收盘价构造：开盘 = 上一收盘加微小跳空，高低价在开收之外
    再扩 ``intraday`` 幅度，保证 ``low <= open/close <= high`` 恒成立。
    供执行演示在「只有价格序列」时也能跑通 TWAP/VWAP。
    """
    close = prices.iloc[:, 0] if isinstance(prices, pd.DataFrame) else pd.Series(prices)
    close = close.astype("float64")
    rng = np.random.default_rng(seed)
    t = len(close)
    prev = np.concatenate([[float(close.iloc[0])], close.to_numpy()[:-1]])
    gap = rng.standard_normal(t) * intraday * 0.4
    open_ = prev * np.exp(gap)
    hi = np.maximum(open_, close.to_numpy()) * (1.0 + np.abs(rng.standard_normal(t)) * intraday)
    lo = np.minimum(open_, close.to_numpy()) * (1.0 - np.abs(rng.standard_normal(t)) * intraday)
    if volumes is not None:
        vol = (volumes.iloc[:, 0] if isinstance(volumes, pd.DataFrame) else volumes)
        vol = pd.Series(vol).reindex(close.index).fillna(0.0).astype("float64")
    else:
        vol = pd.Series(1.0e6 * np.exp(rng.standard_normal(t) * 0.3), index=close.index)
    out = pd.DataFrame({"symbol": symbol, "open": open_, "high": hi, "low": lo,
                        "close": close.to_numpy(), "volume": vol.to_numpy()},
                       index=close.index)
    return out


def describe_panel(prices: pd.DataFrame, volumes: Optional[pd.DataFrame] = None,
                   names: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """把面板的基本信息汇总成字典（用于报告头部）。"""
    info: Dict[str, object] = {
        "n_assets": int(prices.shape[1]),
        "n_days": int(prices.shape[0]),
        "start": str(pd.Timestamp(prices.index[0]).date()),
        "end": str(pd.Timestamp(prices.index[-1]).date()),
        "assets": list(map(str, prices.columns)),
    }
    if volumes is not None:
        info["total_volume"] = float(np.nansum(volumes.to_numpy(dtype="float64")))
    if names:
        info["names"] = list(map(str, names))
    return info


# ---------------------------------------------------------------------------
# 示例脚本用的「三级回退」解析：本地真实数据 → 联网抓取 → 合成数据
# ---------------------------------------------------------------------------
def resolve_panel(dataset: str = "ashare", data_dir: Optional[str] = None,
                  fetch: bool = True, fallback_synthetic: bool = True,
                  start: Optional[str] = None, end: Optional[str] = None,
                  max_assets: Optional[int] = None, max_days: Optional[int] = None,
                  n_assets: int = 8, n_days: int = 400, seed: int = 7,
                  verbose: bool = True
                  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str]:
    """按「本地 CSV → 联网抓取 → 合成面板」的顺序解析价格面板。

    返回 ``(prices, volumes, source_description)``；``source_description`` 直接写进
    报告头部，因此产物永远能看出数字来自真实数据还是合成数据。

    - 本地目录有数据：直接读（**不联网**）；
    - 本地缺失且 ``fetch=True``：调用 ``kairos_data.ashare.fetch_universe`` 抓取；
    - 抓取失败（离线环境）且 ``fallback_synthetic=True``：退化为
      :func:`synthetic_prices`，并打印明确警告。
    """
    d = data_dir or default_data_dir(dataset)
    label = os.path.relpath(d, os.path.dirname(kairos_root())) if os.path.isdir(d) else d
    if has_local_data(d, dataset):
        prices, volumes = load_panel(d, dataset, fetch=False, start=start, end=end,
                                     max_assets=max_assets, max_days=max_days)
        syms = list(map(str, prices.columns))
        if verbose:
            print(f"[data] 读取本地真实行情: {d} ({len(syms)} 个标的, {len(prices)} 个交易日)")
        return prices, volumes, f"{label} ({len(syms)} 个标的真实日线)"

    if fetch:
        try:
            if verbose:
                print(f"[data] 本地无数据，联网抓取 {dataset} 到 {d} ...")
            fetch_dataset(dataset, d, start=start or "2016-01-01")
            prices, volumes = load_panel(d, dataset, fetch=False, start=start, end=end,
                                         max_assets=max_assets, max_days=max_days)
            return prices, volumes, f"{label} ({prices.shape[1]} 个标的真实日线，本次联网抓取)"
        except Exception as exc:  # noqa: BLE001 - 离线环境需要优雅降级
            if not fallback_synthetic:
                raise
            if verbose:
                print(f"[data] 抓取失败（{type(exc).__name__}: {exc}），回退合成数据")
    elif not fallback_synthetic:
        raise FileNotFoundError(f"未找到本地行情数据: {d}（且 fetch=False）")

    prices = synthetic_prices(n_assets=n_assets, n_days=n_days, seed=seed)
    volumes = synthetic_volumes(prices, seed=seed + 1)
    if verbose:
        print(f"[data] 使用合成面板: {prices.shape[1]} 资产 × {prices.shape[0]} 日 (seed={seed})")
    return prices, volumes, f"synthetic({prices.shape[1]}x{prices.shape[0]}, seed={seed})"


def resolve_ohlcv(dataset: str = "ashare", symbol: Optional[str] = None,
                  data_dir: Optional[str] = None, last: int = 60, fetch: bool = True,
                  fallback_synthetic: bool = True, n_bars: int = 60, seed: int = 7,
                  verbose: bool = True) -> Tuple[pd.DataFrame, str, str]:
    """解析单标的 OHLCV DataFrame（执行演示用）。

    返回 ``(ohlcv, symbol, source_description)``。``symbol`` 省略时取该数据集的第一个标的。
    本地缺失时按需联网抓取，再失败则用 :func:`synthetic_ohlcv` 兜底。
    """
    d = data_dir or default_data_dir(dataset)
    syms = universe_symbols(dataset)
    sym = symbol or (syms[0] if syms else "SIM")
    if not has_local_data(d, dataset) and fetch:
        try:
            if verbose:
                print(f"[data] 本地无数据，联网抓取 {dataset} 到 {d} ...")
            fetch_dataset(dataset, d)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[data] 抓取失败（{type(exc).__name__}: {exc}）")
    if has_local_data(d, dataset):
        try:
            df = ohlcv_frame(d, sym, dataset=dataset, last=last)
            if len(df) >= 5:
                if verbose:
                    print(f"[data] 读取本地真实 OHLCV: {os.path.join(d, sym + '.csv')}"
                          f"（最近 {len(df)} 根 bar）")
                return df, sym, f"{dataset}/{sym}.csv 真实日线（最近 {len(df)} 根 bar）"
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[data] 读取 {sym} 失败（{type(exc).__name__}: {exc}），回退合成行情")
    if not fallback_synthetic:
        raise FileNotFoundError(f"未找到 {sym} 的 OHLCV 数据: {d}")
    prices = synthetic_prices(n_assets=1, n_days=n_bars, seed=seed, prefix=sym[:3] or "S")
    df = synthetic_ohlcv(prices, seed=seed + 1, symbol=sym)
    if verbose:
        print(f"[data] 使用合成 OHLCV: {len(df)} 根 bar (seed={seed})")
    return df, sym, f"synthetic OHLCV({sym}, {len(df)} bars, seed={seed})"
