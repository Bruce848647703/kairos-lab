"""端到端流水线 ①：A 股「数据 → 因子 → 组合 → 回测 → 风险」。

这是 Kairos Lab 的主线流水线，用**真实 A 股日线**把五个兄弟包串成一条完整的
研究链路，每一阶段的产物都是下一阶段的输入：

::

    prices/volumes (kairos_data)
        │  pct_change(lookback) 构造动量因子
        ▼
    预处理 winsorize + zscore (kairos_factor.FactorPipeline)
        │  forward_returns / rank_ic / ic_summary / quantile_returns / FactorReport
        ▼
    因子评价（IC、IR、t 值、分层收益、IC 衰减、覆盖率、换手）
        │  分数最高的 top_quantile → 目标权重面板（滞后生效，防未来）
        ▼
    组合构建 (kairos_portfolio)
        │  因子多头（等权 / 分数倾斜） vs 因子选股 + Ledoit-Wolf 协方差 + 风险平价
        ▼
    向量化回测 (kairos_backtest.VectorBacktester)
        │  收益 / 净值 / 换手 / 绩效指标
        ▼
    风险分析 (kairos_risk)
           VaR / ES / 回撤报告 / PSR / DSR / 因子风险分解 / 成分风险

产物：返回 dict（各阶段关键对象）；``out_dir`` 非空时额外写出
``REPORT.md``（中文，含真实数字）、``equity.csv``、``metrics.json``、
``weights.csv``、``factor_ic.csv``。

防未来函数约定
--------------
- 因子在 t 日只用 ``prices[:t]``（``pct_change(lookback)``）；
- 远期收益在 t 行的取值来自 ``(t, t+h]``，与因子行 t 直接配对即天然防未来；
- 目标权重面板相对因子再滞后 ``weight_lag`` 期（默认 1）；
- ``VectorBacktester`` 内部还会做一期成交滞后（``held = w.shift(1)``）。
  因此实际持仓相对因子信号共有 ``weight_lag + 1`` 期延迟，属**保守**设定。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import report as rp
from .._bootstrap import require

kf = require("kairos_factor")
kp = require("kairos_portfolio")
kb = require("kairos_backtest")
kr = require("kairos_risk")

#: 年化基准（日线）
ANNUAL = 252
#: 支持的优化器
OPTIMIZERS = ("risk_parity", "mean_variance", "min_variance", "max_sharpe", "equal_weight")
#: kairos_backtest.summary 的指标名 -> 中文标签（仅用于报告展示）
METRIC_LABELS = {
    "total_return": "累计收益", "cagr": "年化收益", "volatility": "年化波动",
    "sharpe": "夏普比率", "sortino": "索提诺比率", "max_drawdown": "最大回撤",
    "calmar": "卡玛比率", "win_rate": "日胜率", "profit_factor": "盈亏比",
    "periods": "样本期数",
}


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def _safe(func, *args, default: Any = float("nan"), **kwargs) -> Any:
    """执行 ``func``，失败（样本退化/缺可选依赖）时返回 ``default`` 而不中断流水线。"""
    try:
        return func(*args, **kwargs)
    except Exception:  # noqa: BLE001 - 演示流水线需对退化样本保持健壮
        return default


def _fin(x: Any) -> float:
    """转 float，非有限值转 NaN。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _last_active(weights: pd.DataFrame) -> pd.Series:
    """取权重面板中最后一个「非全零」的截面；全零则返回最后一行。"""
    total = weights.abs().sum(axis=1)
    idx = total[total > 1e-12].index
    row = idx[-1] if len(idx) else weights.index[-1]
    return weights.loc[row]


def _annualize_per_period(x: float, periods_per_year: float = ANNUAL,
                          horizon: int = 1) -> float:
    """把「每 horizon 期」的平均收益年化。"""
    v = _fin(x)
    if np.isnan(v) or v <= -1.0:
        return float("nan")
    return float((1.0 + v) ** (periods_per_year / max(1, int(horizon))) - 1.0)


# ---------------------------------------------------------------------------
# 阶段 1：因子
# ---------------------------------------------------------------------------
def momentum_factor(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """动量因子：过去 ``lookback`` 期累计收益（``prices.pct_change(lookback)``）。"""
    if lookback < 1:
        raise ValueError("lookback 必须 >= 1")
    return prices.astype("float64").pct_change(int(lookback))


def build_preprocessor(method: str = "mad", n_mad: float = 3.0) -> Any:
    """构造标准预处理流水线：去极值 → 截面标准化（``kairos_factor.FactorPipeline``）。"""
    pipe = kf.FactorPipeline()
    pipe.add(kf.winsorize, name="winsorize", method=method, n_mad=n_mad)
    pipe.add(kf.zscore, name="zscore")
    return pipe


def factor_stage(prices: pd.DataFrame, lookback: int = 60, horizon: int = 5,
                 n_quantiles: int = 5, horizons: Iterable[int] = (1, 2, 3, 5, 10),
                 winsor_method: str = "mad") -> Dict[str, Any]:
    """因子阶段：构造动量因子、预处理、并用 ``kairos_factor`` 做完整评价。"""
    raw = momentum_factor(prices, lookback)
    pipe = build_preprocessor(winsor_method)
    scores = pipe.fit_transform(raw)

    fwd = kf.forward_returns(prices, horizon=horizon, kind="price")
    ic_series = kf.ic(scores, fwd)
    rank_ic_series = kf.rank_ic(scores, fwd)
    qr = kf.quantile_returns(scores, fwd, n_quantiles=n_quantiles)
    report = kf.FactorReport.build(scores, fwd, n_quantiles=n_quantiles,
                                   prices_or_returns=prices, horizons=tuple(horizons),
                                   kind="price", name=f"mom{lookback}")
    return {
        "name": f"mom{lookback}",
        "lookback": int(lookback),
        "horizon": int(horizon),
        "raw": raw,
        "scores": scores,
        "pipeline": pipe,
        "forward_returns": fwd,
        "ic": ic_series,
        "rank_ic": rank_ic_series,
        "ic_summary": kf.ic_summary(ic_series),
        "rank_ic_summary": kf.ic_summary(rank_ic_series),
        "quantile_returns": qr,
        "report": report,
        "decay": report.decay,
        "coverage_mean": _fin(report.summary.get("coverage_mean")),
        "autocorr_mean": _fin(report.summary.get("autocorr_mean")),
        "long_short_mean": _fin(report.summary.get("long_short_mean")),
    }


# ---------------------------------------------------------------------------
# 阶段 1.5：样本外验证（可选，kairos_ml）
# ---------------------------------------------------------------------------
def walk_forward_validation(factor: Dict[str, Any], train_size: int = 250,
                            test_size: int = 63, expanding: bool = False,
                            cross_check_dates: int = 20) -> Dict[str, Any]:
    """用 ``kairos_ml`` 做 walk-forward 样本外验证与跨包一致性校验（可选阶段）。

    两件事：

    1. **样本外 IC 稳定性**：用 ``km.walk_forward_splits`` 把日期切成严格时序的
       训练/测试折（训练窗口全部早于测试窗口）。每折在训练期估计秩 IC 的均值与符号，
       再在测试期检验：
       - 测试期秩 IC 均值（是否仍与训练期同号）；
       - 多空组合收益的方向命中率 ``km.hit_rate``（把「训练期 IC 的符号」当作预测，
         把「测试期多空收益的符号」当作真值）；
       - 按训练期符号定向后的多空盈亏 ``km.signal_pnl`` 的 t 值与 p 值
         （``km.t_statistic``）。
    2. **跨包一致性校验**：对最后 ``cross_check_dates`` 个截面，分别用
       ``kairos_ml.ic(method="spearman")`` 与 ``kairos_factor.ic`` 计算同一天的秩 IC，
       报告两者的最大绝对差（口径对齐：``km.ic(method="spearman")`` 等价于
       ``kf.rank_ic``，都是「秩上的皮尔逊相关」）。两个独立实现应当吻合到浮点精度
       级别，这既是单元测试，也是「lab 把多个包串起来」的价值所在。

    ``kairos_ml`` 不可用时返回 ``{"available": False, "error": ...}``，流水线照常完成。
    """
    out: Dict[str, Any] = {"available": False, "train_size": int(train_size),
                           "test_size": int(test_size), "expanding": bool(expanding)}
    try:
        km = require("kairos_ml")
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    ic_series: pd.Series = factor["rank_ic"].dropna()
    ls: pd.Series = factor["report"].long_short
    n = len(ic_series)
    if n < int(train_size) + int(test_size):
        out["error"] = (f"有效截面 {n} 期不足以切成 train={train_size}/test={test_size} 的"
                        f" walk-forward 折")
        return out

    rows: List[Dict[str, Any]] = []
    try:
        splits = km.walk_forward_splits(n, train_size=int(train_size),
                                        test_size=int(test_size), expanding=expanding)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    for k, (tr, te) in enumerate(splits, start=1):
        ic_tr = ic_series.iloc[tr]
        ic_te = ic_series.iloc[te]
        sign = float(np.sign(ic_tr.mean())) or 1.0
        ls_te = ls.reindex(ic_te.index).dropna() * sign
        pnl = _safe(km.signal_pnl, pd.Series(sign, index=ls_te.index), ls_te,
                    default=pd.Series(dtype="float64"))
        t_stat, p_value = _safe(km.t_statistic, pnl, default=(float("nan"), float("nan")))
        hit = _safe(km.hit_rate, ls_te, pd.Series(sign, index=ls_te.index),
                    default=float("nan"))
        rows.append({
            "fold": k,
            "train_start": ic_tr.index[0], "train_end": ic_tr.index[-1],
            "test_start": ic_te.index[0], "test_end": ic_te.index[-1],
            "n_train": int(len(ic_tr)), "n_test": int(len(ic_te)),
            "ic_in_sample": _fin(ic_tr.mean()),
            "ic_out_of_sample": _fin(ic_te.mean()),
            "sign_persisted": bool(np.sign(ic_te.mean()) == np.sign(ic_tr.mean())),
            "ls_hit_rate": _fin(hit),
            "ls_t_stat": _fin(t_stat),
            "ls_p_value": _fin(p_value),
        })
    folds = pd.DataFrame(rows).set_index("fold")
    out.update({
        "available": True,
        "folds": folds,
        "n_folds": int(len(folds)),
        "ic_in_sample_mean": _fin(folds["ic_in_sample"].mean()),
        "ic_out_of_sample_mean": _fin(folds["ic_out_of_sample"].mean()),
        "sign_persistence": _fin(folds["sign_persisted"].astype(float).mean()),
        "ls_hit_rate_mean": _fin(folds["ls_hit_rate"].mean()),
        "ls_t_stat_mean": _fin(folds["ls_t_stat"].mean()),
    })

    # 跨包一致性：同一天、同一因子与远期收益，两个独立实现的秩 IC 应当一致
    fwd: pd.DataFrame = factor["forward_returns"]
    scores: pd.DataFrame = factor["scores"]
    k = max(1, int(cross_check_dates))
    dates = ic_series.index[-k:]
    diffs: List[float] = []
    for d in dates:
        f_row = scores.loc[d].dropna()
        r_row = fwd.loc[d].reindex(f_row.index).dropna()
        common = f_row.index.intersection(r_row.index)
        if len(common) < 3:
            continue
        a = _fin(km.ic(f_row[common], r_row[common], method="spearman"))
        # 口径对齐：km.ic(method="spearman") == kf.rank_ic（秩上的皮尔逊相关）；
        # kf.ic 是原始值上的皮尔逊相关，对应 km.ic(method="pearson")。
        # 另外 kairos_factor 按「行=日期、列=资产」的截面口径计算，必须保留日期维度。
        b = _fin(kf.rank_ic(scores.loc[[d], common], fwd.loc[[d], common]).iloc[0])
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(abs(a - b))
    out["cross_check_dates"] = int(len(diffs))
    out["cross_check_max_abs_diff"] = float(max(diffs)) if diffs else float("nan")
    return out


# ---------------------------------------------------------------------------
# 阶段 2：组合
# ---------------------------------------------------------------------------
def factor_weights(scores: pd.DataFrame, top_quantile: float = 0.3,
                   weighting: str = "equal", rebalance: Optional[int] = None,
                   lag: int = 1) -> pd.DataFrame:
    """把因子分数转成**做多**目标权重面板（date × asset）。

    参数
    ----
    top_quantile: 做多因子分数最高的比例（0.3 = 前 30%）。
    weighting:    ``"equal"`` 组内等权；``"score"`` 按截面分位排名倾斜（分数越高权重越大）。
    rebalance:    调仓间隔（期）；``None``/1 表示逐期调仓，>1 时只在每第 k 期重新选股，
                  其间沿用上一期权重（降低换手）。
    lag:          权重相对因子信号的滞后期数（防未来，默认 1）。
    """
    if not (0.0 < float(top_quantile) <= 1.0):
        raise ValueError("top_quantile 必须落在 (0, 1]")
    if weighting not in ("equal", "score"):
        raise ValueError(f"未知加权方式: {weighting!r}，仅支持 'equal' / 'score'")
    s = scores.astype("float64")
    pct = s.rank(axis=1, pct=True)                      # (0,1]，NaN 保持 NaN
    member = s.notna() & (pct > 1.0 - float(top_quantile))
    raw = pct.where(member) if weighting == "score" else member.astype("float64").where(member)
    total = raw.sum(axis=1)
    w = raw.div(total.where(total > 0), axis=0).fillna(0.0)

    k = int(rebalance) if rebalance else 1
    if k > 1:
        keep = np.zeros(len(w), dtype=bool)
        keep[::k] = True
        w = w.where(pd.Series(keep, index=w.index)).ffill().fillna(0.0)
    if lag:
        w = w.shift(int(lag))
    return w.fillna(0.0)


#: 风险平价定点迭代的收敛容差。
RP_TOL = 1e-10
#: damping 退避梯度：(damping, max_iter) 依次尝试，取第一个收敛解。
#: 波动差异极大时小 damping 更稳但更慢，故按「快而不稳 → 慢而稳」排列。
RP_LADDER: Tuple[Tuple[float, int], ...] = (
    (0.5, 1000), (0.3, 1500), (0.25, 3000), (0.2, 5000), (0.1, 8000),
)
#: 窗口内波动低于该阈值的资产视为「不可交易」（长期停牌、价格被前向填充成常数），
#: 直接剔除，避免协方差奇异，也避免风险平价把 ~100% 权重压给它。
MIN_TRADABLE_VOL = 1e-8


def tradable_assets(returns_window: pd.DataFrame,
                    min_vol: float = MIN_TRADABLE_VOL) -> List[Any]:
    """剔除窗口内方差退化（停牌、常数价格）的资产，返回可优化的列名列表。"""
    vol = returns_window.std(ddof=1)
    return [c for c in returns_window.columns
            if np.isfinite(_fin(vol.get(c, np.nan))) and float(vol[c]) > float(min_vol)]


def inverse_volatility_weights(returns: pd.DataFrame, floor: float = 1e-8) -> pd.Series:
    """逆波动率权重 ``w_i ∝ 1/σ_i``。

    它恰是「协方差为对角阵」时风险平价的解析解，因此既可作为对照基准，
    也可作为风险平价无解时的**解析回退**（无需迭代、必然收敛）。
    """
    ret = returns.dropna(how="any")
    if ret.shape[0] < 2 or ret.shape[1] < 1:
        raise ValueError("逆波动率权重至少需要 2 期、1 个资产的无缺失样本")
    vol = ret.std(ddof=1).clip(lower=floor)
    inv = 1.0 / vol
    total = float(inv.sum())
    if not np.isfinite(total) or total <= 0:
        return pd.Series(kp.equal_weight(ret))
    return (inv / total).rename("weight")


def risk_parity_feasible(cov: pd.DataFrame) -> bool:
    """快速判断风险平价是否存在 long-only 内部解（必要条件）。

    从逆波动率起点 ``w0 ∝ 1/σ`` 出发，检查每个资产的边际风险贡献
    ``(Σw0)_i`` 是否为正。若某资产与其余资产显著**负相关**（跨资产组合里的
    黄金 / 国债 vs 权益就很典型），它的边际贡献可能 ≤ 0，此时「各资产风险贡献
    相等且为正」的 long-only 解不存在：定点迭代会把该资产权重压到 0，随后
    ``RC=0`` 使权重再也回不来，残差永久卡在 ``1/n``。

    返回 False 时直接走回退链，省掉注定失败的迭代（实测可省 ~90% 的求解耗时）。
    """
    sigma = np.asarray(cov, dtype="float64")
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1] or sigma.shape[0] == 0:
        return False
    d = np.diag(sigma)
    if not np.all(np.isfinite(sigma)) or np.any(d <= 0):
        return False
    w0 = 1.0 / np.sqrt(d)
    w0 = w0 / w0.sum()
    return bool(np.all(sigma @ w0 > 0.0))


def risk_parity_robust(cov: pd.DataFrame, tol: float = RP_TOL,
                       ladder: Sequence[Tuple[float, int]] = RP_LADDER
                       ) -> Tuple[pd.Series, float]:
    """带 damping 退避梯度的风险平价求解，返回 ``(weights, damping)``。

    ``kairos_portfolio.risk_parity`` 用乘性定点迭代 ``w ← w ⊙ (b/RC)^damping``；
    波动差异极大时不同 damping 的收敛性差别很大，故按梯度依次尝试，取第一个收敛解。
    全部失败（或 :func:`risk_parity_feasible` 判定无内部解）时抛 ``RuntimeError``，
    由 :func:`risk_parity_chain` 接管回退。
    """
    if not risk_parity_feasible(cov):
        raise RuntimeError("风险平价无 long-only 内部解（存在边际风险贡献 ≤ 0 的资产）")
    last: Optional[BaseException] = None
    for damping, max_iter in ladder:
        try:
            w = kp.risk_parity(cov, tol=float(tol), max_iter=int(max_iter),
                               damping=float(damping))
            return pd.Series(w), float(damping)
        except RuntimeError as exc:
            last = exc
    raise RuntimeError(f"风险平价在 damping 梯度 {[d for d, _ in ladder]} 上均未收敛") from last


def risk_parity_chain(returns_window: pd.DataFrame, cov: pd.DataFrame,
                      delta: float = float("nan"), shrink_target: str = "const_corr",
                      tol: float = RP_TOL,
                      ladder: Sequence[Tuple[float, int]] = RP_LADDER
                      ) -> Tuple[pd.Series, str, pd.DataFrame, float]:
    """风险平价的三级求解链，保证任何输入都能给出可用的 long-only 权重。

    1. 在给定的 Ledoit-Wolf 收缩协方差上做定点迭代（damping 退避梯度）；
    2. 失败则把相关系数**全收缩**（δ=1）到常相关目标 r̄ 后重试 —— 保留各资产方差、
       只把不稳定的相关结构正则化，这是 Ledoit-Wolf 自带的端点情形；
    3. 仍失败则退化为逆波动率权重（对角协方差下风险平价的解析解）。

    返回 ``(weights, solver_label, cov_used, delta_used)``；``solver_label`` 会写明
    实际使用的 damping 或回退层级，直接进诊断表与报告，绝不静默降级。
    """
    try:
        w, damping = risk_parity_robust(cov, tol=tol, ladder=ladder)
        return w, f"risk_parity(damping={damping:g})", cov, delta
    except RuntimeError:
        pass
    try:
        lw = kp.ledoit_wolf(returns_window, target=shrink_target, shrinkage=1.0)
        w, damping = risk_parity_robust(lw.cov, tol=tol, ladder=ladder)
        return w, f"risk_parity(δ=1,damping={damping:g})", lw.cov, float(lw.delta)
    except (RuntimeError, ValueError):
        pass
    w = inverse_volatility_weights(returns_window)
    return w, "inverse_vol(风险平价无解)", cov, delta


def solve_weights(optimizer: str, hist_returns: pd.DataFrame, cov: pd.DataFrame,
                  risk_aversion: float = 5.0, delta: float = float("nan"),
                  shrink_target: str = "const_corr", rp_tol: float = RP_TOL,
                  rp_ladder: Sequence[Tuple[float, int]] = RP_LADDER
                  ) -> Tuple[pd.Series, str, pd.DataFrame, float]:
    """用 ``kairos_portfolio`` 在给定协方差上求一组 long-only 权重。

    返回 ``(weights, 求解器标签, 实际使用的协方差, 收缩强度 δ)``。
    ``mean_variance`` / ``min_variance`` / ``max_sharpe`` 的 long-only 路径依赖
    scipy-SLSQP，缺失时由调用方回退等权；``risk_parity`` 走
    :func:`risk_parity_chain` 的三级求解链（纯 numpy，必然给出结果）。
    """
    if optimizer == "risk_parity":
        return risk_parity_chain(hist_returns, cov, delta=delta,
                                 shrink_target=shrink_target, tol=rp_tol, ladder=rp_ladder)
    if optimizer == "min_variance":
        return (pd.Series(kp.min_variance(cov, long_only=True)),
                "min_variance(long_only)", cov, delta)
    if optimizer == "mean_variance":
        mu = kp.mean_returns(hist_returns)
        w = kp.mean_variance(mu, cov, risk_aversion=risk_aversion, long_only=True)
        return pd.Series(w), "mean_variance(long_only)", cov, delta
    if optimizer == "max_sharpe":
        mu = kp.mean_returns(hist_returns)
        return pd.Series(kp.max_sharpe(mu, cov, long_only=True)), "max_sharpe(long_only)", cov, delta
    if optimizer == "equal_weight":
        return pd.Series(kp.equal_weight(hist_returns)), "equal_weight", cov, delta
    raise ValueError(f"未知优化器: {optimizer!r}，可选 {OPTIMIZERS}")


def optimized_weights(prices: pd.DataFrame, scores: Optional[pd.DataFrame] = None,
                      top_quantile: float = 0.3, est_window: int = 120,
                      rebalance: int = 21, optimizer: str = "risk_parity",
                      shrink_target: str = "const_corr", risk_aversion: float = 5.0,
                      lag: int = 1, min_obs: Optional[int] = None
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """滚动构建「因子选股 + 协方差优化」的目标权重面板。

    流程（每个再平衡日 d，只用 ``prices[:d]``）：

    1. 取最近 ``est_window`` 期收益，按因子分数选出前 ``top_quantile`` 的标的；
    2. ``kairos_portfolio.ledoit_wolf`` 估计收缩协方差（记录收缩强度 δ）；
    3. ``solve_weights`` 求 long-only 权重（默认风险平价，纯 numpy 定点迭代）；
    4. 写入 d 行，非再平衡日沿用上一期权重；最后整体滞后 ``lag`` 期。

    ``scores=None`` 时对全宇宙做优化（不做因子选股）。求解失败（样本退化、
    缺 scipy）自动回退等权，并在诊断表的 ``solver`` 列标注。

    返回 ``(weights, diagnostics)``；diagnostics 逐次记录再平衡日、入选数、δ、
    求解器与组合波动。
    """
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"未知优化器: {optimizer!r}，可选 {OPTIMIZERS}")
    returns = prices.astype("float64").pct_change()
    dates = prices.index
    n = len(dates)
    est_window = max(5, int(est_window))
    if min_obs is None:
        min_obs = max(10, min(est_window // 2, max(10, n // 3)))
    min_obs = int(min_obs)

    # NaN 初始化 + 每个再平衡日写入**整行**（未入选资产显式置 0）：
    # 这样 ffill 传播的是「完整的一行持仓」，上一期的旧持仓不会残留；
    # 首个再平衡日之前保持 NaN → 最终填 0（空仓）。
    w = pd.DataFrame(np.nan, index=dates, columns=prices.columns, dtype="float64")
    diag: List[Dict[str, Any]] = []
    start = min(est_window, max(n - 2, 0))
    step = max(1, int(rebalance))
    for i in range(start, n, step):
        d = dates[i]
        names: List[Any]
        if scores is None:
            names = list(prices.columns)
        else:
            sel = scores.iloc[i].dropna()
            if len(sel) < 2:
                continue
            k = max(2, int(round(len(sel) * float(top_quantile))))
            names = list(sel.nlargest(min(k, len(sel))).index)
        sub = returns.iloc[max(0, i - est_window + 1): i + 1][names].dropna(how="any")
        if len(sub) < min_obs or len(names) < 2:
            continue
        # 剔除窗口内价格被填成常数的标的（停牌），否则协方差奇异 / 风险平价失稳
        keep = tradable_assets(sub)
        if len(keep) < 2:
            continue
        sub, names = sub[keep], keep
        delta = float("nan")
        solver = optimizer
        cov = None
        try:
            lw = kp.ledoit_wolf(sub, target=shrink_target)
            cov, delta = lw.cov, float(lw.delta)
            w_i, solver, cov, delta = solve_weights(optimizer, sub, cov, risk_aversion,
                                                    delta=delta,
                                                    shrink_target=shrink_target)
        except Exception:  # noqa: BLE001 - scipy 缺失或样本奇异时回退等权
            w_i, solver, cov = pd.Series(kp.equal_weight(sub)), "equal_weight(fallback)", None
        vals = pd.Series(w_i, dtype="float64").reindex(names).fillna(0.0)
        row = pd.Series(0.0, index=w.columns, dtype="float64")
        row.loc[vals.index] = vals.to_numpy()
        w.loc[d] = row.to_numpy()
        # portfolio_volatility(periods_per_year=252) 已经年化，此处不再乘 sqrt(252)
        vol = _fin(_safe(kp.portfolio_volatility, vals, cov, periods_per_year=ANNUAL)) \
            if cov is not None else float("nan")
        diag.append({"date": d, "n_assets": int(len(names)), "n_obs": int(len(sub)),
                     "shrinkage_delta": delta, "solver": solver,
                     "ann_volatility": vol,
                     "max_weight": float(vals.max()) if len(vals) else float("nan")})

    w = w.ffill().fillna(0.0)
    if lag:
        w = w.shift(int(lag))
    return w.fillna(0.0), pd.DataFrame(diag)


def static_reference_weights(returns: pd.DataFrame, optimizer: str = "mean_variance",
                             shrink_target: str = "const_corr",
                             risk_aversion: float = 5.0) -> Dict[str, Any]:
    """全样本静态优化参考（期末一次性求解，仅作对照，不参与回测）。"""
    out: Dict[str, Any] = {"optimizer": optimizer}
    try:
        lw = kp.ledoit_wolf(returns, target=shrink_target)
        out["shrinkage_delta"] = float(lw.delta)
        w, used, cov_used, delta_used = solve_weights(optimizer, returns, lw.cov,
                                                     risk_aversion, delta=out["shrinkage_delta"],
                                                     shrink_target=shrink_target)
        out["solver"] = used
        out["weights"] = pd.Series(w)
        out["volatility"] = _fin(kp.portfolio_volatility(w, cov_used, periods_per_year=ANNUAL))
        rc = kp.risk_contributions(w, cov_used)
        out["risk_pct"] = pd.Series(rc.pct)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def portfolio_stage(prices: pd.DataFrame, factor: Dict[str, Any],
                    top_quantile: float = 0.3, weighting: str = "equal",
                    rebalance: Optional[int] = None, est_window: int = 120,
                    opt_rebalance: int = 21, optimizer: str = "risk_parity",
                    risk_aversion: float = 5.0, lag: int = 1) -> Dict[str, Any]:
    """组合阶段：因子多头权重 + 优化权重 + 全宇宙等权基准。"""
    scores = factor["scores"]
    rb = int(rebalance) if rebalance else int(factor["horizon"])
    w_factor = factor_weights(scores, top_quantile=top_quantile, weighting=weighting,
                              rebalance=rb, lag=lag)
    w_opt, diag = optimized_weights(prices, scores, top_quantile=top_quantile,
                                    est_window=est_window, rebalance=opt_rebalance,
                                    optimizer=optimizer, risk_aversion=risk_aversion,
                                    lag=lag)
    n_assets = prices.shape[1]
    w_eq = pd.DataFrame(np.full((len(prices), n_assets), 1.0 / n_assets),
                        index=prices.index, columns=prices.columns)
    w_bench = w_eq

    returns = prices.astype("float64").pct_change().dropna()
    static = static_reference_weights(returns, optimizer="mean_variance",
                                      risk_aversion=risk_aversion)
    last = _last_active(w_opt if w_opt.abs().sum().sum() > 0 else w_factor)
    names = [c for c in last.index if float(last[c]) > 1e-9]
    sub = returns[names].tail(max(20, est_window // 2))
    cov_last = None
    risk_pct = None
    try:
        lw = kp.ledoit_wolf(sub, target="const_corr")
        cov_last = lw.cov
        wl = pd.Series(last)[names]
        risk_pct = pd.Series(kp.risk_contributions(wl, cov_last).pct)
    except Exception:  # noqa: BLE001
        pass

    return {
        "factor_weights": w_factor,
        "optimized_weights": w_opt,
        "benchmark_weights": w_bench,
        "diagnostics": diag,
        "rebalance": rb,
        "opt_rebalance": int(opt_rebalance),
        "optimizer": optimizer,
        "weighting": weighting,
        "top_quantile": float(top_quantile),
        "weight_lag": int(lag),
        "static_reference": static,
        "last_weights": last,
        "last_cov": cov_last,
        "last_risk_pct": risk_pct,
        "avg_holding": float((w_factor > 1e-9).sum(axis=1).replace(0, np.nan).mean()),
        "shrinkage_delta_mean": _fin(diag["shrinkage_delta"].mean()) if len(diag) else float("nan"),
    }


# ---------------------------------------------------------------------------
# 阶段 3：回测
# ---------------------------------------------------------------------------
def backtest_stage(prices: pd.DataFrame, weights: Dict[str, pd.DataFrame],
                   cost_rate: float = 0.001, risk_free: float = 0.0,
                   annual: int = ANNUAL) -> Dict[str, Any]:
    """回测阶段：对每个权重面板跑 ``VectorBacktester``，汇总绩效与换手。"""
    bt = kb.VectorBacktester(cost_rate=float(cost_rate))
    results: Dict[str, Any] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    turnover: Dict[str, float] = {}
    equity: Dict[str, pd.Series] = {}
    for name, w in weights.items():
        res = bt.run(prices, w)
        results[name] = res
        metrics[name] = res.metrics(risk_free=risk_free, periods_per_year=annual)
        turnover[name] = _fin(res.turnover.mean() * annual)      # 年化单边换手
        equity[name] = res.equity.rename(name)
    eq = pd.DataFrame(equity)
    frame = pd.DataFrame({k: v.returns.rename(k) for k, v in results.items()})
    return {
        "results": results,
        "metrics": metrics,
        "turnover_annualized": turnover,
        "equity": eq,
        "returns": frame,
        "cost_rate": float(cost_rate),
        "summary_frame": pd.DataFrame(
            {k: pd.Series(v) for k, v in metrics.items()}
        ).T if metrics else pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# 阶段 4：风险
# ---------------------------------------------------------------------------
def risk_stage(returns: pd.DataFrame, prices: pd.DataFrame,
               last_weights: pd.Series, cov: Optional[pd.DataFrame],
               confidence: float = 0.95, annual: int = ANNUAL,
               n_trials: int = 6, main: str = "factor") -> Dict[str, Any]:
    """风险阶段：VaR/ES、回撤报告、PSR/DSR、因子风险分解与成分风险。"""
    r = returns[main] if main in returns.columns else returns.iloc[:, 0]
    out: Dict[str, Any] = {"confidence": float(confidence), "main": main}
    out["var"] = _fin(_safe(kr.historical_var, r, confidence))
    out["es"] = _fin(_safe(kr.expected_shortfall, r, confidence, method="historical"))
    out["var_annualized"] = out["var"] * np.sqrt(annual) if np.isfinite(out["var"]) else float("nan")
    out["es_annualized"] = out["es"] * np.sqrt(annual) if np.isfinite(out["es"]) else float("nan")
    out["max_drawdown"] = _fin(_safe(kr.max_drawdown, r))
    dd = _safe(kr.drawdown_report, r, periods_per_year=annual, default=None)
    out["drawdown_report"] = dd
    out["drawdown"] = dd.summary() if dd is not None else {}
    out["psr"] = _fin(_safe(kr.psr_from_returns, r))
    dsr = _safe(kr.dsr_from_returns, r, n_trials=int(n_trials), default=None)
    out["dsr"] = dsr
    out["dsr_value"] = _fin(getattr(dsr, "dsr", float("nan")))
    out["dsr_sr0"] = _fin(getattr(dsr, "sr0", float("nan")))
    out["n_trials"] = int(n_trials)
    out["risk_report"] = _safe(kr.risk_report, returns, confidence=confidence,
                               periods_per_year=float(annual), default=pd.DataFrame())
    out["sharpe"] = _fin(_safe(kr.sharpe_ratio, r, periods_per_year=float(annual)))
    out["volatility"] = _fin(_safe(kr.volatility, r, periods_per_year=float(annual)))
    out["skewness"] = _fin(_safe(kr.skewness, r))
    out["kurtosis"] = _fin(_safe(kr.kurtosis, r))

    # 因子（市场模型）风险分解：以等权全宇宙收益为市场因子
    rets = prices.astype("float64").pct_change().dropna()
    model = None
    breakdown = None
    try:
        market = rets.mean(axis=1)
        model = kr.market_model(rets, market, market_name="MKT")
        wl = pd.Series(last_weights).reindex(prices.columns).fillna(0.0)
        if float(wl.abs().sum()) > 1e-9:
            breakdown = kr.factor_risk_decomposition(wl, model)
    except Exception:  # noqa: BLE001
        model, breakdown = None, None
    out["factor_model"] = model
    out["factor_breakdown"] = breakdown

    comp = None
    if cov is not None:
        try:
            wl = pd.Series(last_weights).reindex(cov.columns).fillna(0.0)
            if float(wl.abs().sum()) > 1e-9:
                comp = kr.component_risk(wl, cov)
        except Exception:  # noqa: BLE001
            comp = None
    out["component_risk"] = comp
    return out


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def _quantile_table(factor: Dict[str, Any], annual: int) -> pd.DataFrame:
    """分层收益表：每 horizon 期均值、年化、t 值与累计。"""
    qr: pd.DataFrame = factor["quantile_returns"]
    h = int(factor["horizon"])
    rows = []
    for col in qr.columns:
        s = qr[col].dropna()
        mean = float(s.mean()) if len(s) else float("nan")
        std = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
        t = mean / std * np.sqrt(len(s)) if std and np.isfinite(std) and std > 1e-14 else float("nan")
        rows.append({
            "分层": col,
            f"每期({h}日)均值": mean,
            "年化收益": _annualize_per_period(mean, annual, h),
            "年化波动": std * np.sqrt(annual / h) if np.isfinite(std) else float("nan"),
            "t值": t,
            "胜率": float((s > 0).mean()) if len(s) else float("nan"),
            "样本期数": int(len(s)),
        })
    return pd.DataFrame(rows).set_index("分层")


def _data_section(prices: pd.DataFrame, volumes: Optional[pd.DataFrame],
                  meta: Dict[str, Any]) -> str:
    """数据小节：面板规模、区间、样本与横截面波动概览。"""
    rets = prices.pct_change().dropna()
    ann_vol = rets.std(ddof=1) * np.sqrt(ANNUAL)
    cum = (1.0 + rets).prod() - 1.0
    ann_ret = (1.0 + cum) ** (ANNUAL / max(len(rets), 1)) - 1.0
    table = pd.DataFrame({"区间累计收益": cum, "区间年化收益": ann_ret,
                          "区间年化波动": ann_vol,
                          "年化夏普": ann_ret / ann_vol.where(ann_vol > 0)})
    table = table.sort_values("区间累计收益", ascending=False)
    show = table if len(table) <= 12 else pd.concat([table.head(6), table.tail(6)])
    info = {
        "资产数": int(prices.shape[1]),
        "交易日数": int(prices.shape[0]),
        "价格缺失值个数": int(prices.isna().sum().sum()),
        "年化波动(横截面中位数)": _fin(ann_vol.median()),
        "累计收益(横截面中位数)": _fin(cum.median()),
        "最好资产累计收益": _fin(cum.max()),
        "最差资产累计收益": _fin(cum.min()),
    }
    lines = [rp.dict_to_md(info, "指标", "数值"), "",
             "**逐资产概览**（按区间累计收益排序；超过 12 个资产时只显示最好/最差各 6 个）", "",
             rp.frame_to_md(show, floatfmt=".4f", index_label="资产")]
    if volumes is not None:
        lines += ["", f"成交量面板：{volumes.shape[0]} 日 × {volumes.shape[1]} 资产，"
                      f"日均总成交量 {float(volumes.sum(axis=1).mean()):,.0f}"
                      f"（单位以数据源口径为准，可用于执行阶段的参与率约束）。"]
    return "\n".join(lines)


def _factor_section(factor: Dict[str, Any], annual: int) -> str:
    """因子小节：构造方式、IC/IR、分层收益与 IC 衰减。"""
    ric = factor["rank_ic_summary"]
    ic = factor["ic_summary"]
    scalars = {
        "因子": factor["name"],
        "回看期(lookback)": int(factor["lookback"]),
        "远期收益期限(horizon)": int(factor["horizon"]),
        "预处理": " → ".join(factor["pipeline"].names),
        "IC 均值": ic["mean"],
        "IC 标准差": ic["std"],
        "ICIR": ic["ir"],
        "秩IC 均值": ric["mean"],
        "秩IC 标准差": ric["std"],
        "秩ICIR(=IC_IR)": ric["ir"],
        "秩IC t值": ric["t_stat"],
        "秩IC>0 占比": ric["positive_ratio"],
        "有效截面数": int(ric["count"]),
        "多空(Q末-Q1)每期均值": factor["long_short_mean"],
        "因子覆盖率均值": factor["coverage_mean"],
        "因子自相关(滞后1期)": factor["autocorr_mean"],
    }
    parts = [rp.dict_to_md(scalars, "指标", "数值"), "",
             "**分层（分位）组合收益**：Q1 = 因子值最低组，Q5 = 因子值最高组。", "",
             rp.frame_to_md(_quantile_table(factor, annual), floatfmt=".4f", index_label="分层")]
    decay = factor.get("decay")
    if decay is not None and len(decay):
        view = decay.copy()
        if "count" in view.columns:
            view["count"] = view["count"].astype(int)
        parts += ["", "**IC 衰减**（同一因子在不同远期期限上的预测力）", "",
                  rp.frame_to_md(view, floatfmt=".4f", index_label="期限")]
    return "\n".join(parts)


def _validation_section(val: Dict[str, Any]) -> str:
    """样本外验证小节（kairos_ml 可选阶段）。"""
    if not val.get("available"):
        return ("_未执行样本外验证_"
                + (f"：{val['error']}" if val.get("error") else "")
                + "。该阶段依赖可选的 `kairos_ml`，缺失或样本过短时自动跳过，"
                  "不影响其余阶段。")
    scalars = {
        "walk-forward 折数": int(val["n_folds"]),
        "训练窗口(期)": int(val["train_size"]),
        "测试窗口(期)": int(val["test_size"]),
        "窗口方式": "扩展(expanding)" if val["expanding"] else "滚动(rolling)",
        "样本内秩IC均值": val["ic_in_sample_mean"],
        "样本外秩IC均值": val["ic_out_of_sample_mean"],
        "符号保持比例": val["sign_persistence"],
        "多空方向命中率均值": val["ls_hit_rate_mean"],
        "定向后多空 t 值均值": val["ls_t_stat_mean"],
        "跨包校验截面数": int(val["cross_check_dates"]),
        "kairos_ml.ic(spearman) 与 kairos_factor.rank_ic 最大绝对差":
            val["cross_check_max_abs_diff"],
    }
    parts = [rp.dict_to_md(scalars, "指标", "数值"),
             "> 口径说明：「样本外秩IC」衡量因子在整个截面上的单调性，"
             "「多空方向命中率 / t 值」只看两端分位组（Q末 − Q1）的收益方向，"
             "两者在个别窗口可能异号，属正常现象。"]
    folds: pd.DataFrame = val.get("folds")
    if folds is not None and len(folds):
        view = folds.copy()
        for col in ("train_start", "train_end", "test_start", "test_end"):
            if col in view.columns:
                view[col] = [str(pd.Timestamp(x).date()) for x in view[col]]
        parts += ["", "**逐折明细**（训练窗口严格早于测试窗口，无重叠、无未来信息）", "",
                  rp.frame_to_md(view, floatfmt=".4f", index_label="fold")]
    diff = val["cross_check_max_abs_diff"]
    if np.isfinite(diff):
        parts += ["", f"> 跨包一致性：`kairos_ml.ic(method='spearman')` 与 "
                      f"`kairos_factor.rank_ic` 在相同截面上的秩 IC 最大绝对差为 "
                      f"{diff:.2e}，两套独立实现吻合到浮点精度。"]
    return "\n".join(parts)


def _portfolio_section(port: Dict[str, Any], prices: pd.DataFrame) -> str:
    """组合小节：权重构造规则、优化诊断与期末权重/风险贡献。"""
    diag: pd.DataFrame = port["diagnostics"]
    scalars = {
        "做多比例(top_quantile)": port["top_quantile"],
        "因子组合加权方式": port["weighting"],
        "因子组合调仓间隔(期)": int(port["rebalance"]),
        "优化组合调仓间隔(期)": int(port["opt_rebalance"]),
        "优化器": port["optimizer"],
        "权重滞后(防未来)": int(port["weight_lag"]),
        "再平衡次数": int(len(diag)),
        "Ledoit-Wolf 收缩强度 δ 均值": port["shrinkage_delta_mean"],
        "因子组合平均持仓数": port["avg_holding"],
        "资产总数": int(prices.shape[1]),
    }
    parts = [rp.dict_to_md(scalars, "指标", "数值")]
    if len(diag):
        parts += ["", "**滚动优化诊断（最近 8 次再平衡）**", "",
                  rp.frame_to_md(diag.tail(8).set_index("date"), floatfmt=".4f",
                                 index_label="再平衡日")]
    last = port["last_weights"]
    pct = port["last_risk_pct"]
    if last is not None:
        active = last[last > 1e-9].sort_values(ascending=False)
        tbl = pd.DataFrame({"期末权重": active.head(12)})
        if pct is not None:
            tbl["风险贡献占比"] = pct.reindex(tbl.index)
        parts += ["", "**期末持仓（优化组合，权重最大的 12 只）**", "",
                  rp.frame_to_md(tbl, floatfmt=".4f", index_label="资产")]
    static = port["static_reference"]
    if "weights" in static:
        sw = pd.Series(static["weights"])
        sw = sw[sw > 1e-9].sort_values(ascending=False)
        parts += ["", f"**全样本静态 {static.get('solver', static['optimizer'])} 参考权重（前 10）**"
                      f"，δ={static.get('shrinkage_delta', float('nan')):.4f}，"
                      f"年化波动={_fin(static.get('volatility')) * 100:.2f}%", "",
                  rp.series_to_md(sw.head(10), name="权重", index_label="资产")]
    elif "error" in static:
        parts += ["", f"全样本静态优化不可用：`{static['error']}`（缺 scipy 时自动跳过）。"]
    return "\n".join(parts)


def _backtest_section(bt: Dict[str, Any], annual: int) -> str:
    """回测小节：绩效汇总表 + 换手与成本。"""
    frame: pd.DataFrame = bt["summary_frame"]          # index=组合, columns=指标
    order = ["total_return", "cagr", "volatility", "sharpe", "sortino",
             "max_drawdown", "calmar", "win_rate", "profit_factor", "periods"]
    view = frame[[c for c in order if c in frame.columns]].T   # index=指标, columns=组合
    view = view.rename(index=METRIC_LABELS)
    if "样本期数" in view.index:
        view.loc["样本期数"] = view.loc["样本期数"].astype(int)
    parts = [f"回测口径：单边成本率 {bt['cost_rate']:.4%}，年化基准 {annual} 期/年，"
             f"权重滞后一期成交（`held = w.shift(1)`）。", "",
             rp.frame_to_md(view.T, floatfmt=".4f", index_label="指标"), "",
             "**年化单边换手**", "",
             rp.dict_to_md({k: v for k, v in bt["turnover_annualized"].items()},
                           "组合", "年化换手")]
    eq: pd.DataFrame = bt["equity"]
    if len(eq):
        parts += ["", "**净值曲线（首/中/末三个时点）**", "",
                  rp.frame_to_md(pd.concat([eq.iloc[[0]], eq.iloc[[len(eq) // 2]],
                                            eq.iloc[[-1]]]), floatfmt=".4f",
                                 index_label="日期")]
    return "\n".join(parts)


def _risk_section(risk: Dict[str, Any], annual: int) -> str:
    """风险小节：VaR/ES、回撤、PSR/DSR、因子与成分风险分解。"""
    c = risk["confidence"]
    scalars = {
        f"历史 VaR({c:.0%}, 每日)": risk["var"],
        f"期望损失 ES({c:.0%}, 每日)": risk["es"],
        f"VaR({c:.0%}, 年化)": risk["var_annualized"],
        f"ES({c:.0%}, 年化)": risk["es_annualized"],
        "最大回撤": risk["max_drawdown"],
        "年化波动": risk["volatility"],
        "年化夏普": risk["sharpe"],
        "偏度": risk["skewness"],
        "超额峰度": risk["kurtosis"],
        "PSR(概率化夏普)": risk["psr"],
        f"DSR(缩水夏普, n_trials={risk['n_trials']})": risk["dsr_value"],
        "DSR 基准夏普 SR0": risk["dsr_sr0"],
    }
    dd = risk.get("drawdown") or {}
    if dd:
        scalars.update({
            "当前回撤": dd.get("current_drawdown", float("nan")),
            "最长回撤期数": int(dd.get("longest_drawdown_periods") or 0),
            "水下时间占比": dd.get("time_underwater_ratio", float("nan")),
            "Calmar": dd.get("calmar_ratio", float("nan")),
            "回撤区间数": int(dd.get("n_episodes") or 0),
        })
    parts = [f"主分析对象：`{risk['main']}` 组合的每期净收益。",
             "> 口径说明：`kairos_risk.sharpe_ratio` 用**几何**年化收益，"
             "`kairos_backtest.sharpe_ratio` 用**算术**年化收益，两者数值略有差异属正常。",
             "",
             rp.dict_to_md(scalars, "指标", "数值")]
    rr: pd.DataFrame = risk.get("risk_report")
    if rr is not None and len(rr):
        keep = [r for r in ("annualized_return", "volatility", "sharpe", "historical_var",
                            "parametric_var_normal", "modified_var", "es_historical",
                            "es_student_t", "downside_deviation", "sortino",
                            "skewness", "excess_kurtosis") if r in rr.index]
        parts += ["", "**各组合风险体检（kairos_risk.risk_report）**", "",
                  rp.frame_to_md(rr.loc[keep], floatfmt=".4f", index_label="指标")]
    brk = risk.get("factor_breakdown")
    if brk is not None:
        parts += ["", "**因子（市场模型）风险分解**：组合方差 = 因子部分 + 特质部分。", "",
                  rp.dict_to_md({
                      "组合市场暴露 β": _fin(brk.portfolio_exposure.iloc[0]),
                      "因子方差占比": _fin(brk.factor_share),
                      "特质方差占比": 1.0 - _fin(brk.factor_share),
                      "组合年化波动(模型)": _fin(brk.volatility) * np.sqrt(annual),
                      "因子数": int(getattr(brk, "factor_contrib", pd.Series()).size),
                  }, "指标", "数值")]
    comp = risk.get("component_risk")
    if comp is not None:
        parts += ["", "**成分风险分解（欧拉分配）**", "",
                  rp.dict_to_md({
                      "组合每期波动": _fin(comp.volatility),
                      "加权平均波动": _fin(comp.weighted_volatility),
                      "分散化比率": _fin(comp.diversification_ratio),
                      "风险有效资产数": _fin(comp.effective_n),
                      "风险 Herfindahl": _fin(comp.herfindahl),
                  }, "指标", "数值")]
    return "\n".join(parts)


def build_report(meta: Dict[str, Any], prices: pd.DataFrame,
                 volumes: Optional[pd.DataFrame], factor: Dict[str, Any],
                 port: Dict[str, Any], bt: Dict[str, Any], risk: Dict[str, Any],
                 validation: Optional[Dict[str, Any]] = None,
                 annual: int = ANNUAL) -> str:
    """把各阶段产物渲染成一份中文 Markdown 报告正文（不含一级标题）。"""
    head = [
        f"- 生成时间：{meta['generated_at']}",
        f"- 数据来源：{meta['source']}",
        f"- 样本区间：{meta['start']} ~ {meta['end']}（{meta['n_days']} 个交易日 × {meta['n_assets']} 个资产）",
        f"- 流水线参数：lookback={meta['lookback']}, horizon={meta['horizon']}, "
        f"top_quantile={meta['top_quantile']}, cost_rate={meta['cost_rate']:.4%}, "
        f"optimizer={meta['optimizer']}",
        "- 依赖版本：" + " / ".join(f"{k} {v}" for k, v in meta["versions"].items()),
    ]
    sections = [
        ("", "\n".join(head)),
        ("数据（kairos_data）", _data_section(prices, volumes, meta)),
        ("因子（kairos_factor）", _factor_section(factor, annual)),
        ("样本外验证（kairos_ml，可选）", _validation_section(validation or {})),
        ("组合（kairos_portfolio）", _portfolio_section(port, prices)),
        ("回测（kairos_backtest）", _backtest_section(bt, annual)),
        ("风险（kairos_risk）", _risk_section(risk, annual)),
        ("结论与口径说明", rp.bullets([
            f"动量因子 `{factor['name']}` 的秩 IC 为 "
            f"{factor['rank_ic_summary']['mean']:+.4f}"
            f"（t = {factor['rank_ic_summary']['t_stat']:+.2f}），"
            f"IC>0 的截面占比 {factor['rank_ic_summary']['positive_ratio']:.1%}；"
            "分层收益若呈「Q1 最高」形态，说明该股票池在 lookback 上表现为**反转**而非动量，"
            "这与 A 股短中期动量偏弱的普遍经验一致。",
            "股票池是**事后精选**的流动性好的大盘股，等权买入持有本身就带幸存者偏差，"
            "因此「因子组合跑输等权基准」在本样本里并不意外；基准只用于对照口径，"
            "不代表可投资的指数收益。",
            f"权重相对因子信号滞后 {meta['weight_lag']} 期，回测器再做一期成交滞后"
            f"（`held = w.shift(1)`），因此实际持仓相对信号共有 {meta['weight_lag'] + 1} 期延迟，"
            "属保守设定，宁可低估表现也不引入未来信息。",
            f"换手成本按单边 {meta['cost_rate']:.2%} × 换手率计提；回测器假设每日再平衡到目标权重，"
            "因此换手口径高于「只在调仓日交易」的真实情形，成本估计偏保守。",
            f"PSR = {risk['psr']:.3f}、DSR({risk['n_trials']} trials) = {risk['dsr_value']:.3f} "
            "用于衡量「这个夏普有多大概率不是运气」；DSR 会按试验次数扣减，"
            "试过的参数越多、扣得越狠。",
            f"本样本 {meta['n_assets']} 个资产、{meta['n_days']} 个交易日，"
            "所有数字仅用于演示流水线，**不构成投资建议**。",
        ])),
        ("如何复现", rp.code_block(
            "python examples/run_e2e_equity.py --data-dir <kairos-data>/data/ashare",
            "bash")),
    ]
    body: List[str] = []
    for heading, text in sections:
        if heading:
            body.append(f"## {heading}")
        body.append(text.strip())
    return "\n\n".join(body).strip() + "\n"


def _metrics_payload(meta: Dict[str, Any], factor: Dict[str, Any], port: Dict[str, Any],
                     bt: Dict[str, Any], risk: Dict[str, Any],
                     validation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """组装可 JSON 序列化的指标载荷（``metrics.json``）。"""
    return {
        "meta": meta,
        "factor": {
            "name": factor["name"],
            "lookback": factor["lookback"],
            "horizon": factor["horizon"],
            "ic_summary": factor["ic_summary"],
            "rank_ic_summary": factor["rank_ic_summary"],
            "coverage_mean": factor["coverage_mean"],
            "autocorr_mean": factor["autocorr_mean"],
            "long_short_mean": factor["long_short_mean"],
            "quantile_annualized": _quantile_table(factor, ANNUAL)["年化收益"].to_dict(),
            "decay": factor["decay"],
        },
        "portfolio": {
            "top_quantile": port["top_quantile"],
            "weighting": port["weighting"],
            "optimizer": port["optimizer"],
            "rebalance": port["rebalance"],
            "opt_rebalance": port["opt_rebalance"],
            "weight_lag": port["weight_lag"],
            "n_rebalance": int(len(port["diagnostics"])),
            "shrinkage_delta_mean": port["shrinkage_delta_mean"],
            "avg_holding": port["avg_holding"],
            "last_weights": port["last_weights"],
            "diagnostics": port["diagnostics"],
        },
        "backtest": {
            "cost_rate": bt["cost_rate"],
            "metrics": bt["metrics"],
            "turnover_annualized": bt["turnover_annualized"],
            "summary": bt["summary_frame"],
        },
        "validation": validation or {},
        "risk": {
            "confidence": risk["confidence"],
            "var": risk["var"],
            "es": risk["es"],
            "var_annualized": risk["var_annualized"],
            "es_annualized": risk["es_annualized"],
            "max_drawdown": risk["max_drawdown"],
            "drawdown": risk["drawdown"],
            "psr": risk["psr"],
            "dsr": risk["dsr_value"],
            "dsr_sr0": risk["dsr_sr0"],
            "n_trials": risk["n_trials"],
            "volatility": risk["volatility"],
            "sharpe": risk["sharpe"],
            "risk_report": risk["risk_report"],
        },
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_equity_pipeline(prices: pd.DataFrame, volumes: Optional[pd.DataFrame] = None,
                        lookback: int = 60, horizon: int = 5, cost_rate: float = 0.001,
                        out_dir: Optional[str] = None, top_quantile: float = 0.3,
                        weighting: str = "equal", rebalance: Optional[int] = None,
                        est_window: int = 120, opt_rebalance: int = 21,
                        optimizer: str = "risk_parity", risk_aversion: float = 5.0,
                        confidence: float = 0.95, annual: int = ANNUAL,
                        n_quantiles: int = 5, horizons: Sequence[int] = (1, 2, 3, 5, 10),
                        weight_lag: int = 1, risk_free: float = 0.0,
                        n_trials: int = 6, oos_train: int = 250, oos_test: int = 63,
                        with_validation: bool = True, source: str = "prices panel",
                        name: str = "e2e_equity") -> Dict[str, Any]:
    """跑完整的 A 股端到端研究流水线。

    参数
    ----
    prices:      价格面板 DataFrame(index=日期, columns=资产)，必须为正价格。
    volumes:     可选成交量面板（仅用于报告与执行演示的口径说明）。
    lookback:    动量因子回看期。
    horizon:     远期收益期限，同时作为因子组合的默认调仓间隔。
    cost_rate:   单边换手成本率（如 0.001 = 千一）。
    out_dir:     非空时把 ``REPORT.md`` / ``equity.csv`` / ``metrics.json`` /
                 ``weights.csv`` / ``factor_ic.csv`` 写入该目录。
    top_quantile: 做多因子分数最高的比例。
    weighting:   ``"equal"`` 组内等权 / ``"score"`` 按分数倾斜。
    rebalance:   因子组合调仓间隔；``None`` 表示等于 ``horizon``。
    est_window:  优化组合的协方差估计窗口。
    opt_rebalance: 优化组合调仓间隔。
    optimizer:   ``risk_parity`` / ``mean_variance`` / ``min_variance`` / ``max_sharpe`` /
                 ``equal_weight``。
    risk_aversion: ``mean_variance`` / ``max_sharpe`` 的风险厌恶系数 λ。
    confidence:  VaR/ES 置信度。
    annual:      年化期数（日线 252）。
    n_quantiles: 分层收益的组数。
    horizons:    IC 衰减分析的期限序列。
    weight_lag:  权重相对因子信号的滞后（防未来）。
    risk_free:   无风险利率（每期）。
    n_trials:    DSR 的试验次数（本次研究评估过的策略/参数组合数）。
    oos_train / oos_test: walk-forward 样本外验证的训练/测试窗口长度（期）。
    with_validation: 是否执行可选的样本外验证阶段（依赖 ``kairos_ml``）。
    source:      数据来源描述（写入报告）。
    name:        报告标题用的流水线名。

    返回
    ----
    dict，键为 ``meta`` / ``factor`` / ``portfolio`` / ``backtest`` / ``risk`` /
    ``artifacts``，值为各阶段的原始对象（DataFrame/Series/dataclass）与标量汇总。
    """
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] < 2:
        raise ValueError("prices 必须是至少含 2 个资产的 DataFrame(index=日期, columns=资产)")
    prices = prices.astype("float64").sort_index()
    if volumes is not None:
        volumes = volumes.reindex(index=prices.index, columns=prices.columns)

    meta: Dict[str, Any] = {
        "pipeline": name,
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "n_assets": int(prices.shape[1]),
        "n_days": int(prices.shape[0]),
        "start": str(pd.Timestamp(prices.index[0]).date()),
        "end": str(pd.Timestamp(prices.index[-1]).date()),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "top_quantile": float(top_quantile),
        "cost_rate": float(cost_rate),
        "optimizer": optimizer,
        "weight_lag": int(weight_lag),
        "annual": int(annual),
        "versions": {m.__name__: getattr(m, "__version__", "?")
                     for m in (kf, kp, kb, kr)},
    }

    factor = factor_stage(prices, lookback=lookback, horizon=horizon,
                          n_quantiles=n_quantiles, horizons=horizons)
    validation = (walk_forward_validation(factor, train_size=oos_train, test_size=oos_test)
                  if with_validation else {"available": False})
    port = portfolio_stage(prices, factor, top_quantile=top_quantile, weighting=weighting,
                           rebalance=rebalance, est_window=est_window,
                           opt_rebalance=opt_rebalance, optimizer=optimizer,
                           risk_aversion=risk_aversion, lag=weight_lag)
    bt = backtest_stage(prices, {
        "factor": port["factor_weights"],
        "optimized": port["optimized_weights"],
        "equal_weight": port["benchmark_weights"],
    }, cost_rate=cost_rate, risk_free=risk_free, annual=annual)
    risk = risk_stage(bt["returns"], prices, port["last_weights"], port["last_cov"],
                      confidence=confidence, annual=annual, n_trials=n_trials,
                      main="factor")

    artifacts: Dict[str, str] = {}
    if out_dir:
        d = rp.ensure_dir(out_dir)
        text = build_report(meta, prices, volumes, factor, port, bt, risk,
                            validation, annual)
        title = f"Kairos Lab · {name} 端到端研究报告"
        artifacts["report_md"] = rp.write_text(os.path.join(d, "REPORT.md"),
                                               f"# {title}\n\n{text}")
        eq = bt["equity"].copy()
        eq["factor_drawdown"] = kr.drawdown_series(bt["results"]["factor"].returns)
        artifacts["equity_csv"] = rp.write_csv(eq, os.path.join(d, "equity.csv"))
        artifacts["metrics_json"] = rp.write_json(
            os.path.join(d, "metrics.json"),
            _metrics_payload(meta, factor, port, bt, risk, validation))
        artifacts["weights_csv"] = rp.write_csv(
            pd.concat([port["factor_weights"].add_prefix("factor_"),
                       port["optimized_weights"].add_prefix("opt_")], axis=1),
            os.path.join(d, "weights.csv"))
        artifacts["factor_ic_csv"] = rp.write_csv(
            pd.DataFrame({"ic": factor["ic"], "rank_ic": factor["rank_ic"]}),
            os.path.join(d, "factor_ic.csv"))

    return {"meta": meta, "factor": factor, "validation": validation,
            "portfolio": port, "backtest": bt, "risk": risk, "artifacts": artifacts}


def summarize(result: Dict[str, Any]) -> str:
    """把流水线结果压成一段可直接 print 的中文摘要（示例脚本用）。"""
    meta, factor, port = result["meta"], result["factor"], result["portfolio"]
    bt, risk = result["backtest"], result["risk"]
    ric = factor["rank_ic_summary"]
    lines = [
        f"[{meta['pipeline']}] {meta['source']} | {meta['start']}~{meta['end']} "
        f"| {meta['n_days']}日 × {meta['n_assets']}资产",
        f"  因子 {factor['name']}: 秩IC={ric['mean']:+.4f} ICIR={ric['ir']:+.4f} "
        f"t={ric['t_stat']:+.2f} IC>0占比={ric['positive_ratio']:.1%} "
        f"覆盖率={factor['coverage_mean']:.1%}",
    ]
    for k, m in bt["metrics"].items():
        lines.append(f"  回测 {k:<12} 累计={m['total_return']:+.2%} 年化={m['cagr']:+.2%} "
                     f"波动={m['volatility']:.2%} 夏普={m['sharpe']:+.3f} "
                     f"回撤={m['max_drawdown']:.2%} 换手={bt['turnover_annualized'][k]:.1f}x/年")
    val = result.get("validation") or {}
    if val.get("available"):
        lines.append(f"  样本外验证: {val['n_folds']}折 样本内IC={val['ic_in_sample_mean']:+.4f} "
                     f"样本外IC={val['ic_out_of_sample_mean']:+.4f} "
                     f"符号保持={val['sign_persistence']:.0%} "
                     f"命中率={val['ls_hit_rate_mean']:.1%} "
                     f"跨包秩IC最大差={val['cross_check_max_abs_diff']:.1e}")
    lines.append(f"  风险 VaR({risk['confidence']:.0%})={risk['var']:.4f} "
                 f"ES={risk['es']:.4f} 最大回撤={risk['max_drawdown']:.2%} "
                 f"PSR={risk['psr']:.3f} DSR={risk['dsr_value']:.3f} "
                 f"(δ={port['shrinkage_delta_mean']:.3f})")
    if result["artifacts"]:
        lines.append(f"  产物: {', '.join(os.path.basename(v) for v in result['artifacts'].values())}"
                     f" -> {os.path.dirname(result['artifacts']['report_md'])}")
    return "\n".join(lines)
