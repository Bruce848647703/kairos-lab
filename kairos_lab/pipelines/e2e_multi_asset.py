"""端到端流水线 ②：跨资产 ETF「全天候」配置（数据 → 组合 → 回测 → 风险）。

用**真实 ETF 日线面板**（A 股宽基 / 海外股 / 黄金 / 国债 / 货币）演示
``kairos_portfolio`` 的风险平价配置如何落到 ``kairos_backtest`` 与 ``kairos_risk``：

::

    prices (kairos_data, 9 只跨资产 ETF)
        │  滚动 est_window 期收益
        ▼
    Ledoit-Wolf 收缩协方差 (kairos_portfolio.ledoit_wolf)
        │  定点迭代风险平价（纯 numpy，天然 long-only）
        ▼
    目标权重面板（再平衡日重算，其间沿用；整体滞后 lag 期防未来）
        │  对照：等权 / 逆波动率 / 静态均值-方差 / 有效前沿
        ▼
    向量化回测 (kairos_backtest.VectorBacktester)
        ▼
    风险 (kairos_risk)：VaR/ES、回撤报告、PSR/DSR、成分风险、资产类别风险归集

与流水线 ①（A 股因子链）共用 :mod:`kairos_lab.pipelines.e2e_equity` 里的
求解器、报告渲染与 JSON 落盘工具，避免重复实现。

产物：``REPORT.md`` / ``equity.csv`` / ``metrics.json`` / ``weights.csv``。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import report as rp
from .._bootstrap import require
from .e2e_equity import (
    ANNUAL,
    METRIC_LABELS,
    OPTIMIZERS,
    RP_LADDER,
    _fin,
    _last_active,
    _safe,
    inverse_volatility_weights,
    risk_parity_chain,
    solve_weights,
    tradable_assets,
)

kp = require("kairos_portfolio")
kb = require("kairos_backtest")
kr = require("kairos_risk")


# ---------------------------------------------------------------------------
# 组合构建
# ---------------------------------------------------------------------------
def rolling_inverse_vol(prices: pd.DataFrame, est_window: int = 120, rebalance: int = 21,
                        lag: int = 1, min_obs: Optional[int] = None) -> pd.DataFrame:
    """滚动逆波动率权重面板（与 :func:`rolling_weights` 同一再平衡节奏，同样防未来）。"""
    returns = prices.astype("float64").pct_change()
    dates = prices.index
    n = len(dates)
    est_window = max(5, int(est_window))
    if min_obs is None:
        min_obs = max(10, min(est_window // 2, max(10, n // 3)))
    w = pd.DataFrame(np.nan, index=dates, columns=prices.columns, dtype="float64")
    for i in range(min(est_window, max(n - 2, 0)), n, max(1, int(rebalance))):
        sub = returns.iloc[max(0, i - est_window + 1): i + 1].dropna(how="any")
        if len(sub) < int(min_obs) or sub.shape[1] < 2:
            continue
        vals = _safe(inverse_volatility_weights, sub, default=None)
        if vals is None:
            continue
        w.loc[dates[i]] = vals.reindex(w.columns).fillna(0.0).to_numpy()
    w = w.ffill().fillna(0.0)
    if lag:
        w = w.shift(int(lag))
    return w.fillna(0.0)


def rolling_weights(prices: pd.DataFrame, est_window: int = 120, rebalance: int = 21,
                    optimizer: str = "risk_parity", shrink_target: str = "const_corr",
                    risk_aversion: float = 5.0, lag: int = 1,
                    min_obs: Optional[int] = None, target_vol: Optional[float] = None,
                    cash_assets: Sequence[Any] = (),
                    rp_ladder: Sequence[Tuple[float, int]] = RP_LADDER,
                    annual: int = ANNUAL) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """滚动构建全资产组合的目标权重面板（只用截至再平衡日的历史）。

    每个再平衡日：取最近 ``est_window`` 期收益 → 剔除停牌等方差退化资产与
    ``cash_assets`` 现金池 → ``ledoit_wolf`` 收缩协方差 → ``solve_weights`` 求
    long-only 权重；``target_vol`` 非空时再用 ``kp.target_volatility`` 把组合整体
    缩放到目标波动，剩余敞口按等权放进现金池（无现金池时即持币，权重和 < 1）。
    非再平衡日沿用上一期权重，最后整体滞后 ``lag`` 期。

    量纲说明：``target_vol`` 是**年化**目标波动，而协方差是**每期**的，因此内部按
    ``target_vol / √annual`` 折算后再传给 ``kp.target_volatility``（该函数要求
    target_vol 与 cov 量纲一致），否则缩放永远不会触发。

    为什么要单独处理现金池：货币 ETF 的年化波动只有 ~0.1%，若与权益资产一起做
    风险平价，「等风险贡献」会把 >90% 的权重压给它，组合退化成现金。业界通行做法是
    把现金作为**残差池**而非风险平价的一条腿。

    返回 ``(weights, diagnostics)``；诊断表逐次记录再平衡日、样本数、参与优化的资产数、
    收缩强度 δ、实际求解器、年化波动、敞口与现金比例。
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

    cash_set = [c for c in cash_assets if c in set(prices.columns)]
    w = pd.DataFrame(np.nan, index=dates, columns=prices.columns, dtype="float64")
    diag: List[Dict[str, Any]] = []
    step = max(1, int(rebalance))
    start = min(est_window, max(n - 2, 0))
    for i in range(start, n, step):
        d = dates[i]
        hist = returns.iloc[max(0, i - est_window + 1): i + 1].dropna(how="any")
        if len(hist) < min_obs or hist.shape[1] < 2:
            continue
        risky = [c for c in tradable_assets(hist) if c not in set(cash_set)]
        if len(risky) < 2:
            continue
        sub = hist[risky]
        delta, solver, cov = float("nan"), optimizer, None
        try:
            lw = kp.ledoit_wolf(sub, target=shrink_target)
            cov, delta = lw.cov, float(lw.delta)
            w_i, solver, cov, delta = solve_weights(optimizer, sub, cov, risk_aversion,
                                                    delta=delta,
                                                    shrink_target=shrink_target,
                                                    rp_ladder=rp_ladder)
        except Exception:  # noqa: BLE001 - scipy 缺失或样本奇异时回退等权
            w_i, solver = pd.Series(kp.equal_weight(sub)), "equal_weight(fallback)"
        vals = pd.Series(w_i, dtype="float64").reindex(sub.columns).fillna(0.0)
        cash, exposure = 0.0, 1.0
        if target_vol and cov is not None:
            # 年化目标 -> 每期目标，与 cov 的量纲对齐（否则缩放永远不触发）
            tv_period = float(target_vol) / float(np.sqrt(annual))
            tv = _safe(kp.target_volatility, vals, cov, tv_period,
                       max_leverage=1.0, default=None)
            if tv is not None:
                vals = pd.Series(tv.weights, dtype="float64").reindex(vals.index).fillna(0.0)
                cash, exposure = _fin(tv.cash_weight), _fin(tv.exposure)
        row = pd.Series(0.0, index=w.columns, dtype="float64")
        row.loc[vals.index] = vals.to_numpy()
        if cash_set and cash > 0:                       # 现金池等权承接剩余敞口
            row.loc[cash_set] = row.loc[cash_set].to_numpy() + cash / len(cash_set)
            cash = 0.0
        w.loc[d] = row.to_numpy()
        vol = _fin(_safe(kp.portfolio_volatility, vals, cov, periods_per_year=ANNUAL)) \
            if cov is not None else float("nan")
        diag.append({"date": d, "n_obs": int(len(sub)), "n_risky": int(len(risky)),
                     "shrinkage_delta": delta, "solver": solver,
                     "ann_volatility": vol, "exposure": exposure, "cash_weight": cash,
                     "max_weight": float(row.max()) if len(row) else float("nan")})

    w = w.ffill().fillna(0.0)
    if lag:
        w = w.shift(int(lag))
    return w.fillna(0.0), pd.DataFrame(diag)


def static_weights(prices: pd.DataFrame, est_window: Optional[int] = None,
                   shrink_target: str = "const_corr",
                   cash_assets: Sequence[Any] = ()) -> Dict[str, pd.Series]:
    """期末一次性求解的静态权重对照（等权 / 逆波动 / 风险平价 / 最小方差 / 最大夏普）。

    ``cash_assets`` 中的现金池资产不参与求解（权重记 0），口径与滚动构建一致。
    风险平价走 :func:`risk_parity_chain` 的三级求解链，避免负相关资产导致无解。
    """
    returns = prices.astype("float64").pct_change().dropna()
    if est_window:
        returns = returns.tail(int(est_window))
    cash_set = {c for c in cash_assets}
    risky = [c for c in tradable_assets(returns) if c not in cash_set]
    if len(risky) < 2:
        risky = list(returns.columns)
    sub = returns[risky]

    def _pad(w: pd.Series) -> pd.Series:
        """把在 risky 子集上求得的权重补零对齐回全部资产。"""
        return pd.Series(w, dtype="float64").reindex(returns.columns).fillna(0.0)

    out: Dict[str, pd.Series] = {}
    out["equal_weight"] = _pad(pd.Series(kp.equal_weight(sub)))
    out["inverse_vol"] = _pad(inverse_volatility_weights(sub))
    lw = _safe(kp.ledoit_wolf, sub, target=shrink_target, default=None)
    cov = lw.cov if lw is not None else kp.sample_cov(sub)
    delta = float(lw.delta) if lw is not None else float("nan")
    rp_w = _safe(risk_parity_chain, sub, cov, delta=delta, shrink_target=shrink_target,
                 default=None)
    out["risk_parity"] = _pad(rp_w[0]) if rp_w is not None else pd.Series(np.nan)
    mv = _safe(kp.min_variance, cov, long_only=True, default=None)
    out["min_variance"] = _pad(pd.Series(mv)) if mv is not None else pd.Series(np.nan)
    mu = kp.mean_returns(sub)
    ms = _safe(kp.max_sharpe, mu, cov, long_only=True, default=None)
    out["max_sharpe"] = _pad(pd.Series(ms)) if ms is not None else pd.Series(np.nan)
    return {k: v for k, v in out.items() if v is not None and v.notna().any()}


def class_map(columns: Iterable[Any], asset_classes: Optional[Mapping[str, Sequence[str]]]
              ) -> Dict[str, str]:
    """构造 ``{资产: 类别}`` 映射；未归类的资产记为 ``"other"``。"""
    out: Dict[str, str] = {str(c): "other" for c in columns}
    for cls, syms in (asset_classes or {}).items():
        for s in syms:
            if str(s) in out:
                out[str(s)] = str(cls)
    return out


def default_asset_classes(columns: Iterable[Any]) -> Dict[str, List[str]]:
    """尝试用 ``kairos_data.ASSET_CLASSES`` 给面板列做资产类别归类。

    兄弟包不可用或列名对不上时返回空 dict（流水线照常运行，只是少一张类别表）。
    """
    try:
        kd = require("kairos_data")
        raw = dict(getattr(kd, "ASSET_CLASSES", {}) or {})
    except Exception:  # noqa: BLE001
        return {}
    cols = {str(c) for c in columns}
    return {k: [str(s) for s in v if str(s) in cols] for k, v in raw.items()}


def aggregate_by_class(weights: pd.Series, mapping: Mapping[str, str]) -> pd.Series:
    """把资产权重按类别聚合（用于「类别层」配置视角）。"""
    if not mapping:
        return pd.Series(dtype="float64")
    s = pd.Series(weights, dtype="float64")
    groups = pd.Series([mapping.get(str(k), "other") for k in s.index], index=s.index)
    return s.groupby(groups).sum().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# 回测与风险（与流水线 ① 同口径）
# ---------------------------------------------------------------------------
def backtest_stage(prices: pd.DataFrame, weights: Dict[str, pd.DataFrame],
                   cost_rate: float = 0.001, risk_free: float = 0.0,
                   annual: int = ANNUAL) -> Dict[str, Any]:
    """对多个权重面板跑 ``VectorBacktester`` 并汇总绩效/换手/净值。"""
    bt = kb.VectorBacktester(cost_rate=float(cost_rate))
    results, metrics, turnover, equity = {}, {}, {}, {}
    for name, w in weights.items():
        res = bt.run(prices, w)
        results[name] = res
        metrics[name] = res.metrics(risk_free=risk_free, periods_per_year=annual)
        turnover[name] = _fin(res.turnover.mean() * annual)
        equity[name] = res.equity.rename(name)
    return {
        "results": results,
        "metrics": metrics,
        "turnover_annualized": turnover,
        "equity": pd.DataFrame(equity),
        "returns": pd.DataFrame({k: v.returns.rename(k) for k, v in results.items()}),
        "cost_rate": float(cost_rate),
        "summary_frame": pd.DataFrame({k: pd.Series(v) for k, v in metrics.items()}).T
        if metrics else pd.DataFrame(),
    }


def risk_stage(returns: pd.DataFrame, prices: pd.DataFrame, last_weights: pd.Series,
               cov: Optional[pd.DataFrame], confidence: float = 0.95,
               annual: int = ANNUAL, n_trials: int = 5, main: str = "risk_parity",
               class_mapping: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """风险阶段：VaR/ES、回撤报告、PSR/DSR、成分风险与类别风险归集。"""
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
    out["modified_var"] = _fin(_safe(kr.modified_var, r, confidence))
    # POT 设定：用损失最差的 5% 拟合 GPD，再外推到 99% 置信度的 VaR/ES
    out["tail"] = _safe(kr.tail_summary, r, confidence=0.99, tail_quantile=0.95,
                        default=None)

    comp = _safe(kr.component_risk, last_weights, cov, default=None) if cov is not None else None
    out["component_risk"] = comp
    if comp is not None and class_mapping:
        pct = pd.Series(comp.pct, dtype="float64")
        groups = pd.Series([class_mapping.get(str(k), "other") for k in pct.index],
                           index=pct.index)
        out["class_risk_pct"] = pct.groupby(groups).sum().sort_values(ascending=False)
    return out


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def _asset_table(prices: pd.DataFrame, annual: int, mapping: Mapping[str, str]) -> pd.DataFrame:
    """逐资产统计表（含资产类别）。"""
    rets = prices.pct_change().dropna()
    cum = (1.0 + rets).prod() - 1.0
    vol = rets.std(ddof=1) * np.sqrt(annual)
    tbl = pd.DataFrame({
        "类别": pd.Series({str(c): mapping.get(str(c), "other") for c in prices.columns}),
        "累计收益": cum,
        "年化收益": (1.0 + cum) ** (annual / max(len(rets), 1)) - 1.0,
        "年化波动": vol,
        "夏普": ((1.0 + cum) ** (annual / max(len(rets), 1)) - 1.0) / vol.where(vol > 0),
        "最大回撤": pd.Series({c: _fin(_safe(kr.max_drawdown, rets[c])) for c in rets.columns}),
    })
    return tbl.sort_values("累计收益", ascending=False)


def _data_section(prices: pd.DataFrame, meta: Dict[str, Any], annual: int,
                  mapping: Mapping[str, str]) -> str:
    rets = prices.pct_change().dropna()
    corr = rets.corr()
    off = corr.to_numpy()[np.triu_indices_from(corr.to_numpy(), k=1)]
    info = {
        "资产数": int(prices.shape[1]),
        "交易日数": int(prices.shape[0]),
        "横截面平均相关系数": _fin(np.nanmean(off)) if off.size else float("nan"),
        "横截面相关系数(最小/最大)": f"{_fin(np.nanmin(off)):.3f} / {_fin(np.nanmax(off)):.3f}"
        if off.size else rp.NA,
        "等权组合年化波动": _fin(_safe(kr.volatility, rets.mean(axis=1),
                                     periods_per_year=float(annual))),
    }
    parts = [rp.dict_to_md(info, "指标", "数值"), "",
             "**逐资产概览**", "",
             rp.frame_to_md(_asset_table(prices, annual, mapping), floatfmt=".4f",
                            index_label="资产")]
    if len(corr):
        parts += ["", "**相关系数矩阵**", "", rp.frame_to_md(corr, floatfmt=".3f",
                                                          index_label="资产")]
    return "\n".join(parts)


def _portfolio_section(port: Dict[str, Any], static: Dict[str, pd.Series],
                       frontier: Optional[Any], annual: int) -> str:
    diag: pd.DataFrame = port["diagnostics"]
    scalars = {
        "协方差估计窗口(期)": int(port["est_window"]),
        "再平衡间隔(期)": int(port["rebalance"]),
        "优化器": port["optimizer"],
        "权重滞后(防未来)": int(port["weight_lag"]),
        "再平衡次数": int(len(diag)),
        "Ledoit-Wolf 收缩强度 δ 均值": port["shrinkage_delta_mean"],
        "Ledoit-Wolf 收缩强度 δ 末值": port["shrinkage_delta_last"],
        "目标年化波动": port["target_vol"] if port["target_vol"] else "未启用（满仓）",
        "现金池资产": ", ".join(port["cash_assets"]) or "无（全部资产参与风险平价）",
        "期末未投资比例": port["cash_weight_last"],
    }
    parts = [rp.dict_to_md(scalars, "指标", "数值")]
    if len(diag):
        parts += ["", "**滚动优化诊断（最近 8 次再平衡）**", "",
                  rp.frame_to_md(diag.tail(8).set_index("date"), floatfmt=".4f",
                                 index_label="再平衡日")]
    if static:
        tbl = pd.DataFrame(static)
        tbl.loc["合计"] = tbl.sum()
        parts += ["", "**期末静态权重对照**（同一段历史一次性求解）", "",
                  rp.frame_to_md(tbl, floatfmt=".4f", index_label="资产")]
        cls = port.get("class_weights")
        if cls is not None and len(cls):
            parts += ["", "**期末风险平价权重的资产类别归集**", "",
                      rp.series_to_md(cls, name="类别权重", index_label="类别")]
    last = port["last_weights"]
    pct = port["last_risk_pct"]
    if last is not None:
        tbl = pd.DataFrame({"期末权重": pd.Series(last)})
        if pct is not None:
            tbl["风险贡献占比"] = pd.Series(pct).reindex(tbl.index)
            tbl["贡献/权重比"] = tbl["风险贡献占比"] / tbl["期末权重"].where(
                tbl["期末权重"] > 1e-9)
        parts += ["", "**期末风险平价权重与成分风险**（风险贡献占比 ≈ 1/n 即达到风险平价）",
                  "", rp.frame_to_md(tbl.sort_values("期末权重", ascending=False),
                                    floatfmt=".4f", index_label="资产")]
    if frontier is not None:
        fr = getattr(frontier, "frame", None)
        if fr is not None:
            parts += ["", "**静态有效前沿**（long-only 均值-方差，全样本协方差；"
                          "target_return / volatility 均为**每期**口径，乘 252 / √252 得年化）",
                      "",
                      rp.frame_to_md(_safe(fr, default=pd.DataFrame()), floatfmt=".4f",
                                     index_label="点")]
    return "\n".join(parts)


def _backtest_section(bt: Dict[str, Any], annual: int) -> str:
    frame: pd.DataFrame = bt["summary_frame"]
    order = ["total_return", "cagr", "volatility", "sharpe", "sortino",
             "max_drawdown", "calmar", "win_rate", "profit_factor", "periods"]
    view = frame[[c for c in order if c in frame.columns]].T
    view = view.rename(index=METRIC_LABELS)
    if "样本期数" in view.index:
        view.loc["样本期数"] = view.loc["样本期数"].astype(int)
    parts = [f"回测口径：单边成本率 {bt['cost_rate']:.4%}，年化基准 {annual} 期/年，"
             f"目标权重每日再平衡（换手按 |目标 − 漂移后持仓| 计），"
             f"成交滞后一期（`held = w.shift(1)`）。", "",
             rp.frame_to_md(view, floatfmt=".4f", index_label="指标"), "",
             "**年化单边换手**", "",
             rp.dict_to_md(bt["turnover_annualized"], "组合", "年化换手")]
    eq: pd.DataFrame = bt["equity"]
    if len(eq):
        picks = pd.concat([eq.iloc[[0]], eq.iloc[[len(eq) // 4]], eq.iloc[[len(eq) // 2]],
                           eq.iloc[[3 * len(eq) // 4]], eq.iloc[[-1]]])
        parts += ["", "**净值曲线（五个时点）**", "",
                  rp.frame_to_md(picks, floatfmt=".4f", index_label="日期")]
    return "\n".join(parts)


def _risk_section(risk: Dict[str, Any], annual: int) -> str:
    c = risk["confidence"]
    scalars = {
        f"历史 VaR({c:.0%}, 每日)": risk["var"],
        f"期望损失 ES({c:.0%}, 每日)": risk["es"],
        f"修正 VaR(Cornish-Fisher, {c:.0%})": risk["modified_var"],
        f"VaR({c:.0%}, 年化)": risk["var_annualized"],
        f"ES({c:.0%}, 年化)": risk["es_annualized"],
        "最大回撤": risk["max_drawdown"],
        "年化波动": risk["volatility"],
        "年化夏普(几何)": risk["sharpe"],
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
    parts = [f"主分析对象：`{risk['main']}` 组合的每期净收益。", "",
             rp.dict_to_md(scalars, "指标", "数值")]
    rr: pd.DataFrame = risk.get("risk_report")
    if rr is not None and len(rr):
        keep = [x for x in ("annualized_return", "volatility", "sharpe", "historical_var",
                            "parametric_var_normal", "parametric_var_student_t",
                            "modified_var", "es_historical", "es_student_t",
                            "downside_deviation", "sortino", "skewness",
                            "excess_kurtosis") if x in rr.index]
        parts += ["", "**各组合风险体检（kairos_risk.risk_report）**", "",
                  rp.frame_to_md(rr.loc[keep], floatfmt=".4f", index_label="指标")]
    comp = risk.get("component_risk")
    if comp is not None:
        parts += ["", "**成分风险分解（欧拉分配，期末权重）**", "",
                  rp.dict_to_md({
                      "组合每期波动": _fin(comp.volatility),
                      "组合年化波动": _fin(comp.volatility) * np.sqrt(annual),
                      "加权平均波动": _fin(comp.weighted_volatility),
                      "分散化比率": _fin(comp.diversification_ratio),
                      "风险有效资产数": _fin(comp.effective_n),
                      "风险 Herfindahl": _fin(comp.herfindahl),
                  }, "指标", "数值")]
    cls = risk.get("class_risk_pct")
    if cls is not None and len(cls):
        parts += ["", "**按资产类别归集的风险贡献占比**", "",
                  rp.series_to_md(cls, name="风险占比", index_label="类别")]
    tail = risk.get("tail")
    if tail is not None and len(tail):
        parts += ["", "**极值理论（EVT/GPD）尾部摘要**：把损失尾部拟合广义帕累托分布后"
                      "外推的 VaR/ES，与经验值对比可看出尾部厚度。", "",
                  rp.frame_to_md(tail, floatfmt=".5f", index_label="指标")]
    return "\n".join(parts)


def build_report(meta: Dict[str, Any], prices: pd.DataFrame, port: Dict[str, Any],
                 static: Dict[str, pd.Series], frontier: Optional[Any],
                 bt: Dict[str, Any], risk: Dict[str, Any], annual: int = ANNUAL) -> str:
    """渲染中文 Markdown 报告正文（不含一级标题）。"""
    head = [
        f"- 生成时间：{meta['generated_at']}",
        f"- 数据来源：{meta['source']}",
        f"- 样本区间：{meta['start']} ~ {meta['end']}（{meta['n_days']} 个交易日 × {meta['n_assets']} 个资产）",
        f"- 流水线参数：est_window={meta['est_window']}, rebalance={meta['rebalance']}, "
        f"optimizer={meta['optimizer']}, cost_rate={meta['cost_rate']:.4%}, "
        f"target_vol={meta['target_vol'] or '未启用'}, "
        f"cash_assets={','.join(meta['cash_assets']) or '无'}",
        "- 依赖版本：" + " / ".join(f"{k} {v}" for k, v in meta["versions"].items()),
    ]
    mapping = meta["class_map"]
    sections = [
        ("", "\n".join(head)),
        ("数据（kairos_data）", _data_section(prices, meta, annual, mapping)),
        ("组合（kairos_portfolio）", _portfolio_section(port, static, frontier, annual)),
        ("回测（kairos_backtest）", _backtest_section(bt, annual)),
        ("风险（kairos_risk）", _risk_section(risk, annual)),
        ("结论与口径说明", rp.bullets([
            "风险平价按**成分风险贡献相等**分配权重，因此高波动资产（如成长/海外权益）"
            "权重被压低，低波动资产（国债）权重被抬高，天然具备「全天候」的防御属性。",
            "现金类资产（货币 ETF）**不参与**风险平价求解：其年化波动仅 ~0.1%，"
            "若一并等风险贡献会把 >90% 权重吸走，组合退化为持币。它被留作"
            "``target_vol`` 波动率叠加的残差池；未启用目标波动时其权重为 0。",
            "在**不加杠杆**（``max_leverage=1``）的约束下，「等风险贡献」会把绝大部分"
            "权重压给低波动的国债 ETF（本样本约九成），组合波动被压得很低、夏普很高，"
            "但绝对收益有限。这是风险平价的固有属性而非实现缺陷：真实的全天候组合靠"
            "**债券加杠杆**把总波动抬到目标水平。本流水线提供两个出口：用 ``--target-vol`` "
            "配合 ``max_leverage>1`` 做波动率目标化，或用 ``kp.risk_parity(cov, risk_budget=...)`` "
            "显式指定各资产/类别的风险预算。",
            "求解器标签里的 ``δ=1`` 表示该次再平衡的样本相关结构下风险平价**无 long-only "
            "内部解**（黄金/国债与权益显著负相关，边际风险贡献为负），已自动改用「相关系数"
            "全收缩到常相关目标」的协方差重解，仍属 Ledoit-Wolf 家族，未静默降级为等权。",
            "Ledoit-Wolf 收缩强度 δ 越大，说明样本协方差越不可信、越向结构化目标收缩；"
            "资产数少而样本短时常出现 δ 接近 0 或 1 的极端值。",
            "所有权重都在再平衡日**只用截至当日的历史**求解，并整体滞后一期生效；"
            "回测器再做一期成交滞后，因此不存在未来函数。",
            f"本样本仅 {meta['n_assets']} 个资产、{meta['n_days']} 个交易日，"
            "指标（尤其 PSR/DSR）受样本长度限制，仅用于演示流水线，不构成投资建议。",
        ])),
        ("如何复现", rp.code_block(
            "python examples/run_e2e_multi_asset.py --data-dir <kairos-data>/data/etf", "bash")),
    ]
    body: List[str] = []
    for heading, text in sections:
        if heading:
            body.append(f"## {heading}")
        body.append(text.strip())
    return "\n\n".join(body).strip() + "\n"


def _metrics_payload(meta: Dict[str, Any], port: Dict[str, Any],
                     static: Dict[str, pd.Series], bt: Dict[str, Any],
                     risk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "meta": meta,
        "portfolio": {
            "est_window": port["est_window"],
            "rebalance": port["rebalance"],
            "optimizer": port["optimizer"],
            "weight_lag": port["weight_lag"],
        "target_vol": port["target_vol"],
            "cash_assets": port["cash_assets"],
            "n_rebalance": int(len(port["diagnostics"])),
            "shrinkage_delta_mean": port["shrinkage_delta_mean"],
            "shrinkage_delta_last": port["shrinkage_delta_last"],
            "cash_weight_last": port["cash_weight_last"],
            "exposure_mean": port["exposure_mean"],
            "last_weights": port["last_weights"],
            "last_risk_pct": port["last_risk_pct"],
            "class_weights": port.get("class_weights"),
            "static_weights": pd.DataFrame(static) if static else None,
            "diagnostics": port["diagnostics"],
        },
        "backtest": {
            "cost_rate": bt["cost_rate"],
            "metrics": bt["metrics"],
            "turnover_annualized": bt["turnover_annualized"],
            "summary": bt["summary_frame"],
        },
        "risk": {
            "confidence": risk["confidence"],
            "var": risk["var"],
            "es": risk["es"],
            "modified_var": risk["modified_var"],
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
            "component_risk": {
                "volatility": _fin(getattr(risk.get("component_risk"), "volatility", np.nan)),
                "diversification_ratio": _fin(getattr(risk.get("component_risk"),
                                                      "diversification_ratio", np.nan)),
                "effective_n": _fin(getattr(risk.get("component_risk"), "effective_n", np.nan)),
            },
            "class_risk_pct": risk.get("class_risk_pct"),
            "tail_summary": risk.get("tail"),
            "risk_report": risk["risk_report"],
        },
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_multi_asset_pipeline(prices: pd.DataFrame,
                             volumes: Optional[pd.DataFrame] = None,
                             est_window: int = 120, rebalance: int = 21,
                             cost_rate: float = 0.001, out_dir: Optional[str] = None,
                             optimizer: str = "risk_parity",
                             shrink_target: str = "const_corr",
                             risk_aversion: float = 5.0, weight_lag: int = 1,
                             target_vol: Optional[float] = None,
                             confidence: float = 0.95, annual: int = ANNUAL,
                             risk_free: float = 0.0, n_trials: int = 5,
                             asset_classes: Optional[Mapping[str, Sequence[str]]] = "auto",
                             cash_assets: Any = "auto", n_frontier_points: int = 8,
                             source: str = "prices panel",
                             name: str = "e2e_multi_asset") -> Dict[str, Any]:
    """跑完整的跨资产「全天候」配置流水线。

    参数
    ----
    prices:        ETF 价格面板 DataFrame(index=日期, columns=资产)。
    volumes:       可选成交量面板（仅写入报告的数据小节）。
    est_window:    协方差估计窗口（期）。
    rebalance:     再平衡间隔（期）。
    cost_rate:     单边换手成本率。
    out_dir:       非空时写出 ``REPORT.md`` / ``equity.csv`` / ``metrics.json`` / ``weights.csv``。
    optimizer:     ``risk_parity`` / ``mean_variance`` / ``min_variance`` / ``max_sharpe`` /
                   ``equal_weight``。
    shrink_target: Ledoit-Wolf 收缩目标（``const_corr`` / ``diag``）。
    risk_aversion: ``mean_variance`` / ``max_sharpe`` 的 λ。
    weight_lag:    权重滞后（防未来）。
    target_vol:    非空时把组合缩放到该**年化**目标波动（内部折算到每期），
                   不足部分放进现金池或持币。
    confidence:    VaR/ES 置信度。
    annual:        年化期数。
    n_trials:      DSR 试验次数。
    asset_classes: ``{类别: [资产...]}``；``"auto"`` 时尝试用 ``kairos_data.ASSET_CLASSES``，
                   ``None`` 表示不做类别归集。
    cash_assets:   现金池资产（不参与风险平价，仅作 ``target_vol`` 叠加的残差承接）；
                   ``"auto"`` 时取 ``kairos_data.ASSET_CLASSES["cash"]`` 与面板列的交集，
                   ``None`` / 空序列表示不设现金池（全部资产一起做风险平价）。
    n_frontier_points: 静态有效前沿的采样点数（缺 scipy 时自动跳过）。
    source:        数据来源描述（写入报告）。
    name:          流水线名。

    返回 dict，键为 ``meta`` / ``portfolio`` / ``static`` / ``frontier`` /
    ``backtest`` / ``risk`` / ``artifacts``。
    """
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] < 2:
        raise ValueError("prices 必须是至少含 2 个资产的 DataFrame(index=日期, columns=资产)")
    prices = prices.astype("float64").sort_index()
    if volumes is not None:
        volumes = volumes.reindex(index=prices.index, columns=prices.columns)

    if isinstance(asset_classes, str) and asset_classes == "auto":
        classes: Dict[str, List[str]] = default_asset_classes(prices.columns)
    else:
        classes = {str(k): [str(s) for s in v] for k, v in (asset_classes or {}).items()}
    mapping = class_map(prices.columns, classes)
    if isinstance(cash_assets, str) and cash_assets == "auto":
        cash_list: List[Any] = [c for c in classes.get("cash", []) if c in mapping]
    else:
        cash_list = [c for c in (cash_assets or []) if c in mapping]

    w_rp, diag = rolling_weights(prices, est_window=est_window, rebalance=rebalance,
                                 optimizer=optimizer, shrink_target=shrink_target,
                                 risk_aversion=risk_aversion, lag=weight_lag,
                                 target_vol=target_vol, cash_assets=cash_list,
                                 annual=annual)
    returns = prices.pct_change()
    w_eq = pd.DataFrame(np.full((len(prices), prices.shape[1]), 1.0 / prices.shape[1]),
                        index=prices.index, columns=prices.columns)
    w_iv = rolling_inverse_vol(prices, est_window=est_window, rebalance=rebalance,
                               lag=weight_lag)

    port: Dict[str, Any] = {
        "risk_parity_weights": w_rp,
        "equal_weight_weights": w_eq,
        "inverse_vol_weights": w_iv,
        "diagnostics": diag,
        "est_window": int(est_window),
        "rebalance": int(rebalance),
        "optimizer": optimizer,
        "weight_lag": int(weight_lag),
        "target_vol": float(target_vol) if target_vol else None,
        "cash_assets": [str(c) for c in cash_list],
        "shrinkage_delta_mean": _fin(diag["shrinkage_delta"].mean()) if len(diag) else float("nan"),
        "shrinkage_delta_last": _fin(diag["shrinkage_delta"].iloc[-1]) if len(diag) else float("nan"),
        "cash_weight_last": _fin(diag["cash_weight"].iloc[-1]) if len(diag) and "cash_weight" in diag else 0.0,
        "exposure_mean": _fin(diag["exposure"].mean()) if len(diag) and "exposure" in diag else 1.0,
    }
    last = _last_active(w_rp if w_rp.abs().sum().sum() > 0 else w_eq)
    port["last_weights"] = last
    tail = returns.tail(max(20, int(est_window) // 2)).dropna(how="any")
    lw = _safe(kp.ledoit_wolf, tail, target=shrink_target, default=None)
    cov_last = lw.cov if lw is not None else None
    port["last_cov"] = cov_last
    rc = _safe(kp.risk_contributions, pd.Series(last), cov_last, default=None) \
        if cov_last is not None else None
    port["last_risk_pct"] = pd.Series(rc.pct) if rc is not None else None
    port["class_weights"] = aggregate_by_class(last, mapping)

    static = static_weights(prices, est_window=est_window, shrink_target=shrink_target,
                            cash_assets=cash_list)
    frontier = None
    if cov_last is not None:
        mu_full = kp.mean_returns(returns.dropna(how="any"))
        lw_full = _safe(kp.ledoit_wolf, returns.dropna(how="any"), target=shrink_target,
                        default=None)
        if lw_full is not None:
            frontier = _safe(kp.efficient_frontier, mu_full, lw_full.cov,
                             n_points=int(n_frontier_points), long_only=True, default=None)

    bt = backtest_stage(prices, {
        optimizer: w_rp,
        "equal_weight": w_eq,
        "inverse_vol": w_iv,
    }, cost_rate=cost_rate, risk_free=risk_free, annual=annual)
    risk = risk_stage(bt["returns"], prices, last, cov_last, confidence=confidence,
                      annual=annual, n_trials=n_trials, main=optimizer,
                      class_mapping=mapping)

    meta: Dict[str, Any] = {
        "pipeline": name,
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "n_assets": int(prices.shape[1]),
        "n_days": int(prices.shape[0]),
        "start": str(pd.Timestamp(prices.index[0]).date()),
        "end": str(pd.Timestamp(prices.index[-1]).date()),
        "est_window": int(est_window),
        "rebalance": int(rebalance),
        "optimizer": optimizer,
        "target_vol": float(target_vol) if target_vol else None,
        "cash_assets": [str(c) for c in cash_list],
        "cost_rate": float(cost_rate),
        "weight_lag": int(weight_lag),
        "annual": int(annual),
        "class_map": mapping,
        "assets": [str(c) for c in prices.columns],
        "versions": {m.__name__: getattr(m, "__version__", "?") for m in (kp, kb, kr)},
    }

    artifacts: Dict[str, str] = {}
    if out_dir:
        d = rp.ensure_dir(out_dir)
        text = build_report(meta, prices, port, static, frontier, bt, risk, annual)
        artifacts["report_md"] = rp.write_text(
            os.path.join(d, "REPORT.md"),
            f"# Kairos Lab · {name} 跨资产全天候配置报告\n\n{text}")
        eq = bt["equity"].copy()
        eq["risk_parity_drawdown"] = kr.drawdown_series(bt["results"][optimizer].returns)
        artifacts["equity_csv"] = rp.write_csv(eq, os.path.join(d, "equity.csv"))
        artifacts["metrics_json"] = rp.write_json(
            os.path.join(d, "metrics.json"),
            _metrics_payload(meta, port, static, bt, risk))
        artifacts["weights_csv"] = rp.write_csv(w_rp, os.path.join(d, "weights.csv"))

    return {"meta": meta, "portfolio": port, "static": static, "frontier": frontier,
            "backtest": bt, "risk": risk, "artifacts": artifacts}


def summarize(result: Dict[str, Any]) -> str:
    """把流水线结果压成一段可直接 print 的中文摘要。"""
    meta, port, bt, risk = result["meta"], result["portfolio"], result["backtest"], result["risk"]
    lines = [
        f"[{meta['pipeline']}] {meta['source']} | {meta['start']}~{meta['end']} "
        f"| {meta['n_days']}日 × {meta['n_assets']}资产",
        f"  组合 {meta['optimizer']}: 窗口={meta['est_window']} 调仓={meta['rebalance']}期 "
        f"再平衡{len(port['diagnostics'])}次 δ均值={port['shrinkage_delta_mean']:.3f} "
        f"平均敞口={port['exposure_mean']:.1%} 期末未投资={port['cash_weight_last']:.1%}"
        + (f" 目标波动={meta['target_vol']:.1%}" if meta['target_vol'] else " 目标波动=未启用"),
    ]
    for k, m in bt["metrics"].items():
        lines.append(f"  回测 {k:<13} 累计={m['total_return']:+.2%} 年化={m['cagr']:+.2%} "
                     f"波动={m['volatility']:.2%} 夏普={m['sharpe']:+.3f} "
                     f"回撤={m['max_drawdown']:.2%} 换手={bt['turnover_annualized'][k]:.1f}x/年")
    comp = risk.get("component_risk")
    lines.append(f"  风险 VaR({risk['confidence']:.0%})={risk['var']:.4f} ES={risk['es']:.4f} "
                 f"修正VaR={risk['modified_var']:.4f} 最大回撤={risk['max_drawdown']:.2%} "
                 f"PSR={risk['psr']:.3f} DSR={risk['dsr_value']:.3f}"
                 + (f" 分散化比率={_fin(comp.diversification_ratio):.3f} "
                    f"有效资产数={_fin(comp.effective_n):.2f}" if comp is not None else ""))
    cls = port.get("class_weights")
    if cls is not None and len(cls):
        lines.append("  期末类别权重: " + ", ".join(f"{k}={v:.1%}" for k, v in cls.items()))
    if result["artifacts"]:
        lines.append(f"  产物: {', '.join(os.path.basename(v) for v in result['artifacts'].values())}"
                     f" -> {os.path.dirname(result['artifacts']['report_md'])}")
    return "\n".join(lines)
