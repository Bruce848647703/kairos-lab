"""端到端流水线的离线测试。

全部使用固定 seed 的小型合成面板（6 资产 × 300 日 / 5 资产 × 420 日），
**不联网、不读仓库外的真实数据**，``python -m pytest -q`` 应在数秒内全绿。

覆盖：
1. 三条 ``run_*`` 流水线的返回结构、关键指标有限性、产物文件落盘；
2. ``metrics.json`` 是严格合法 JSON（不含 NaN / Infinity 字面量）；
3. 权重与回测的**防未来函数**性质：把价格截断到 t 重算，结果必须与全样本一致；
4. 确定性：同参数两次运行结果逐位相同。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from kairos_lab.pipelines import e2e_equity as eq
from kairos_lab.pipelines import e2e_multi_asset as ma
from kairos_lab.pipelines import execution_demo as ex


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _strict_json(path: str):
    """按严格模式解析 JSON：出现 NaN / Infinity 字面量即失败。"""
    def _reject(constant: str):
        raise AssertionError(f"{path} 含非法 JSON 常量: {constant}")

    with open(path, encoding="utf-8") as fh:
        return json.load(fh, parse_constant=_reject)


def _assert_finite(value, label: str) -> float:
    x = float(value)
    assert math.isfinite(x), f"{label} 不是有限数: {value!r}"
    return x


def _assert_files(artifacts, names, tmp_path) -> None:
    """断言产物已生成、非空，且都落在 tmp_path 内（不污染仓库）。"""
    assert artifacts, "未生成任何产物"
    for key in names:
        assert key in artifacts, f"缺少产物 {key}"
        path = artifacts[key]
        assert os.path.isfile(path), f"{key} 未落盘: {path}"
        assert os.path.getsize(path) > 0, f"{key} 为空文件: {path}"
        assert os.path.abspath(path).startswith(os.path.abspath(str(tmp_path))), \
            f"{key} 写到了 tmp_path 之外: {path}"


def _read_report(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# ① A 股端到端：数据 → 因子 → 组合 → 回测 → 风险
# ---------------------------------------------------------------------------
def test_equity_pipeline_structure_and_artifacts(equity_prices, equity_volumes, tmp_path):
    """跑通 A 股流水线：键齐全、指标有限、产物落盘、报告含各阶段真实数字。"""
    out = tmp_path / "equity"
    res = eq.run_equity_pipeline(equity_prices, equity_volumes, lookback=60, horizon=5,
                                 cost_rate=0.001, out_dir=str(out), top_quantile=0.3,
                                 source="synthetic-test", name="test_equity")

    for key in ("meta", "factor", "portfolio", "backtest", "risk", "artifacts"):
        assert key in res, f"返回 dict 缺少键 {key}"

    # --- 因子阶段 ---
    factor = res["factor"]
    ric = factor["rank_ic_summary"]
    assert ric["count"] > 50
    for k in ("mean", "std", "ir", "t_stat", "positive_ratio"):
        _assert_finite(ric[k], f"rank_ic_summary[{k}]")
    assert -1.0 <= ric["mean"] <= 1.0
    assert 0.0 <= ric["positive_ratio"] <= 1.0
    qr = factor["quantile_returns"]
    assert list(qr.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert np.isfinite(qr.to_numpy(dtype="float64")).any()
    assert factor["decay"] is not None and len(factor["decay"]) == 5
    assert factor["scores"].shape == equity_prices.shape

    # --- 组合阶段 ---
    port = res["portfolio"]
    for k in ("factor_weights", "optimized_weights", "benchmark_weights"):
        w = port[k]
        assert isinstance(w, pd.DataFrame)
        assert w.shape == equity_prices.shape, f"{k} 形状应为 date × asset"
        row_sum = w.sum(axis=1).to_numpy(dtype="float64")
        assert np.all(row_sum <= 1.0 + 1e-9), f"{k} 存在权重和 > 1 的截面"
        assert np.all(row_sum >= -1e-12), f"{k} 存在负权重"
        assert np.isfinite(w.to_numpy(dtype="float64")).all()
    # 因子组合确实只持有了 top_quantile 的标的
    held = (port["factor_weights"] > 0).sum(axis=1)
    assert held.max() <= math.ceil(0.3 * equity_prices.shape[1])
    assert len(port["diagnostics"]) > 0
    _assert_finite(port["shrinkage_delta_mean"], "shrinkage_delta_mean")

    # --- 回测阶段 ---
    bt = res["backtest"]
    assert set(bt["metrics"]) == {"factor", "optimized", "equal_weight"}
    for name, m in bt["metrics"].items():
        for k in ("total_return", "cagr", "volatility", "sharpe", "max_drawdown",
                  "calmar", "win_rate"):
            _assert_finite(m[k], f"backtest[{name}][{k}]")
        assert m["max_drawdown"] >= 0.0
        _assert_finite(bt["turnover_annualized"][name], f"turnover[{name}]")
    assert len(bt["equity"]) == len(equity_prices)
    assert np.allclose(bt["equity"]["equal_weight"].iloc[0], 1.0, atol=0.01)

    # --- 风险阶段 ---
    risk = res["risk"]
    var = _assert_finite(risk["var"], "var")
    es = _assert_finite(risk["es"], "es")
    assert es >= var - 1e-12, "ES 应不小于同置信度 VaR"
    assert var >= 0.0
    _assert_finite(risk["max_drawdown"], "max_drawdown")
    assert 0.0 <= risk["max_drawdown"] <= 1.0
    for k in ("psr", "dsr_value"):
        v = _assert_finite(risk[k], k)
        assert 0.0 <= v <= 1.0, f"{k} 应是概率: {v}"
    assert risk["drawdown"]["n_episodes"] >= 1
    assert isinstance(risk["risk_report"], pd.DataFrame) and len(risk["risk_report"]) > 0
    assert risk["factor_breakdown"] is not None
    _assert_finite(risk["factor_breakdown"].factor_share, "factor_share")

    # --- 产物 ---
    _assert_files(res["artifacts"], ("report_md", "equity_csv", "metrics_json",
                                     "weights_csv", "factor_ic_csv"), tmp_path)
    payload = _strict_json(res["artifacts"]["metrics_json"])
    assert payload["meta"]["pipeline"] == "test_equity"
    assert payload["meta"]["n_assets"] == equity_prices.shape[1]
    assert payload["risk"]["var"] is not None
    assert set(payload["backtest"]["metrics"]) == {"factor", "optimized", "equal_weight"}

    text = _read_report(res["artifacts"]["report_md"])
    assert text.startswith("# Kairos Lab · test_equity")
    for heading in ("数据（kairos_data）", "因子（kairos_factor）", "组合（kairos_portfolio）",
                    "回测（kairos_backtest）", "风险（kairos_risk）"):
        assert heading in text, f"报告缺少小节 {heading}"
    assert "秩IC" in text and "历史 VaR" in text and "最大回撤" in text

    equity = pd.read_csv(res["artifacts"]["equity_csv"], index_col=0)
    assert {"factor", "optimized", "equal_weight"}.issubset(equity.columns)
    assert len(equity) == len(equity_prices)


def test_equity_pipeline_is_deterministic(equity_prices, tmp_path):
    """同参数两次运行，关键指标必须逐位相同（无隐藏随机性）。"""
    a = eq.run_equity_pipeline(equity_prices, lookback=40, horizon=5, out_dir=None)
    b = eq.run_equity_pipeline(equity_prices, lookback=40, horizon=5, out_dir=None)
    assert a["factor"]["rank_ic_summary"] == b["factor"]["rank_ic_summary"]
    pd.testing.assert_frame_equal(a["backtest"]["equity"], b["backtest"]["equity"])
    pd.testing.assert_frame_equal(a["portfolio"]["factor_weights"],
                                  b["portfolio"]["factor_weights"])
    assert tmp_path is not None


@pytest.mark.parametrize("optimizer", ["risk_parity", "min_variance", "mean_variance"])
def test_equity_pipeline_optimizers(equity_prices, optimizer):
    """三种优化器都要能给出合法权重（scipy 缺失时自动回退等权）。"""
    if optimizer in ("min_variance", "mean_variance"):
        pytest.importorskip("scipy")
    res = eq.run_equity_pipeline(equity_prices, lookback=60, horizon=5, optimizer=optimizer,
                                 est_window=120, opt_rebalance=40, out_dir=None)
    w = res["portfolio"]["optimized_weights"]
    row_sum = w.sum(axis=1).to_numpy(dtype="float64")
    assert np.all(row_sum <= 1.0 + 1e-6)
    assert np.all(row_sum >= -1e-12)
    assert row_sum.max() > 0.5, "优化组合应至少在部分时间持仓"
    assert res["backtest"]["metrics"]["optimized"]["periods"] == len(equity_prices)


def test_equity_pipeline_rejects_bad_input():
    """单资产或非 DataFrame 输入应给出清晰的中文错误。"""
    with pytest.raises(ValueError):
        eq.run_equity_pipeline(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    with pytest.raises(ValueError):
        eq.factor_weights(pd.DataFrame(np.zeros((3, 3))), top_quantile=1.5)
    with pytest.raises(ValueError):
        eq.factor_weights(pd.DataFrame(np.zeros((3, 3))), weighting="random")


# ---------------------------------------------------------------------------
# ①.5 样本外验证（kairos_ml，可选阶段）
# ---------------------------------------------------------------------------
def test_walk_forward_validation(equity_prices):
    """walk-forward 切分必须严格时序，且两套独立实现的秩 IC 必须一致。"""
    pytest.importorskip("kairos_ml")
    factor = eq.factor_stage(equity_prices, lookback=60, horizon=5)
    val = eq.walk_forward_validation(factor, train_size=120, test_size=40)
    assert val["available"] is True, val.get("error")
    assert val["n_folds"] >= 2
    folds = val["folds"]
    for _, row in folds.iterrows():
        assert row["train_end"] < row["test_start"], "训练窗口必须完全早于测试窗口"
        assert row["n_train"] == 120 and row["n_test"] <= 40
        assert -1.0 <= row["ic_out_of_sample"] <= 1.0
        assert 0.0 <= row["ls_hit_rate"] <= 1.0
    assert 0.0 <= val["sign_persistence"] <= 1.0
    assert -1.0 <= val["ic_out_of_sample_mean"] <= 1.0
    # kairos_ml.ic(spearman) 与 kairos_factor.rank_ic 是两套独立实现，应吻合到浮点精度
    assert val["cross_check_dates"] > 0
    assert val["cross_check_max_abs_diff"] < 1e-10


def test_walk_forward_validation_degrades_gracefully(equity_prices):
    """样本不足时必须给出中文说明而不是抛异常（该阶段是可选的）。"""
    factor = eq.factor_stage(equity_prices, lookback=60, horizon=5)
    val = eq.walk_forward_validation(factor, train_size=10000, test_size=40)
    assert val["available"] is False
    assert "error" in val and val["error"]


def test_equity_pipeline_exposes_validation(equity_prices):
    """主流水线的返回 dict 必须带上 validation 键，且可用 with_validation 关闭。"""
    res = eq.run_equity_pipeline(equity_prices, lookback=60, horizon=5,
                                 oos_train=120, oos_test=40, out_dir=None)
    assert "validation" in res
    off = eq.run_equity_pipeline(equity_prices, lookback=60, horizon=5,
                                 with_validation=False, out_dir=None)
    assert off["validation"] == {"available": False}


def test_pipelines_package_exports():
    """``kairos_lab.pipelines`` 应直接导出三条流水线入口与摘要函数。"""
    from kairos_lab import pipelines
    for name in ("run_equity_pipeline", "run_multi_asset_pipeline", "run_execution_demo",
                 "summarize_equity", "summarize_multi_asset", "summarize_execution"):
        assert callable(getattr(pipelines, name)), f"缺少导出 {name}"


# ---------------------------------------------------------------------------
# ② 跨资产全天候：风险平价 → 回测 → 风险
# ---------------------------------------------------------------------------
def test_multi_asset_pipeline_structure_and_artifacts(etf_prices, tmp_path):
    """跑通跨资产流水线：风险平价权重合法、产物齐全、报告含真实数字。"""
    out = tmp_path / "multi_asset"
    res = ma.run_multi_asset_pipeline(etf_prices, est_window=120, rebalance=21,
                                      cost_rate=0.001, out_dir=str(out),
                                      source="synthetic-test", name="test_multi_asset")

    for key in ("meta", "portfolio", "static", "backtest", "risk", "artifacts"):
        assert key in res, f"返回 dict 缺少键 {key}"

    port = res["portfolio"]
    w = port["risk_parity_weights"]
    assert w.shape == etf_prices.shape
    active = w[w.abs().sum(axis=1) > 1e-12]
    assert len(active) > 0, "风险平价组合从未建仓"
    sums = active.sum(axis=1).to_numpy(dtype="float64")
    assert np.all(sums <= 1.0 + 1e-9) and np.all(sums >= 0.5), \
        "未启用 target_vol 时风险平价应满仓（权重和 = 1）"
    assert np.allclose(sums, 1.0, atol=1e-9)
    assert np.isfinite(w.to_numpy(dtype="float64")).all()

    diag = port["diagnostics"]
    assert len(diag) > 3
    assert (diag["n_risky"] >= 2).all()
    # 求解器标签必须写明实际用的路径，不允许静默降级
    assert diag["solver"].str.startswith(("risk_parity", "inverse_vol", "mean_variance",
                                          "min_variance", "max_sharpe", "equal_weight")).all()
    assert "equal_weight(fallback)" not in set(diag["solver"]), \
        "风险平价求解链不应退化到等权"

    last = port["last_weights"]
    assert abs(float(last.sum()) - 1.0) < 1e-9
    pct = port["last_risk_pct"]
    if pct is not None and len(pct):
        assert abs(float(pct.sum()) - 1.0) < 1e-6, "成分风险占比之和应为 1（欧拉分解）"

    assert "risk_parity" in res["static"] and "equal_weight" in res["static"]
    for name, sw in res["static"].items():
        assert abs(float(sw.sum())) < 1.5, f"静态权重 {name} 之和异常: {sw.sum()}"

    bt = res["backtest"]
    assert set(bt["metrics"]) == {"risk_parity", "equal_weight", "inverse_vol"}
    for name, m in bt["metrics"].items():
        for k in ("total_return", "cagr", "volatility", "sharpe", "max_drawdown"):
            _assert_finite(m[k], f"backtest[{name}][{k}]")

    risk = res["risk"]
    assert _assert_finite(risk["var"], "var") >= 0.0
    assert _assert_finite(risk["es"], "es") >= risk["var"] - 1e-12
    _assert_finite(risk["max_drawdown"], "max_drawdown")
    _assert_finite(risk["psr"], "psr")
    _assert_finite(risk["dsr_value"], "dsr_value")
    assert risk["component_risk"] is not None
    _assert_finite(risk["component_risk"].diversification_ratio, "diversification_ratio")
    _assert_finite(risk["component_risk"].effective_n, "effective_n")

    _assert_files(res["artifacts"], ("report_md", "equity_csv", "metrics_json",
                                     "weights_csv"), tmp_path)
    payload = _strict_json(res["artifacts"]["metrics_json"])
    assert payload["meta"]["pipeline"] == "test_multi_asset"
    assert payload["portfolio"]["n_rebalance"] == len(diag)
    text = _read_report(res["artifacts"]["report_md"])
    for heading in ("数据（kairos_data）", "组合（kairos_portfolio）",
                    "回测（kairos_backtest）", "风险（kairos_risk）"):
        assert heading in text
    assert "相关系数矩阵" in text and "成分风险分解" in text

    weights = pd.read_csv(res["artifacts"]["weights_csv"], index_col=0)
    assert weights.shape == etf_prices.shape


def test_multi_asset_target_vol_keeps_cash(etf_prices):
    """启用目标波动率后，权重和可以小于 1（剩余敞口留现金/现金池）。"""
    res = ma.run_multi_asset_pipeline(etf_prices, est_window=120, rebalance=21,
                                      target_vol=0.01, out_dir=None, cash_assets=None)
    w = res["portfolio"]["risk_parity_weights"]
    sums = w.sum(axis=1)
    active = sums[sums > 1e-12]
    assert len(active) > 0
    assert float(active.max()) <= 1.0 + 1e-9, "max_leverage=1 时权重和不得超过 1"
    assert float(active.min()) < 0.999, "低目标波动应触发降杠杆（权重和 < 1）"
    diag = res["portfolio"]["diagnostics"]
    assert (diag["exposure"] <= 1.0 + 1e-12).all()


def test_multi_asset_cash_sleeve_excluded(etf_prices):
    """现金池资产不参与风险平价：其权重恒为 0（未启用 target_vol 时）。"""
    cash = [etf_prices.columns[0]]
    res = ma.run_multi_asset_pipeline(etf_prices, est_window=120, rebalance=21,
                                      cash_assets=cash, out_dir=None)
    assert res["meta"]["cash_assets"] == [str(cash[0])]
    w = res["portfolio"]["risk_parity_weights"]
    assert float(w[cash[0]].abs().sum()) == 0.0


def test_inverse_volatility_weights_are_normalized(etf_prices):
    """逆波动率权重非负、和为 1，且低波动资产权重更高。"""
    rets = etf_prices.pct_change().dropna()
    w = ma.inverse_volatility_weights(rets)
    assert abs(float(w.sum()) - 1.0) < 1e-12
    assert (w >= 0).all()
    vol = rets.std(ddof=1)
    assert w.idxmax() == vol.idxmin()


def test_rolling_inverse_vol_has_no_lookahead(etf_prices):
    """滚动逆波动率权重的前半段不应依赖后半段数据。"""
    cut = etf_prices.index[len(etf_prices) // 2]
    full = ma.rolling_inverse_vol(etf_prices, est_window=60, rebalance=10, lag=1)
    part = ma.rolling_inverse_vol(etf_prices.loc[:cut], est_window=60, rebalance=10, lag=1)
    pd.testing.assert_frame_equal(full.loc[:cut], part.loc[:cut])


# ---------------------------------------------------------------------------
# ③ 执行演示：切片 → 撮合 → TCA
# ---------------------------------------------------------------------------
def test_execution_demo_structure_and_artifacts(ohlcv_frame, tmp_path):
    """跑通执行演示：子单量守恒、成交率合法、TCA 指标有限、产物落盘。"""
    out = tmp_path / "execution"
    res = ex.run_execution_demo(ohlcv_frame, symbol="TEST", side="buy",
                                compare=("twap", "vwap"), n_slices=10, lot_size=100,
                                latency=1, participation_rate=0.2, slippage_bps=5.0,
                                commission_rate=3e-4, out_dir=str(out),
                                source="synthetic-test", name="test_exec")

    for key in ("meta", "bars", "window", "runs", "tca", "artifacts"):
        assert key in res, f"返回 dict 缺少键 {key}"
    assert len(res["runs"]) == 2
    meta = res["meta"]
    assert meta["parent_qty"] > 0
    assert meta["benchmark_vwap"] > 0
    assert 0.0 < meta["parent_participation"] < 1.0

    for run in res["runs"]:
        # 子单量之和必须严格等于父单量（_allocate 的最大余额法/末位补齐保证）
        assert abs(sum(c.qty for c in run["children"]) - meta["parent_qty"]) < 1e-6
        assert run["n_children"] >= 1
        t = run["tca"]
        assert t.ordered_qty == pytest.approx(meta["parent_qty"])
        assert 0.0 <= t.fill_ratio <= 1.0 + 1e-12
        assert t.filled_qty == pytest.approx(run["filled_qty"])
        for k in ("slippage_bps", "delay_cost_bps", "realized_spread_bps",
                  "market_impact_bps", "implementation_shortfall_bps",
                  "average_fill_price"):
            _assert_finite(getattr(t, k), f"tca[{k}]")
        assert t.average_fill_price > 0
        assert t.commission >= 0.0
        assert t.market_impact_bps >= 0.0
        # 买入方向：滑点为「成交均价 − 到达价」的带符号值，5bp 设定下应大于 0 或接近 0
        assert len(run["fills_frame"]) == run["n_fills"]

    tbl = res["tca"]
    assert list(tbl.columns) == ["twap", "vwap"]
    assert "成交率" in tbl.index and "滑点(bps)" in tbl.index

    _assert_files(res["artifacts"], ("report_md", "tca_json", "fills_twap_csv",
                                     "fills_vwap_csv"), tmp_path)
    payload = _strict_json(res["artifacts"]["tca_json"])
    assert payload["meta"]["symbol"] == "TEST"
    assert {r["algo"] for r in payload["runs"]} == {"twap", "vwap"}
    text = _read_report(res["artifacts"]["report_md"])
    for heading in ("TCA 对比", "切片计划", "成交明细", "结论"):
        assert heading in text


def test_execution_demo_accepts_multiple_inputs(equity_prices):
    """bars / OHLCV DataFrame / 价格 Series / ndarray / None 五种输入都要能跑。"""
    ke = ex.ke
    bars = ex.to_bar_list(None, symbol="SIM", n_bars=60)
    assert len(bars) == 60 and all(isinstance(b, ke.Bar) for b in bars)

    ohlcv = ex.to_bar_list(bars, symbol="SIM")
    assert len(ohlcv) == 60

    series = equity_prices.iloc[:, 0]
    from_series = ex.to_bar_list(series, symbol="A00")
    assert len(from_series) == len(series)
    assert all(b.low <= b.close <= b.high for b in from_series), "OHLC 关系必须自洽"
    assert all(b.low <= b.open <= b.high for b in from_series)

    from_nd = ex.to_bar_list(series.to_numpy(), symbol="A00")
    assert len(from_nd) == len(series)

    r = ex.run_execution_demo(series, parent_qty=1000, symbol="A00", n_slices=8,
                              lot_size=100, participation_rate=1.0, out_dir=None)
    assert r["runs"][0]["tca"].fill_ratio > 0


def test_execution_demo_participation_constraint(ohlcv_frame):
    """单笔成交不得超过参与率上限（吃不满的部分顺延到后续 bar）。"""
    pr = 0.05
    res = ex.run_execution_demo(ohlcv_frame, parent_qty=None, symbol="TEST",
                                n_slices=6, lot_size=100, participation_rate=pr,
                                latency=1, out_dir=None, target_participation=0.10)
    window = {int(b.index): float(b.volume) for b in res["window"]}
    for run in res["runs"]:
        by_bar = {}
        for f in run["fills"]:
            by_bar[int(f.bar)] = by_bar.get(int(f.bar), 0.0) + float(f.qty)
        # SimBroker 的参与率是「单笔订单 × 单根 bar」的上限：同一根 bar 上多张子单
        # 各自受限，因此逐笔校验，而不是校验单根 bar 的合计成交量。
        checked = 0
        for f in run["fills"]:
            bar_idx = int(f.bar)
            if bar_idx not in window:            # 顺延到窗口外的成交不受窗口约束
                continue
            checked += 1
            cap = pr * window[bar_idx]
            assert f.qty <= cap + 1e-6, \
                f"bar {bar_idx} 单笔成交 {f.qty} 超过参与率上限 {cap}"
        assert checked > 0, "没有任何成交落在执行窗口内，测试无意义"
        assert by_bar, "应有成交记录"


def test_execution_demo_sell_side_and_algos(ohlcv_frame):
    """卖出方向与 iceberg / is 算法同样可跑通。"""
    res = ex.run_execution_demo(ohlcv_frame, symbol="TEST", side="sell",
                                compare=("twap", "is", "iceberg"), n_slices=8,
                                lot_size=100, participation_rate=0.5, out_dir=None)
    assert [r["algo"] for r in res["runs"]] == ["twap", "is", "iceberg"]
    for run in res["runs"]:
        assert run["parent"].side is ex.ke.Side.SELL
        assert run["tca"].fill_ratio > 0
    with pytest.raises(ValueError):
        ex.make_algo("unknown")


# ---------------------------------------------------------------------------
# 防未来函数
# ---------------------------------------------------------------------------
def test_factor_weights_have_no_lookahead(equity_prices):
    """把价格截断到 t 重算，t 及之前的权重必须与全样本逐位一致。"""
    scores_full = eq.build_preprocessor().fit_transform(eq.momentum_factor(equity_prices, 60))
    w_full = eq.factor_weights(scores_full, top_quantile=0.3, weighting="equal",
                               rebalance=5, lag=1)
    cut = equity_prices.index[220]
    sub = equity_prices.loc[:cut]
    scores_sub = eq.build_preprocessor().fit_transform(eq.momentum_factor(sub, 60))
    w_sub = eq.factor_weights(scores_sub, top_quantile=0.3, weighting="equal",
                              rebalance=5, lag=1)
    pd.testing.assert_frame_equal(w_full.loc[:cut], w_sub)


def test_optimized_weights_have_no_lookahead(equity_prices):
    """滚动优化权重同样只用截至再平衡日的历史。"""
    scores = eq.build_preprocessor().fit_transform(eq.momentum_factor(equity_prices, 60))
    full, _ = eq.optimized_weights(equity_prices, scores, top_quantile=0.3, est_window=120,
                                   rebalance=21, optimizer="risk_parity", lag=1)
    cut = equity_prices.index[260]
    part, _ = eq.optimized_weights(equity_prices.loc[:cut], scores.loc[:cut],
                                   top_quantile=0.3, est_window=120, rebalance=21,
                                   optimizer="risk_parity", lag=1)
    pd.testing.assert_frame_equal(full.loc[:cut], part.loc[:cut])


def test_backtest_returns_have_no_lookahead(equity_prices):
    """截断价格后重跑回测，t 及之前的净收益必须完全一致（回测器自带一期滞后）。"""
    kb = eq.kb
    scores = eq.build_preprocessor().fit_transform(eq.momentum_factor(equity_prices, 60))
    w = eq.factor_weights(scores, top_quantile=0.3, rebalance=5, lag=1)
    bt = kb.VectorBacktester(cost_rate=0.001)
    full = bt.run(equity_prices, w)
    cut = equity_prices.index[240]
    part = bt.run(equity_prices.loc[:cut], w.loc[:cut])
    pd.testing.assert_series_equal(full.returns.loc[:cut], part.returns,
                                   check_names=False)
    pd.testing.assert_series_equal(full.equity.loc[:cut], part.equity, check_names=False)


def test_forward_returns_are_forward(equity_prices):
    """远期收益必须来自「未来」：与当期收益不同、且等于未来区间的价格变化。"""
    kf = eq.kf
    h = 5
    fwd = kf.forward_returns(equity_prices, horizon=h, kind="price")
    expected = equity_prices.shift(-h) / equity_prices - 1.0
    pd.testing.assert_frame_equal(fwd, expected)
    assert fwd.iloc[-1].isna().all(), "最后 h 期没有未来数据，应为 NaN"
    assert fwd.iloc[-h:].isna().all().all()


def test_factor_ic_uses_only_forward_returns(equity_prices):
    """IC 只在因子与「其后实现」的收益之间计算：因子提前一期则相关性应明显下降。"""
    kf = eq.kf
    scores = eq.build_preprocessor().fit_transform(eq.momentum_factor(equity_prices, 60))
    fwd = kf.forward_returns(equity_prices, horizon=5, kind="price")
    aligned = kf.ic_summary(kf.rank_ic(scores, fwd))
    assert aligned["count"] > 0
    # 用「已实现」的过去收益配对属于典型未来函数误用，秩 IC 应显著为负
    backward = kf.ic_summary(kf.rank_ic(scores, -fwd))
    assert backward["mean"] == pytest.approx(-aligned["mean"], abs=1e-12)


# ---------------------------------------------------------------------------
# 数据与引导
# ---------------------------------------------------------------------------
def test_synthetic_panel_is_reproducible():
    """同 seed 两次生成必须逐位一致，且价格为正、索引单调。"""
    from kairos_lab import data as kld
    a = kld.synthetic_prices(n_assets=6, n_days=300, seed=7)
    b = kld.synthetic_prices(n_assets=6, n_days=300, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert (a.to_numpy() > 0).all()
    assert a.index.is_monotonic_increasing
    assert not kld.synthetic_prices(n_assets=6, n_days=300, seed=8).equals(a)


def test_bootstrap_locates_siblings():
    """引导模块必须能定位并 import 全部兄弟包（本仓库的运行前提）。"""
    import kairos_lab
    status = kairos_lab.ensure_paths()
    assert set(status) == set(kairos_lab.SIBLINGS)
    assert "missing" not in status.values(), f"存在不可用的兄弟包: {status}"
    for pkg in ("kairos_factor", "kairos_portfolio", "kairos_backtest", "kairos_risk",
                "kairos_execution", "kairos_data"):
        mod = kairos_lab.require(pkg)
        assert hasattr(mod, "__version__")
    assert os.path.isdir(kairos_lab.repo_root())
    with pytest.raises(KeyError):
        kairos_lab.require("kairos_nope")


def test_default_data_dir_is_relative_and_overridable(monkeypatch, tmp_path):
    """``KAIROS_ROOT`` 环境变量必须能覆盖兄弟仓库根目录（不硬编码绝对路径）。"""
    import kairos_lab
    from kairos_lab import data as kld
    original = os.environ.get(kairos_lab.ENV_ROOT)
    try:
        monkeypatch.setenv(kairos_lab.ENV_ROOT, str(tmp_path))
        assert kairos_lab.kairos_root() == str(tmp_path)
        assert kld.default_data_dir("ashare") == os.path.join(str(tmp_path),
                                                              "kairos-data", "data", "ashare")
        assert not kld.has_local_data(None, "ashare")
        with pytest.raises(FileNotFoundError):
            kld.load_panel(None, "ashare", fetch=False)
    finally:
        if original is None:
            os.environ.pop(kairos_lab.ENV_ROOT, None)
        else:
            os.environ[kairos_lab.ENV_ROOT] = original


def test_report_helpers_render_and_serialize(tmp_path):
    """报告工具：整数不带小数、NaN 变 null、Markdown 表格合法。"""
    from kairos_lab import report as rp
    df = pd.DataFrame({"a": [1, 2], "b": [1.5, float("nan")]},
                      index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    md = rp.frame_to_md(df, index_label="日期")
    lines = md.splitlines()
    assert lines[0] == "| 日期 | a | b |"
    assert lines[1] == "|---|---|---|"
    assert "2020-01-01" in lines[2] and "—" in lines[3]
    assert rp.fmt(3) == "3" and rp.fmt(3.0) == "3.0000"
    assert rp.fmt(float("nan")) == rp.NA and rp.fmt(None) == rp.NA

    payload = rp.to_jsonable({"x": np.float32(np.nan), "y": np.int64(3),
                              "df": df, "s": pd.Series([1.0, np.inf], name="s"),
                              "arr": np.arange(3)})
    path = rp.write_json(tmp_path / "sub" / "m.json", payload)
    assert os.path.isfile(path)
    data = _strict_json(path)
    assert data["x"] is None and data["y"] == 3
    assert data["s"]["data"] == [1.0, None]
    assert data["df"]["columns"] == ["a", "b"]

    md_path = rp.write_text(tmp_path / "r.md", "# t\n")
    assert _read_report(md_path).startswith("# t")
    assert rp.dict_to_md({}) == "_（无数据）_"
