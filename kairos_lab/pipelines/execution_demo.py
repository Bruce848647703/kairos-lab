"""端到端流水线 ③：交易执行演示（父单 → 切片 → 模拟撮合 → TCA）。

用 ``kairos_execution`` 把一张父单在**真实日线 bar** 上完整走一遍执行链路::

    父单 Order(符号, 方向, 数量)
        │  TWAP / VWAP.schedule(parent_order, market=bars)  -> 子单列表
        ▼
    SimBroker（延迟 latency 根 bar 成交、参与率上限、滑点 bps、佣金）
        │  execute_children(broker, children, bars, symbol) -> Fill 列表
        ▼
    analyze(parent, fills, bars) -> TCAReport（滑点 / 成交率 / 价差 / 冲击 / 实现差额）

同一父单会分别用 ``compare`` 里列出的算法各跑一遍（每次都用**全新的** broker 与
account，互不干扰），最后汇总成一张对比表：可以直观看到 VWAP 切片相对市场基准
VWAP 的偏离通常小于 TWAP，而 TWAP 的成交节奏更均匀。

关键口径
--------
- **决策价** ``decision_price``：窗口开始前一根 bar 的收盘价（做出交易决策时看到的价）；
- **到达价** ``arrival_price``：窗口第一根 bar 的开盘价（订单到达市场时的价）；
- **滑点**：成交均价 vs 到达价；**延迟成本**：到达价 vs 决策价；
  **已实现价差**：成交均价 vs 窗口市场 VWAP；**实现差额**：相对决策价的总成本（含佣金）。
- 成交量单位以数据源口径为准（A 股为「手」）；本演示把 ``qty`` 与 ``bar.volume``
  视为同一单位，因此参与率约束与冲击估计在量纲上自洽。
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .. import report as rp
from .._bootstrap import require

ke = require("kairos_execution")

#: 支持的执行算法
ALGOS = ("twap", "vwap", "iceberg", "is")
#: 支持的方向
SIDES = ("buy", "sell")

BarsLike = Union[Sequence[Any], pd.DataFrame, pd.Series, np.ndarray, None]


# ---------------------------------------------------------------------------
# 行情归一化
# ---------------------------------------------------------------------------
def _looks_like_ohlcv(df: pd.DataFrame) -> bool:
    """判断 DataFrame 是否含 OHLCV 四价列。"""
    return {"open", "high", "low", "close"}.issubset({str(c).lower() for c in df.columns})


def to_bar_list(bars_or_prices: BarsLike, symbol: str = "SIM", n_bars: int = 120,
                seed: int = 7) -> List[Any]:
    """把多形态输入归一成 ``kairos_execution.Bar`` 列表（bar.index = 列表下标）。

    支持：

    - ``None``                → 用 ``ke.make_ohlcv`` 生成合成 OHLCV（离线、固定 seed）；
    - ``List[Bar]``           → 直接使用（必要时重排 index）；
    - OHLCV ``DataFrame``     → ``ke.to_bars(df, symbol)``；
    - 价格 ``Series``/单列或
      多列价格 ``DataFrame``  → 先构造成 OHLCV 再转 bar；
    - 一维 ``ndarray``        → 同上（视为收盘价序列）。
    """
    if bars_or_prices is None:
        df = ke.make_ohlcv(n_bars=n_bars, seed=seed, symbol=symbol)
        return ke.to_bars(df, symbol)

    if isinstance(bars_or_prices, (list, tuple)) and len(bars_or_prices) \
            and isinstance(bars_or_prices[0], ke.Bar):
        bars = list(bars_or_prices)
        # bar.index 必须是「在完整 bar 序列中的位置」，否则子单的 created_bar 会错位
        need_fix = any(int(getattr(b, "index", -1)) != i for i, b in enumerate(bars))
        if need_fix:
            bars = [dataclasses.replace(b, index=i, symbol=b.symbol or symbol)
                    for i, b in enumerate(bars)]
        return bars

    if isinstance(bars_or_prices, pd.DataFrame):
        df = bars_or_prices
        if _looks_like_ohlcv(df):
            low = {str(c).lower(): c for c in df.columns}
            view = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce")
                                 for k, v in low.items()
                                 if k in ("open", "high", "low", "close", "volume")},
                                index=df.index)
            if "volume" not in view.columns:
                view["volume"] = 1.0
            return ke.to_bars(view.dropna(), symbol)
        prices = df.iloc[:, 0]
    elif isinstance(bars_or_prices, pd.Series):
        prices = bars_or_prices
    else:
        arr = np.asarray(bars_or_prices, dtype="float64").ravel()
        prices = pd.Series(arr, name=symbol)

    from ..data import synthetic_ohlcv          # 局部导入，避免包级循环依赖
    ohlcv = synthetic_ohlcv(prices.to_frame(symbol), seed=seed, symbol=symbol)
    return ke.to_bars(ohlcv, symbol)


def pick_window(bars: Sequence[Any], n_slices: int, latency: int
                ) -> Tuple[int, int, List[Any]]:
    """在执行窗口上切片：返回 ``(起点, 终点, 窗口 bar 列表)``（终点为开区间）。

    子单在 ``created_bar`` 提交、第 ``created_bar + latency`` 根 bar 才成交，
    因此窗口末尾必须留出 ``latency`` 根 bar，否则最后几张子单永远无法成交。
    """
    n = len(bars)
    if n < 3:
        raise ValueError(f"bar 数量过少（{n}），无法演示切片执行")
    k = max(1, min(int(n_slices), max(1, n - int(latency) - 2)))
    end = n - int(latency)
    start = max(1, end - k)
    return start, end, list(bars[start:end])


def make_algo(name: str, lot_size: Optional[int] = 1,
              display_size: Optional[float] = None, decay: float = 0.7) -> Any:
    """按名称构造执行算法实例。"""
    key = str(name).lower()
    if key == "twap":
        return ke.TWAP(lot_size=lot_size)
    if key == "vwap":
        return ke.VWAP(lot_size=lot_size)
    if key == "iceberg":
        if not display_size:
            raise ValueError("iceberg 需要 display_size（每批显露量）")
        return ke.Iceberg(display_size=float(display_size), lot_size=lot_size)
    if key == "is":
        return ke.ImplementationShortfall(decay=float(decay), lot_size=lot_size)
    raise ValueError(f"未知执行算法: {name!r}，可选 {ALGOS}")


def _side(side: str) -> Any:
    key = str(side).lower()
    if key in ("buy", "b", "1"):
        return ke.Side.BUY
    if key in ("sell", "s", "-1"):
        return ke.Side.SELL
    raise ValueError(f"未知方向: {side!r}，可选 {SIDES}")


# ---------------------------------------------------------------------------
# 单次执行
# ---------------------------------------------------------------------------
def execute_once(parent: Any, bars: Sequence[Any], window: Sequence[Any],
                 symbol: str, algo_name: str, lot_size: Optional[int] = 1,
                 latency: int = 1, participation_rate: float = 0.10,
                 slippage_bps: float = 5.0, commission_rate: float = 3e-4,
                 commission_min: float = 5.0, market_ref: str = "close",
                 decision_price: Optional[float] = None,
                 arrival_price: Optional[float] = None,
                 impact_coef_bps: float = 10.0, cash: Optional[float] = None,
                 display_size: Optional[float] = None, decay: float = 0.7) -> Dict[str, Any]:
    """用指定算法把父单在窗口上执行一遍，返回成交明细与 TCA 报告。

    每次都新建 ``SimBroker`` 与 ``Account``，因此多个算法之间完全独立、可对比。
    """
    algo = make_algo(algo_name, lot_size=lot_size, display_size=display_size, decay=decay)
    children = algo.schedule(parent, list(window))
    if not children:
        raise RuntimeError(f"{algo_name} 未能切出任何子单（父单量 {parent.qty}）")

    account = ke.Account(cash=float(cash) if cash is not None else 0.0)
    broker = ke.SimBroker(account=account, latency=int(latency),
                          participation_rate=float(participation_rate),
                          slippage_bps=float(slippage_bps),
                          commission_rate=float(commission_rate),
                          commission_min=float(commission_min),
                          market_ref=market_ref)
    fills = ke.execute_children(broker, children, list(bars), symbol)
    frame = ke.fills_to_frame(fills)

    if arrival_price is None:
        arrival_price = float(window[0].open)
    if decision_price is None:
        decision_price = arrival_price
    tca = ke.analyze(parent, fills, bars=list(window), arrival_price=arrival_price,
                     decision_price=decision_price, impact_coef_bps=impact_coef_bps)

    filled = float(sum(f.qty for f in fills))
    qty_by_bar: Dict[int, float] = {}
    for f in fills:
        qty_by_bar[int(f.bar)] = qty_by_bar.get(int(f.bar), 0.0) + float(f.qty)
    planned = pd.Series([float(c.qty) for c in children],
                        index=[int(c.created_bar) for c in children])
    filled_by_bar = (pd.Series(qty_by_bar).sort_index() if qty_by_bar
                     else pd.Series(dtype="float64"))
    vol_by_bar = pd.Series({int(b.index): float(b.volume) for b in window})
    return {
        "algo": algo_name,
        "parent": parent,
        "children": children,
        "fills": fills,
        "fills_frame": frame,
        "tca": tca,
        "account": account,
        "broker": broker,
        "n_children": len(children),
        "n_fills": len(fills),
        "filled_qty": filled,
        "unfilled_qty": float(parent.qty) - filled,
        "planned_qty": planned,
        "filled_by_bar": filled_by_bar,
        "participation_by_bar": (filled_by_bar / vol_by_bar.reindex(filled_by_bar.index))
        if len(filled_by_bar) else pd.Series(dtype="float64"),
        "avg_fill_price": tca.average_fill_price,
        "notional": filled * tca.average_fill_price,
        "commission": tca.commission,
    }


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def tca_frame(runs: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """把多次执行的 TCA 汇总成一张对比表（index=指标，columns=算法）。"""
    cols = {}
    for run in runs:
        t = run["tca"]
        cols[run["algo"]] = {
            "下单量": t.ordered_qty,
            "成交量": t.filled_qty,
            "成交率": t.fill_ratio,
            "子单数": float(run["n_children"]),
            "成交笔数": float(t.num_fills),
            "决策价": t.decision_price,
            "到达价": t.arrival_price,
            "成交均价": t.average_fill_price,
            "滑点(bps)": t.slippage_bps,
            "延迟成本(bps)": t.delay_cost_bps,
            "相对市场VWAP价差(bps)": t.realized_spread_bps,
            "市场冲击估计(bps)": t.market_impact_bps,
            "实现差额(bps)": t.implementation_shortfall_bps,
            "实现差额(货币)": t.implementation_shortfall,
            "佣金": t.commission,
            "成交金额": run["notional"],
        }
    return pd.DataFrame(cols)


def build_report(meta: Dict[str, Any], runs: Sequence[Dict[str, Any]],
                 window: Sequence[Any]) -> str:
    """渲染中文 Markdown 报告正文（不含一级标题）。"""
    head = [
        f"- 生成时间：{meta['generated_at']}",
        f"- 行情来源：{meta['source']}",
        f"- 标的 / 方向 / 下单量：{meta['symbol']} / {meta['side']} / {meta['parent_qty']:,.0f}",
        f"- bar 总数 {meta['n_bars']}，执行窗口 [{meta['window_start']}, {meta['window_end']}) "
        f"共 {meta['n_window']} 根（{meta['window_start_date']} ~ {meta['window_end_date']}）",
        f"- 撮合设置：latency={meta['latency']} 根 bar，参与率上限={meta['participation_rate']:.0%}/bar，"
        f"滑点={meta['slippage_bps']:.1f}bp，佣金={meta['commission_rate']:.2%}"
        f"（最低 {meta['commission_min']:.2f}），市价参考价={meta['market_ref']}",
        f"- 窗口市场基准 VWAP={meta['benchmark_vwap']:.4f}，"
        f"窗口成交量合计={meta['window_volume']:,.0f}，"
        f"父单占窗口成交量={meta['parent_participation']:.3%}",
        "- 依赖版本：" + " / ".join(f"{k} {v}" for k, v in meta["versions"].items()),
    ]
    view = tca_frame(runs).astype(object)
    # 数量/金额类指标用千分位整数字符串展示，避免 "12000.0000" 这种噪声
    for row in ("下单量", "成交量", "子单数", "成交笔数", "成交金额", "佣金",
                "实现差额(货币)"):
        if row in view.index:
            view.loc[row] = [f"{float(v):,.0f}" if pd.notna(v) else rp.NA
                             for v in view.loc[row]]

    parts = ["\n".join(head)]
    parts += ["## 撮合与成本口径", rp.bullets([
        "**决策价**：窗口开始前一根 bar 的收盘价（下单决策时看到的价格）。",
        "**到达价**：窗口第一根 bar 的开盘价（订单到达市场时的价格）。",
        "**滑点**：成交均价相对到达价的带方向偏离（买入成交价更高 = 正 = 不利）。",
        "**相对市场 VWAP 价差**：成交均价相对窗口成交量加权均价的偏离，"
        "衡量「有没有跟上市场自然成交节奏」。",
        "**市场冲击**：平方根参与率模型 ``coef × √(成交量/窗口总量)`` 的粗估，"
        "只用于横向对比不同算法。",
        "**实现差额 (IS)**：相对决策价的总成本（含佣金），换算成 bps 后可直接比较。",
    ])]
    parts += ["## TCA 对比（kairos_execution.analyze）",
              rp.frame_to_md(view, floatfmt=".4f", index_label="指标")]

    plan_rows = []
    for run in runs:
        q = run["planned_qty"]
        plan_rows.append({
            "算法": run["algo"],
            "子单数": int(len(q)),
            "首张子单bar": int(q.index.min()) if len(q) else -1,
            "末张子单bar": int(q.index.max()) if len(q) else -1,
            "最大子单量": float(q.max()) if len(q) else float("nan"),
            "最小子单量": float(q.min()) if len(q) else float("nan"),
            "子单量标准差": float(q.std(ddof=1)) if len(q) > 1 else 0.0,
            "成交量": run["filled_qty"],
            "未成交量": run["unfilled_qty"],
        })
    parts += ["## 切片计划", rp.frame_to_md(pd.DataFrame(plan_rows).set_index("算法"),
                                          floatfmt=".2f", index_label="算法")]

    fills_parts = []
    for run in runs:
        f: pd.DataFrame = run["fills_frame"]
        if len(f) == 0:
            continue
        show = f if len(f) <= 6 else pd.concat([f.head(3), f.tail(3)])
        fills_parts.append(f"**{run['algo']}**（共 {len(f)} 笔成交）\n\n"
                           + rp.frame_to_md(show.set_index("fill_id"), floatfmt=".4f",
                                            index_label="成交号"))
    if fills_parts:
        parts += ["## 成交明细（超过 6 笔时只显示首尾各 3 笔）", "\n\n".join(fills_parts)]

    best = min(runs, key=lambda r: abs(r["tca"].realized_spread_bps))
    cheap = min(runs, key=lambda r: r["tca"].implementation_shortfall_bps)
    parts += ["## 结论", rp.bullets([
        f"相对市场基准 VWAP 偏离最小的是 **{best['algo']}**"
        f"（{best['tca'].realized_spread_bps:+.2f} bps），"
        f"实现差额最低的是 **{cheap['algo']}**"
        f"（{cheap['tca'].implementation_shortfall_bps:+.2f} bps）。",
        f"父单占窗口成交量 {meta['parent_participation']:.2%}，"
        f"参与率上限 {meta['participation_rate']:.0%}/bar 决定了单根 bar 的成交上限；"
        "若窗口内吃不满，未成交量会顺延到后续 bar，成交率随之小于 1。",
        f"成交延迟 latency={meta['latency']} 根 bar 会放大「延迟成本」：决策价与到达价"
        "之间的漂移与算法无关，但会被计入实现差额，这正是执行风险的主要来源之一。",
        "本演示为**纯离线模拟撮合**，不接任何真实券商通道；滑点/佣金/参与率均为设定值，"
        "结果只用于比较算法间的相对优劣。",
    ])]
    parts += ["## 如何复现", rp.code_block("python examples/run_execution_demo.py", "bash")]
    return "\n\n".join(p.strip() for p in parts).strip() + "\n"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_execution_demo(bars_or_prices: BarsLike = None, parent_qty: Optional[float] = None,
                       symbol: str = "SIM", side: str = "buy",
                       compare: Sequence[str] = ("twap", "vwap"), n_slices: int = 20,
                       lot_size: Optional[int] = 100, latency: int = 1,
                       participation_rate: float = 0.10, slippage_bps: float = 5.0,
                       commission_rate: float = 3e-4, commission_min: float = 5.0,
                       market_ref: str = "close", impact_coef_bps: float = 10.0,
                       target_participation: float = 0.02, cash: Optional[float] = None,
                       display_size: Optional[float] = None, decay: float = 0.7,
                       bar_labels: Optional[Sequence[Any]] = None,
                       out_dir: Optional[str] = None, source: str = "synthetic OHLCV",
                       name: str = "execution_demo") -> Dict[str, Any]:
    """跑完整的执行演示：切分父单 → 模拟撮合 → TCA 对比。

    参数
    ----
    bars_or_prices: ``Bar`` 列表 / OHLCV DataFrame / 价格 Series 或单列 DataFrame /
                    一维 ndarray；``None`` 时用 ``ke.make_ohlcv`` 生成合成行情。
    parent_qty:     父单数量；``None`` 时按 ``target_participation × 窗口成交量`` 自动确定
                    （并向下取整到 ``lot_size`` 的整数倍），保证在参与率约束下可基本成交。
    symbol:         标的代码。
    side:           ``"buy"`` / ``"sell"``。
    compare:        要对比的算法名序列（``twap`` / ``vwap`` / ``iceberg`` / ``is``）。
    n_slices:       TWAP/VWAP/IS 的切片数（受窗口长度约束）。
    lot_size:       最小交易单位；``None`` 允许碎股。
    latency:        成交延迟（根 bar）。
    participation_rate: 单根 bar 的成交量参与率上限。
    slippage_bps / commission_rate / commission_min / market_ref: 撮合与成本设定。
    impact_coef_bps: 平方根冲击模型系数。
    display_size:   ``iceberg`` 每批显露量；省略时取父单量的 1/4。
    decay:          ``is``（ImplementationShortfall）的几何衰减系数。
    bar_labels:     与 bars 等长的日期标签（用于报告展示）；省略时自动从
                    DataFrame/Series 输入的 index 推断。
    target_participation: ``parent_qty=None`` 时父单占窗口成交量的目标比例。
    cash:           账户初始现金（``enforce_cash=False``，因此不会限制成交）。
    out_dir:        非空时写出 ``REPORT.md`` / ``tca.json`` / ``fills_<algo>.csv``。
    source:         行情来源描述（写入报告）。
    name:           流水线名。

    返回 dict，键为 ``meta`` / ``bars`` / ``window`` / ``runs`` / ``tca`` / ``artifacts``。
    """
    if bar_labels is None and isinstance(bars_or_prices, (pd.DataFrame, pd.Series)):
        bar_labels = list(bars_or_prices.index)
    bars = to_bar_list(bars_or_prices, symbol=symbol)
    if len(bars) < 5:
        raise ValueError(f"至少需要 5 根 bar，收到 {len(bars)}")
    symbol = str(symbol) if symbol else str(bars[0].symbol)
    start, end, window = pick_window(bars, n_slices, latency)
    side_enum = _side(side)

    window_volume = float(sum(b.volume for b in window))
    benchmark = float(ke.benchmark_vwap(window))
    if parent_qty is None:
        raw = float(target_participation) * window_volume
        lot = int(lot_size) if lot_size else 1
        parent_qty = max(float(lot), float(int(raw / lot) * lot))
    parent_qty = float(parent_qty)
    if parent_qty <= 0:
        raise ValueError("parent_qty 必须为正")

    if display_size is None and "iceberg" in [str(a).lower() for a in compare]:
        display_size = parent_qty / 4.0
    decision_price = float(bars[start - 1].close) if start > 0 else float(window[0].open)
    arrival_price = float(window[0].open)
    px = arrival_price if arrival_price > 0 else max(benchmark, 1.0)
    cash_used = float(cash) if cash is not None else 2.0 * parent_qty * px

    runs: List[Dict[str, Any]] = []
    for algo_name in compare:
        parent = ke.Order(order_id=f"parent-{algo_name}", symbol=symbol, side=side_enum,
                          qty=parent_qty)
        runs.append(execute_once(
            parent, bars, window, symbol, algo_name, lot_size=lot_size, latency=latency,
            participation_rate=participation_rate, slippage_bps=slippage_bps,
            commission_rate=commission_rate, commission_min=commission_min,
            market_ref=market_ref, decision_price=decision_price,
            arrival_price=arrival_price, impact_coef_bps=impact_coef_bps, cash=cash_used,
            display_size=display_size, decay=decay))

    meta: Dict[str, Any] = {
        "pipeline": name,
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "symbol": symbol,
        "side": side_enum.name,
        "parent_qty": parent_qty,
        "n_bars": len(bars),
        "window_start": int(start),
        "window_end": int(end),
        "n_window": len(window),
        "window_start_date": _bar_date(bar_labels, start, len(bars)),
        "window_end_date": _bar_date(bar_labels, end - 1, len(bars)),
        "window_volume": window_volume,
        "benchmark_vwap": benchmark,
        "parent_participation": parent_qty / window_volume if window_volume > 0 else float("nan"),
        "n_slices": int(n_slices),
        "lot_size": lot_size,
        "latency": int(latency),
        "participation_rate": float(participation_rate),
        "slippage_bps": float(slippage_bps),
        "commission_rate": float(commission_rate),
        "commission_min": float(commission_min),
        "market_ref": market_ref,
        "impact_coef_bps": float(impact_coef_bps),
        "decision_price": decision_price,
        "arrival_price": arrival_price,
        "algos": [r["algo"] for r in runs],
        "versions": {m.__name__: getattr(m, "__version__", "?") for m in (ke,)},
    }

    artifacts: Dict[str, str] = {}
    if out_dir:
        d = rp.ensure_dir(out_dir)
        artifacts["report_md"] = rp.write_text(
            os.path.join(d, "REPORT.md"),
            f"# Kairos Lab · {name} 交易执行与 TCA 报告\n\n{build_report(meta, runs, window)}")
        artifacts["tca_json"] = rp.write_json(os.path.join(d, "tca.json"), {
            "meta": meta,
            "tca": tca_frame(runs),
            "runs": [{"algo": r["algo"], "n_children": r["n_children"],
                      "n_fills": r["n_fills"], "filled_qty": r["filled_qty"],
                      "unfilled_qty": r["unfilled_qty"], "notional": r["notional"],
                      "tca": r["tca"]} for r in runs],
        })
        for run in runs:
            f: pd.DataFrame = run["fills_frame"]
            artifacts[f"fills_{run['algo']}_csv"] = rp.write_csv(
                f, os.path.join(d, f"fills_{run['algo']}.csv"))

    return {"meta": meta, "bars": bars, "window": window, "runs": runs,
            "tca": tca_frame(runs), "artifacts": artifacts}


def _bar_date(labels: Optional[Sequence[Any]], i: int, n_bars: int) -> str:
    """取出第 i 根 bar 的日期标签；无标签时退化为 ``bar#i``。"""
    if labels is not None and 0 <= i < len(labels):
        try:
            return str(pd.Timestamp(labels[i]).date())
        except (TypeError, ValueError):
            return str(labels[i])
    return f"bar#{i}" if 0 <= i < n_bars else "—"


def summarize(result: Dict[str, Any]) -> str:
    """把演示结果压成一段可直接 print 的中文摘要。"""
    meta = result["meta"]
    lines = [
        f"[{meta['pipeline']}] {meta['source']} | {meta['symbol']} {meta['side']} "
        f"{meta['parent_qty']:,.0f} | 窗口 {meta['n_window']} 根 bar "
        f"({meta['window_start_date']}~{meta['window_end_date']})",
        f"  撮合: latency={meta['latency']} 参与率≤{meta['participation_rate']:.0%}/bar "
        f"滑点={meta['slippage_bps']:.1f}bp 佣金={meta['commission_rate']:.2%} "
        f"决策价={meta['decision_price']:.4f} 到达价={meta['arrival_price']:.4f} "
        f"基准VWAP={meta['benchmark_vwap']:.4f}",
        f"  父单占窗口成交量={meta['parent_participation']:.3%}",
    ]
    for run in result["runs"]:
        t = run["tca"]
        lines.append(
            f"  {run['algo']:<8} 子单{run['n_children']:>3}张 成交{run['n_fills']:>3}笔 "
            f"成交率={t.fill_ratio:.2%} 均价={t.average_fill_price:.4f} "
            f"滑点={t.slippage_bps:+.2f}bp 相对VWAP={t.realized_spread_bps:+.2f}bp "
            f"延迟={t.delay_cost_bps:+.2f}bp 冲击≈{t.market_impact_bps:.2f}bp "
            f"IS={t.implementation_shortfall_bps:+.2f}bp 佣金={t.commission:,.2f}")
    if result["artifacts"]:
        lines.append(f"  产物: {', '.join(os.path.basename(v) for v in result['artifacts'].values())}"
                     f" -> {os.path.dirname(result['artifacts']['report_md'])}")
    return "\n".join(lines)
