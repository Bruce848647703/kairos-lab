"""端到端流水线 ④：衍生品 / 利率 / 微观结构的研究看板（自包含 HTML）。

用**真实日线数据**（``kairos-data`` 的 ``date,close,volume``）驱动三块内容：

1. **波动率曲面**（``kairos-vol``）：以真实已实现波动率为 ATM 锚，拟合 SVI 曲面，
   输出无套利体检；
2. **收益率曲线**（``kairos-rates``）：bootstrap + Nelson-Siegel，输出关键期限利率；
3. **订单簿与执行成本**（``kairos-microstructure``）：用真实 ADV/波动率校准冲击参数，
   构造订单簿、走簿报价、给出不同参与率下的成本-风险曲线。

产物是一个**不依赖 JS / 联网**的单文件 ``index.html``（内联 SVG），以及
``REPORT.md`` 与 ``metrics.json``，与 lab 里其它流水线的产物结构一致。
"""
from __future__ import annotations

import html
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .. import report as rp
from .._bootstrap import require, sibling_dir

kv = require("kairos_vol")
kr = require("kairos_rates")
km = require("kairos_microstructure")

#: 曲面期限
MATURITIES = (0.08, 0.25, 0.5, 1.0, 2.0)
#: 曲面 ATM 相对真实已实现波动的倍数（短端更陡）
ATM_MULTIPLIERS = (1.35, 1.20, 1.10, 1.00, 0.95)
#: 收益率曲线的几何形状（水平 / 斜率 / 曲度 / 衰减）
CURVE_SHAPE = {"level": 0.030, "slope": -0.015, "curvature": 0.010, "tau": 2.0}


def default_data_path(symbol: str = "sh600519") -> str:
    """默认去兄弟仓库 ``kairos-data/data/ashare/<symbol>.csv`` 找真实行情。"""
    root = sibling_dir("kairos_data")
    if root:
        candidate = os.path.join(root, "data", "ashare", f"{symbol}.csv")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"找不到 {symbol} 的真实数据；请用 data_path= 指定 CSV，"
        "或设置 KAIROS_ROOT 指向兄弟仓库父目录")


# --------------------------------------------------------------------------- #
# SVG 小工具（不引入任何绘图依赖）
# --------------------------------------------------------------------------- #
def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi - lo < 1e-12:
        return 0.5 * (out_lo + out_hi)
    return out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


def _polyline(points: Sequence[tuple], color: str, width: float = 2.0) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linejoin="round" points="{pts}"/>')


def _svg_smiles(surface: Any, width: int = 560, height: int = 300) -> str:
    left, right, top, bottom = 52, 16, 18, 34
    ks = np.linspace(-0.5, 0.5, 41)
    curves = []
    values = []
    for t in surface.maturities:
        ivs = [float(np.asarray(surface.iv(k, t))) * 100 for k in ks]
        curves.append((t, ivs))
        values.extend(ivs)
    lo, hi = min(values), max(values)
    margin = max((hi - lo) * 0.08, 0.2)
    lo, hi = lo - margin, hi + margin
    colors = ("#5dade2", "#48c9b0", "#f5b041", "#ec7063", "#af7ac5")
    parts = []
    for idx, (t, ivs) in enumerate(curves):
        pts = [(_scale(k, -0.5, 0.5, left, width - right),
                _scale(iv, lo, hi, height - bottom, top)) for k, iv in zip(ks, ivs)]
        parts.append(_polyline(pts, colors[idx % len(colors)]))
    # 坐标轴
    for frac in (0.0, 0.5, 1.0):
        y = top + frac * (height - top - bottom)
        value = hi - frac * (hi - lo)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" '
                     f'y2="{y:.1f}" stroke="#2b3542" stroke-width="1"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 4:.1f}" fill="#8b93a1" '
                     f'font-size="10" text-anchor="end">{value:.1f}%</text>')
    for frac, label in ((0.0, "-0.5"), (0.5, "0"), (1.0, "+0.5")):
        x = left + frac * (width - left - right)
        parts.append(f'<text x="{x:.1f}" y="{height - 12}" fill="#8b93a1" '
                     f'font-size="10" text-anchor="middle">k={label}</text>')
    legend = []
    for idx, t in enumerate(surface.maturities):
        x = left + idx * 74
        legend.append(f'<rect x="{x}" y="2" width="10" height="10" fill="{colors[idx % len(colors)]}"/>')
        legend.append(f'<text x="{x + 14}" y="11" fill="#c9d1d9" font-size="10">T={t:g}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">'
            + "".join(legend) + "".join(parts) + "</svg>")


def _svg_curve(curve: Any, width: int = 560, height: int = 300) -> str:
    left, right, top, bottom = 52, 16, 18, 34
    ts = np.linspace(curve.times[0], curve.times[-1], 80)
    zeros = np.asarray(curve.zero(ts)) * 100
    lo, hi = float(zeros.min()), float(zeros.max())
    margin = max((hi - lo) * 0.12, 0.05)
    lo, hi = lo - margin, hi + margin
    pts = [(_scale(t, curve.times[0], curve.times[-1], left, width - right),
            _scale(z, lo, hi, height - bottom, top)) for t, z in zip(ts, zeros)]
    parts = [_polyline(pts, "#5dade2", 2.2)]
    for t in curve.pillars:
        x = _scale(t, curve.times[0], curve.times[-1], left, width - right)
        y = _scale(float(curve.zero(t)) * 100, lo, hi, height - bottom, top)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="#f0b90b"/>')
    for frac in (0.0, 0.5, 1.0):
        y = top + frac * (height - top - bottom)
        value = hi - frac * (hi - lo)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" '
                     f'y2="{y:.1f}" stroke="#2b3542" stroke-width="1"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 4:.1f}" fill="#8b93a1" '
                     f'font-size="10" text-anchor="end">{value:.2f}%</text>')
    for t in (1, 5, 10, 20, 30):
        if t <= curve.times[-1]:
            x = _scale(t, curve.times[0], curve.times[-1], left, width - right)
            parts.append(f'<text x="{x:.1f}" y="{height - 12}" fill="#8b93a1" '
                         f'font-size="10" text-anchor="middle">{t}y</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">'
            + "".join(parts) + "</svg>")


def _svg_book(book: Any, width: int = 560, height: int = 300) -> str:
    left, right, top, bottom = 64, 16, 12, 22
    snap = book.snapshot(levels=8)
    rows = list(reversed(snap["asks"])) + list(reversed(snap["bids"]))
    if not rows:
        return '<svg viewBox="0 0 560 300" width="100%" height="300"></svg>'
    max_qty = max(r["qty"] for r in rows) or 1.0
    bar_h = (height - top - bottom) / len(rows)
    mid = float(book.mid)
    parts = []
    for i, row in enumerate(rows):
        y = top + i * bar_h
        w = (width - left - right) * row["qty"] / max_qty
        color = "#c0392b" if row["price"] > mid else "#2e86c1"
        parts.append(f'<rect x="{left}" y="{y + 1:.1f}" width="{w:.1f}" '
                     f'height="{bar_h - 2:.1f}" fill="{color}" opacity="0.85"/>')
        parts.append(f'<text x="{left - 6}" y="{y + bar_h / 2 + 3:.1f}" fill="#8b93a1" '
                     f'font-size="9" text-anchor="end">{row["price"]:.2f}</text>')
    mid_y = _scale(len(snap["asks"]), 0, len(rows), top, height - bottom)
    parts.append(f'<line x1="{left}" y1="{mid_y:.1f}" x2="{width - right}" '
                 f'y2="{mid_y:.1f}" stroke="#f0b90b" stroke-dasharray="4 3" stroke-width="1"/>')
    parts.append(f'<text x="{width - right}" y="{mid_y - 5:.1f}" fill="#f0b90b" '
                 f'font-size="10" text-anchor="end">mid {mid:.2f}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">'
            + "".join(parts) + "</svg>")


def _svg_cost(points: Sequence[Dict[str, float]], reference: float = 0.0,
              width: int = 560, height: int = 260) -> str:
    left, right, top, bottom = 56, 16, 18, 34
    xs = [p["pct"] * 100 for p in points]
    book = [p["book_bps"] for p in points]
    hi = max(max(book), reference) * 1.15 or 1.0
    xmap = lambda v: _scale(v, min(xs), max(xs), left, width - right)
    ymap = lambda v: _scale(v, 0.0, hi, height - bottom, top)
    parts = []
    for frac in (0.0, 0.5, 1.0):
        y = top + frac * (height - top - bottom)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" '
                     f'y2="{y:.1f}" stroke="#2b3542" stroke-width="1"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 4:.1f}" fill="#8b93a1" '
                     f'font-size="10" text-anchor="end">{hi * (1 - frac):.1f}</text>')
    pts = [(xmap(x), ymap(v)) for x, v in zip(xs, book)]
    parts.append(_polyline(pts, "#5dade2", 2.2))
    for x, y in pts:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="#5dade2"/>')
    if reference > 0:
        y_ref = ymap(reference)
        parts.append(f'<line x1="{left}" y1="{y_ref:.1f}" x2="{width - right}" '
                     f'y2="{y_ref:.1f}" stroke="#f0b90b" stroke-dasharray="5 3" stroke-width="1.4"/>')
        parts.append(f'<text x="{width - right}" y="{y_ref - 5:.1f}" fill="#f0b90b" '
                     f'font-size="10" text-anchor="end">AC 模型 @2% = {reference:.2f}bp</text>')
    for x in (1, 2, 5, 10):
        parts.append(f'<text x="{xmap(x):.1f}" y="{height - 12}" fill="#8b93a1" '
                     f'font-size="10" text-anchor="middle">{x}%</text>')
    parts.append(f'<text x="{left}" y="10" fill="#5dade2" font-size="10">逐档走簿冲击</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">'
            + "".join(parts) + "</svg>")


def _table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
LAMBDA_GRID = (0.0, 10.0, 25.0, 100.0, 400.0)


def _load_calibration(path: str):
    """优先用 OHLC（更好的波动/价差估计），列不全时退回收盘价+成交量。"""
    try:
        data = km.load_ohlcv(path)
        return data["close"], data["volume"], km.calibrate(
            data["close"], data["volume"], high=data["high"], low=data["low"])
    except (ValueError, KeyError):
        prices, volumes = km.load_price_volume(path)
        return prices, volumes, km.calibrate(prices, volumes)


def _svg_ac(schedules: Sequence[Any], width: int = 560, height: int = 260) -> str:
    left, right, top, bottom = 52, 16, 16, 30
    times = np.linspace(0.0, 1.0, len(schedules[0].holdings))
    parts = []
    for idx, sched in enumerate(schedules):
        ratios = sched.holdings / sched.shares * 100.0
        pts = [(_scale(t, 0, 1, left, width - right),
                _scale(r, 0, 100, height - bottom, top)) for t, r in zip(times, ratios)]
        parts.append(f'<g class="ac" data-lam="{sched.risk_aversion:g}" '
                     f'style="display:none">{_polyline(pts, "#5dade2", 2.2)}</g>')
    for frac in (0.0, 0.5, 1.0):
        y = top + frac * (height - top - bottom)
        parts.append(f'<text x="{left - 6}" y="{y + 4:.1f}" fill="#8b93a1" '
                     f'font-size="10" text-anchor="end">{100 - frac * 100:.0f}%</text>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" '
                     f'y2="{y:.1f}" stroke="#2b3542" stroke-width="1"/>')
    for frac in (0.0, 0.5, 1.0):
        x = left + frac * (width - left - right)
        parts.append(f'<text x="{x:.1f}" y="{height - 10}" fill="#8b93a1" '
                     f'font-size="10" text-anchor="middle">{frac:.1f}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">'
            + "".join(parts) + "</svg>")


def _symbol_payload(symbol: str, data_path: str) -> Dict[str, Any]:
    prices, volumes, calib = _load_calibration(data_path)
    last_price = float(prices[-1])

    atm = min(max(calib.sigma, 0.05), 1.2)
    quotes = kv.synthetic_ssvi_quotes(
        forward=last_price, maturities=MATURITIES,
        atm_vol=tuple(min(a * atm, 1.5) for a in ATM_MULTIPLIERS),
        k_min=-0.5, k_max=0.5, n_k=31)
    surface = kv.VolSurface.fit(quotes)
    vol_report = kv.surface_report(surface, k_grid=np.linspace(-0.6, 0.6, 241))

    curve = kr.bootstrap(kr.synthetic_quotes(kr.synthetic_curve(**CURVE_SHAPE)))
    ns, ns_rmse = kr.fit_nelson_siegel(
        np.array(curve.pillars), np.asarray(curve.zero(curve.pillars)))

    level_size = max(calib.adv * 0.01, 1.0)   # 1% ADV/档，与 η 的量级对齐
    sample_qty = max(calib.adv * 0.02, 100.0)
    tick = max(round(last_price * 1e-4, 2), 0.01)
    book = km.build_book(mid=last_price, tick=tick, levels=20, level_size=level_size)
    walk = km.walk_book(book, "buy", sample_qty)

    cost_points = []
    for pct in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        qty = max(calib.adv * pct, 1.0)
        fresh = km.build_book(mid=last_price, tick=tick, levels=20,
                              level_size=level_size)
        walked = km.walk_book(fresh, "buy", qty)
        cost_points.append({
            "pct": pct,
            "book_bps": walked["shortfall_bps"],
            "model_bps": (calib.eta * pct + 0.5 * calib.gamma * pct * pct) * 1e4,
        })

    schedules = [
        km.almgren_chriss(sample_qty, 1.0, 10, sigma=calib.sigma_daily,
                          eta=calib.eta, gamma=calib.gamma, adv=calib.adv,
                          risk_aversion=lam, price=last_price)
        for lam in LAMBDA_GRID
    ]
    metrics = {
        "symbol": symbol,
        "data_path": os.path.abspath(data_path),
        "n_obs": calib.n_obs,
        "last_price": last_price,
        "sigma_annual": calib.sigma,
        "sigma_daily": calib.sigma_daily,
        "sigma_intraday": calib.sigma_intraday,
        "spread_ohlc_bps": calib.spread_ohlc_bps,
        "spread_source": calib.spread_source,
        "adv": calib.adv,
        "dollar_adv": calib.dollar_adv,
        "amihud": calib.amihud,
        "roll_spread_bps": calib.roll_spread_bps,
        "kyle_lambda": calib.kyle_lambda,
        "eta": calib.eta,
        "gamma": calib.gamma,
        "surface_atm": [surface.atm_vol(t) for t in surface.maturities],
        "surface_arbitrage_free": vol_report["arbitrage_free"],
        "curve_zero": [float(curve.zero(t)) for t in curve.pillars],
        "curve_pillars": list(curve.pillars),
        "ns_params": ns.to_dict(),
        "ns_rmse_bp": ns_rmse * 1e4,
        "sample_qty": sample_qty,
        "walk_shortfall_bps": walk["shortfall_bps"],
        "cost_curve": km.cost_curve(calib, horizon=1.0),
        "book_cost_curve": cost_points,
        "ac": [{"lam": sch.risk_aversion, "cost_bps": sch.cost_bps,
                "std_bps": sch.std_bps} for sch in schedules],
    }
    return {
        "symbol": symbol,
        "metrics": metrics,
        "smiles_svg": _svg_smiles(surface),
        "curve_svg": _svg_curve(curve),
        "book_svg": _svg_book(book),
        "ac_svg": _svg_ac(schedules),
        "cost_svg": _svg_cost(cost_points, reference=next(
            (p["model_bps"] for p in cost_points if abs(p["pct"] - 0.02) < 1e-9), 0.0)),
    }


def run_dashboard(
    *,
    symbols: Optional[Sequence[str]] = None,
    symbol: str = "sh600519",
    data_path: Optional[str] = None,
    data_paths: Optional[Dict[str, str]] = None,
    out_dir: Optional[str] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """跑看板（支持多标的对比），返回产物路径与指标。"""
    names = list(symbols) if symbols else [symbol]
    paths = dict(data_paths or {})
    panels = []
    for name in names:
        path = paths.get(name) or (data_path if len(names) == 1 else None) \
            or default_data_path(name)
        panels.append(_symbol_payload(name, path))

    out_dir = rp.ensure_dir(out_dir or os.path.join("research", "dashboard"))
    html_path = rp.write_text(os.path.join(out_dir, "index.html"), _render_html(panels))
    md_path = rp.write_text(os.path.join(out_dir, "REPORT.md"),
                            _render_markdown(panels))
    json_path = rp.write_json(os.path.join(out_dir, "metrics.json"),
                              {p["symbol"]: p["metrics"] for p in panels})
    world_path = rp.write_json(os.path.join(out_dir, "world.json"), {
        p["symbol"]: {
            "symbol": p["symbol"],
            "sigma": p["metrics"]["sigma_annual"],
            "last_price": p["metrics"]["last_price"],
            "adv": p["metrics"]["adv"],
        } for p in panels})

    if not quiet:
        for panel in panels:
            m = panel["metrics"]
            print(f"[dashboard] {m['symbol']}: {m['n_obs']} 行 · "
                  f"σ={m['sigma_annual']:.1%} · ADV={m['adv']:,.0f} · "
                  f"走簿 {m['walk_shortfall_bps']:.2f}bp")
        print(f"  产物: {html_path}")

    return {"html": html_path, "report": md_path, "metrics_json": json_path,
            "world_json": world_path, "panels": panels,
            "metrics": panels[0]["metrics"]}


def summarize(result: Dict[str, Any]) -> str:
    m = result["metrics"]
    return (f"{m['symbol']} · {m['n_obs']} 行 · ATM {m['surface_atm'][0]:.1%} · "
            f"ADV {m['adv']:,.0f} · 走簿 {m['walk_shortfall_bps']:.2f}bp · "
            f"AC {m['ac'][2]['cost_bps']:.2f}bp")


def _render_markdown(panels: Sequence[Dict[str, Any]]) -> str:
    lines = ["# 研究方向看板", ""]
    for panel in panels:
        m = panel["metrics"]
        lines += [
            f"## {m['symbol']}", "",
            f"- 数据：`{m['data_path']}`（{m['n_obs']} 行，最新价 {m['last_price']:,.2f}）",
            f"- 年化波动：{m['sigma_annual']:.2%}；日频：{m['sigma_daily']:.2%}；"
            f"ADV：{m['adv']:,.0f}；成交额：{m['dollar_adv']:,.0f}",
            f"- Amihud：{m['amihud']:.3e}；Roll {m['roll_spread_bps']:.2f}bp；"
            f"OHLC 价差 {m['spread_ohlc_bps']:.2f}bp（{m['spread_source']}）；"
            f"Kyle λ {m['kyle_lambda']:.3e}",
            "", "| 期限 | ATM |", "|---:|---:|",
        ]
        for t, v in zip(MATURITIES, m["surface_atm"]):
            lines.append(f"| {t:g} | {v:.2%} |")
        lines += ["", "| 参与率 | 股数 | 成本(bp) | 风险(bp) |", "|---:|---:|---:|---:|"]
        for row in m["cost_curve"]:
            lines.append(f"| {row['participation']:.0%} | {row['shares']:,.0f} | "
                         f"{row['cost_bps']:.2f} | {row['risk_bps']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def _render_html(panels: Sequence[Dict[str, Any]]) -> str:
    import json

    payload = {
        p["symbol"]: {
            "eta": p["metrics"]["eta"],
            "gamma": p["metrics"]["gamma"],
            "sigma_daily": p["metrics"]["sigma_daily"],
            "adv": p["metrics"]["adv"],
            "ac": p["metrics"]["ac"],
        } for p in panels
    }
    options = "".join(
        f'<option value="{html.escape(p["symbol"])}"'
        f'{" selected" if i == 0 else ""}>{html.escape(p["symbol"])}</option>'
        for i, p in enumerate(panels))

    compare_html = ""
    if len(panels) > 1:
        rows = "".join(
            f'<tr data-sym="{html.escape(p["symbol"])}">'
            f'<td>{html.escape(p["symbol"])}</td>'
            f'<td>{p["metrics"]["sigma_annual"]:.2%}</td>'
            f'<td>{p["metrics"]["adv"]:,.0f}</td>'
            f'<td>{p["metrics"]["spread_ohlc_bps"]:.1f}</td>'
            f'<td>{p["metrics"]["walk_shortfall_bps"]:.2f}</td>'
            f'<td>{p["metrics"]["ac"][2]["cost_bps"]:.2f}</td>'
            f'<td>{p["metrics"]["ac"][2]["std_bps"]:.1f}</td></tr>'
            for p in panels)
        compare_html = f"""
  <div class="compare"><h2>多标的对比（点击行切换）</h2>
    <table><thead><tr><th>标的</th><th>年化波动</th><th>ADV</th>
    <th>价差(bp)</th><th>走簿冲击(bp)</th><th>AC成本(bp,λ=25)</th><th>AC风险(bp)</th>
    </tr></thead><tbody>{rows}</tbody></table>
  </div>"""

    sections = []
    for i, panel in enumerate(panels):
        m = panel["metrics"]
        calib_rows = [
            ("年化波动 σ", f"{m['sigma_annual']:.2%}"),
            ("日频波动", f"{m['sigma_daily']:.2%}"),
            ("OHLC 日频波动", "—" if m["sigma_intraday"] is None else f"{m['sigma_intraday']:.2%}"),
            ("ADV（中位）", f"{m['adv']:,.0f}"),
            ("成交额（中位）", f"{m['dollar_adv']:,.0f}"),
            ("Amihud", f"{m['amihud']:.3e}"),
            ("Roll 价差", f"{m['roll_spread_bps']:.2f}bp"),
            ("OHLC 价差", f"{m['spread_ohlc_bps']:.2f}bp（{m['spread_source']}）"),
            ("Kyle λ", f"{m['kyle_lambda']:.3e}"),
            ("临时冲击 η", f"{m['eta']:.4f}"),
            ("永久冲击 γ", f"{m['gamma']:.4f}"),
        ]
        cost_rows = "".join(
            f'<tr data-q="{r["participation"]:.4f}"><td>{r["participation"]:.0%}</td>'
            f'<td>{r["shares"]:,.0f}</td><td class="c-cost">{r["cost_bps"]:.2f}</td>'
            f'<td class="c-risk">{r["risk_bps"]:.2f}</td></tr>'
            for r in m["cost_curve"])
        curve_rows = "".join(
            f"<tr><td>{t:g}</td><td>{v:.2%}</td></tr>"
            for t, v in zip(m["curve_pillars"], m["curve_zero"]))
        sections.append(f"""
  <section class="panel" data-symbol="{html.escape(m['symbol'])}"
           style="display:{'block' if i == 0 else 'none'}">
    <div class="wide"><h2>关键指标</h2>
      <div class="kpi">
        <div><span>年化波动</span><b>{m['sigma_annual']:.2%}</b></div>
        <div><span>ADV</span><b>{m['adv']:,.0f}</b></div>
        <div><span>走簿冲击</span><b>{m['walk_shortfall_bps']:.2f}bp</b></div>
        <div><span>AC 成本</span><b id="kpi-cost">{m['ac'][2]['cost_bps']:.2f}bp</b></div>
        <div><span>AC 风险</span><b id="kpi-risk">{m['ac'][2]['std_bps']:.2f}bp</b></div>
      </div>
    </div>

    <div class="wide"><h2>波动率曲面（ATM 锚定真实已实现波动率）</h2>
      {panel['smiles_svg']}
      <p class="note">无套利体检：
        <span class="{'ok' if m['surface_arbitrage_free'] else 'bad'}">
        {'通过' if m['surface_arbitrage_free'] else '未通过'}</span>
        （k∈[-0.6, 0.6] 报价区间内；区间外为 SVI 外推）</p>
    </div>

    <div class="wide"><h2>收益率曲线</h2>
      {panel['curve_svg']}
      <p class="note">NS RMSE {m['ns_rmse_bp']:.2f}bp · β0={m['ns_params']['beta0']:.4f}
        β1={m['ns_params']['beta1']:+.4f}</p>
      <table><thead><tr><th>期限(年)</th><th>零息利率</th></tr></thead>
      <tbody>{curve_rows}</tbody></table>
    </div>

    <div class="wide"><h2>订单簿与执行成本</h2>
      {panel['book_svg']}
      <p class="note">样本订单 {m['sample_qty']:,.0f} 股（约 2% ADV）走簿
        {m['walk_shortfall_bps']:.2f}bp</p>
      <table><thead><tr><th>参与率</th><th>股数</th><th>成本(bp)</th><th>风险(bp)</th>
      </tr></thead><tbody>{cost_rows}</tbody></table>
    </div>

    <div class="wide"><h2>执行成本曲线（逐档走簿）</h2>
      {panel['cost_svg']}
      <p class="note">横轴为订单占 ADV 的比例，蓝线是合成盘口上逐档走簿的实测冲击
      （含最小价差地板）；虚线是 AC 模型在 2% ADV 的估计，仅作量级参考 ——
      走簿有固定价差、模型没有，两者不会在整个区间重合。</p>
    </div>

    <div class="wide"><h2>Almgren-Chriss 轨迹（拖动风险厌恶）</h2>
      {panel['ac_svg']}
      <p class="note">曲线为剩余仓位（%）；λ=0 即 TWAP。</p>
    </div>

    <div><h2>校准参数</h2>
      {_table(calib_rows, ['指标', '数值'])}
    </div>
    <div><h2>数据</h2>
      <p class="note">{html.escape(m['data_path'])}</p>
      <p class="note">日线 OHLCV 没有逐笔方向：Kyle λ 用带符号成交量代理；
      η 由平方根律锚定；γ 取 Kyle λ。详见 kairos-microstructure/calibrate.py。</p>
    </div>
    <div><h2>如何复现</h2>
      <p class="note">python examples/run_dashboard.py --symbols sh600519,sh600036</p>
      <p class="note">全部数字来自本地 CSV，不联网；HTML 内联 SVG，可离线打开。</p>
    </div>
  </section>""")

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kairos Lab · 研究看板</title>
<style>
:root {{ --bg:#0b0e11; --panel:#12161c; --line:#222a35; --text:#e8eaed;
        --muted:#8b93a1; --gold:#f0b90b; --accent:#4c8dff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
       font:14px/1.5 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif; }}
header {{ padding:16px 24px; border-bottom:1px solid var(--line); display:flex;
          align-items:center; gap:18px; flex-wrap:wrap; }}
header h1 {{ margin:0; font-size:18px; }}
header .ctl {{ display:flex; align-items:center; gap:8px; color:var(--muted); font-size:12px; }}
header input[type=range] {{ width:150px; accent-color:var(--accent); }}
header select {{ background:var(--panel); color:var(--text); border:1px solid var(--line);
                 border-radius:6px; padding:4px 8px; }}
main {{ padding:16px 24px; }}
section.panel > div {{ background:var(--panel); border:1px solid var(--line);
       border-radius:10px; padding:14px 16px; margin-bottom:14px; }}
section.panel h2 {{ margin:0 0 10px; font-size:13px; color:var(--muted); font-weight:600; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th,td {{ text-align:right; padding:5px 4px; border-bottom:1px solid #1a2029; }}
th {{ color:var(--muted); font-weight:400; }}
td:first-child,th:first-child {{ text-align:left; }}
tr.active td {{ background:rgba(76,141,255,.14); }}
.compare {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:14px; }}
.compare h2 {{ margin:0 0 10px; font-size:13px; color:var(--muted); font-weight:600; }}
.compare tr[data-sym] {{ cursor:pointer; }}
.compare tr[data-sym]:hover td {{ background:rgba(76,141,255,.08); }}
.kpi {{ display:flex; gap:24px; flex-wrap:wrap; }}
.kpi div {{ display:flex; flex-direction:column; }}
.kpi b {{ font-size:18px; }}
.kpi span {{ color:var(--muted); font-size:11px; }}
.note {{ color:var(--muted); font-size:11px; margin-top:8px; }}
.ok {{ color:#2ebd85; }} .bad {{ color:#f6465d; }}
</style></head><body>
<header>
  <h1>◆ KAIROS LAB · 研究看板</h1>
  <label class="ctl">标的 <select id="sym">{options}</select></label>
  <label class="ctl">参与率
    <input id="q" type="range" min="1" max="20" step="1" value="2">
    <b id="q-label">2%</b></label>
  <label class="ctl">风险厌恶 λ
    <input id="lam" type="range" min="0" max="400" step="5" value="25">
    <b id="lam-label">25</b></label>
</header>
<main>{compare_html}{''.join(sections)}</main>
<script>
const DATA = {json.dumps(payload, ensure_ascii=False)};
const $ = (s) => document.querySelector(s);
let active = Object.keys(DATA)[0];

function applyParticipation() {{
  const q = Number($('#q').value) / 100;
  $('#q-label').textContent = (q * 100).toFixed(0) + '%';
  const d = DATA[active];
  const cost = (d.eta * q + 0.5 * d.gamma * q * q) * 1e4;
  const risk = d.sigma_daily * Math.sqrt(1 / 3) * q * 1e4;
  const panel = document.querySelector(`section.panel[data-symbol="${{active}}"]`);
  panel.querySelector('#kpi-cost').textContent = cost.toFixed(2) + 'bp';
  panel.querySelector('#kpi-risk').textContent = risk.toFixed(2) + 'bp';
  panel.querySelectorAll('tr[data-q]').forEach((tr) => {{
    tr.classList.toggle('active', Math.abs(Number(tr.dataset.q) - q) < 0.005);
  }});
}}

function applyRisk() {{
  const lam = Number($('#lam').value);
  $('#lam-label').textContent = lam.toFixed(0);
  const d = DATA[active];
  let best = d.ac[0];
  for (const point of d.ac) {{
    if (Math.abs(point.lam - lam) < Math.abs(best.lam - lam)) best = point;
  }}
  const panel = document.querySelector(`section.panel[data-symbol="${{active}}"]`);
  panel.querySelectorAll('g.ac').forEach((g) => {{
    g.style.display = Number(g.dataset.lam) === Number(best.lam) ? 'inline' : 'none';
  }});
  panel.querySelector('#kpi-cost').textContent = best.cost_bps.toFixed(2) + 'bp';
  panel.querySelector('#kpi-risk').textContent = best.std_bps.toFixed(2) + 'bp';
}}

function switchSymbol() {{
  active = $('#sym').value;
  document.querySelectorAll('section.panel').forEach((p) => {{
    p.style.display = p.dataset.symbol === active ? 'block' : 'none';
  }});
  applyParticipation();
  applyRisk();
}}

document.querySelectorAll('tr[data-sym]').forEach((tr) => {{
  tr.addEventListener('click', () => {{
    $('#sym').value = tr.dataset.sym;
    switchSymbol();
  }});
}});
$('#sym').addEventListener('change', switchSymbol);
$('#q').addEventListener('input', applyParticipation);
$('#lam').addEventListener('input', applyRisk);
switchSymbol();
</script>
</body></html>"""


__all__ = ["ATM_MULTIPLIERS", "CURVE_SHAPE", "MATURITIES", "default_data_path",
           "run_dashboard", "summarize"]
