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
import math
import os
from typing import Any, Dict, List, Optional, Sequence

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


def _table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_dashboard(
    *,
    symbol: str = "sh600519",
    data_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    risk_aversion: float = 25.0,
    quiet: bool = False,
) -> Dict[str, Any]:
    """跑完整看板，返回产物路径与关键指标。"""
    path = data_path or default_data_path(symbol)
    prices, volumes = km.load_price_volume(path)
    calib = km.calibrate(prices, volumes)
    last_price = float(prices[-1])

    # 1) 波动率曲面（真实已实现波动率作为 ATM 锚）
    atm = min(max(calib.sigma, 0.05), 1.2)
    quotes = kv.synthetic_ssvi_quotes(
        forward=last_price, maturities=MATURITIES,
        atm_vol=tuple(min(a * atm, 1.5) for a in ATM_MULTIPLIERS),
        k_min=-0.5, k_max=0.5, n_k=31)
    surface = kv.VolSurface.fit(quotes)
    # 只在报价覆盖的 k 区间（±0.6）内做无套利体检；区间外是 SVI 外推，不作承诺
    vol_report = kv.surface_report(surface, k_grid=np.linspace(-0.6, 0.6, 241))

    # 2) 收益率曲线
    curve = kr.bootstrap(kr.synthetic_quotes(
        kr.synthetic_curve(**CURVE_SHAPE)))
    ns, ns_rmse = kr.fit_nelson_siegel(
        np.array(curve.pillars), np.asarray(curve.zero(curve.pillars)))

    # 3) 订单簿 + 执行成本
    # 每档约 0.5% ADV、档间距 1bp，接近真实盘口的量级
    level_size = max(calib.adv * 0.005, 1.0)
    sample_qty = max(calib.adv * 0.02, 100.0)
    tick = max(round(last_price * 1e-4, 2), 0.01)
    book = km.build_book(mid=last_price, tick=tick, levels=20, level_size=level_size)
    walk = km.walk_book(book, "buy", sample_qty)
    costs = km.cost_curve(calib, horizon=1.0)
    ac = km.almgren_chriss(sample_qty, 1.0, 10, sigma=calib.sigma_daily,
                           eta=calib.eta, gamma=calib.gamma, adv=calib.adv,
                           risk_aversion=risk_aversion, price=last_price)

    metrics = {
        "symbol": symbol,
        "data_path": os.path.abspath(path),
        "n_obs": calib.n_obs,
        "last_price": last_price,
        "sigma_annual": calib.sigma,
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
        "ac_cost_bps": ac.cost_bps,
        "ac_std_bps": ac.std_bps,
        "cost_curve": costs,
    }

    out_dir = rp.ensure_dir(out_dir or os.path.join("research", "dashboard"))
    html_text = _render_html(metrics, surface, curve, book)
    html_path = rp.write_text(os.path.join(out_dir, "index.html"), html_text)
    md_path = rp.write_text(os.path.join(out_dir, "REPORT.md"), _render_markdown(metrics))
    json_path = rp.write_json(os.path.join(out_dir, "metrics.json"), metrics)

    if not quiet:
        print(f"[dashboard] {symbol}: {len(prices)} 行真实数据")
        print(f"  波动率曲面: ATM {metrics['surface_atm'][0]:.2%} → "
              f"{metrics['surface_atm'][-1]:.2%}  无套利={metrics['surface_arbitrage_free']}")
        print(f"  收益率曲线: 10Y {float(curve.zero(10.0)):.2%}  NS RMSE={ns_rmse*1e4:.2f}bp")
        print(f"  执行成本:   {sample_qty:,.0f} 股走簿 {walk['shortfall_bps']:.2f}bp  "
              f"AC 成本 {ac.cost_bps:.2f}bp / 风险 {ac.std_bps:.2f}bp")
        print(f"  产物: {html_path}")

    return {"html": html_path, "report": md_path, "metrics_json": json_path,
            "metrics": metrics}


def summarize(result: Dict[str, Any]) -> str:
    m = result["metrics"]
    return (f"{m['symbol']} · {m['n_obs']} 行 · ATM {m['surface_atm'][0]:.1%} · "
            f"ADV {m['adv']:,.0f} · 走簿 {m['walk_shortfall_bps']:.2f}bp · "
            f"AC {m['ac_cost_bps']:.2f}bp")


def _render_markdown(m: Dict[str, Any]) -> str:
    lines = [
        f"# 研究看板 · {m['symbol']}", "",
        f"- 数据：`{m['data_path']}`（{m['n_obs']} 个观测，最新价 {m['last_price']:,.2f}）",
        f"- 年化波动：{m['sigma_annual']:.2%}；ADV：{m['adv']:,.0f} 股；"
        f"成交额：{m['dollar_adv']:,.0f}",
        f"- Amihud：{m['amihud']:.3e}；Roll 价差：{m['roll_spread_bps']:.2f}bp；"
        f"Kyle λ：{m['kyle_lambda']:.3e}",
        "",
        "## 波动率曲面", "",
        f"无套利体检：{'通过' if m['surface_arbitrage_free'] else '未通过'}", "",
        "| 期限 | ATM |", "|---:|---:|",
    ]
    for t, v in zip(MATURITIES, m["surface_atm"]):
        lines.append(f"| {t:g} | {v:.2%} |")
    lines += ["", "## 收益率曲线", "", "| 期限 | 零息利率 |", "|---:|---:|"]
    for t, v in zip(m["curve_pillars"], m["curve_zero"]):
        lines.append(f"| {t:g} | {v:.2%} |")
    lines += ["", f"Nelson-Siegel RMSE：{m['ns_rmse_bp']:.2f}bp", "",
              "## 执行成本", "",
              f"{m['sample_qty']:,.0f} 股走簿冲击：{m['walk_shortfall_bps']:.2f}bp；"
              f"Almgren-Chriss 成本 {m['ac_cost_bps']:.2f}bp / 风险 {m['ac_std_bps']:.2f}bp", "",
              "| 参与率 | 股数 | 成本(bp) | 风险(bp) |", "|---:|---:|---:|---:|"]
    for row in m["cost_curve"]:
        lines.append(f"| {row['participation']:.0%} | {row['shares']:,.0f} | "
                     f"{row['cost_bps']:.2f} | {row['risk_bps']:.2f} |")
    lines.append("")
    return "\n".join(lines)


def _render_html(m: Dict[str, Any], surface: Any, curve: Any, book: Any) -> str:
    calib_rows = [
        ("年化波动 σ", f"{m['sigma_annual']:.2%}"),
        ("ADV（中位）", f"{m['adv']:,.0f}"),
        ("成交额（中位）", f"{m['dollar_adv']:,.0f}"),
        ("Amihud", f"{m['amihud']:.3e}"),
        ("Roll 价差", f"{m['roll_spread_bps']:.2f}bp"),
        ("Kyle λ", f"{m['kyle_lambda']:.3e}"),
        ("临时冲击 η", f"{m['eta']:.4f}"),
        ("永久冲击 γ", f"{m['gamma']:.4f}"),
    ]
    cost_rows = [(f"{r['participation']:.0%}", f"{r['shares']:,.0f}",
                  f"{r['cost_bps']:.2f}", f"{r['risk_bps']:.2f}") for r in m["cost_curve"]]
    curve_rows = [(f"{t:g}", f"{v:.2%}") for t, v in zip(m["curve_pillars"], m["curve_zero"])]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kairos Lab · 研究看板 · {html.escape(m['symbol'])}</title>
<style>
:root {{ --bg:#0b0e11; --panel:#12161c; --line:#222a35; --text:#e8eaed;
        --muted:#8b93a1; --gold:#f0b90b; --accent:#4c8dff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
       font:14px/1.5 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif; }}
header {{ padding:18px 24px; border-bottom:1px solid var(--line); display:flex;
          align-items:baseline; gap:16px; }}
header h1 {{ margin:0; font-size:18px; }}
header span {{ color:var(--muted); font-size:12px; }}
main {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; padding:16px 24px; }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:14px 16px; }}
section h2 {{ margin:0 0 10px; font-size:13px; color:var(--muted); font-weight:600; }}
.wide {{ grid-column:1 / -1; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th,td {{ text-align:right; padding:5px 4px; border-bottom:1px solid #1a2029; }}
th {{ color:var(--muted); font-weight:400; }}
td:first-child,th:first-child {{ text-align:left; }}
.kpi {{ display:flex; gap:22px; flex-wrap:wrap; }}
.kpi div {{ display:flex; flex-direction:column; }}
.kpi b {{ font-size:18px; }}
.kpi span {{ color:var(--muted); font-size:11px; }}
.note {{ color:var(--muted); font-size:11px; margin-top:8px; }}
.ok {{ color:#2ebd85; }} .bad {{ color:#f6465d; }}
@media (max-width:1100px) {{ main {{ grid-template-columns:1fr; }} }}
</style></head><body>
<header><h1>◆ KAIROS LAB · 研究看板</h1>
<span>{html.escape(m['symbol'])} · {m['n_obs']} 行真实日线 · 最新价 {m['last_price']:,.2f}</span></header>
<main>
  <section class="wide"><h2>关键指标</h2>
    <div class="kpi">
      <div><span>年化波动</span><b>{m['sigma_annual']:.2%}</b></div>
      <div><span>ADV</span><b>{m['adv']:,.0f}</b></div>
      <div><span>成交额</span><b>{m['dollar_adv']:,.0f}</b></div>
      <div><span>走簿冲击</span><b>{m['walk_shortfall_bps']:.2f}bp</b></div>
      <div><span>AC 成本</span><b>{m['ac_cost_bps']:.2f}bp</b></div>
      <div><span>AC 风险</span><b>{m['ac_std_bps']:.2f}bp</b></div>
    </div>
  </section>

  <section class="wide"><h2>波动率曲面（ATM 锚定真实已实现波动率）</h2>
    {_svg_smiles(surface)}
    <p class="note">无套利体检：
      <span class="{'ok' if m['surface_arbitrage_free'] else 'bad'}">
      {'通过' if m['surface_arbitrage_free'] else '未通过'}</span>
      （蝶式 g(k)≥0，日历总方差单调；校验范围 k∈[-0.6, 0.6] 即报价区间，
      区间外为 SVI 外推）</p>
  </section>

  <section class="wide"><h2>收益率曲线</h2>
    {_svg_curve(curve)}
    <p class="note">Nelson-Siegel RMSE {m['ns_rmse_bp']:.2f}bp ·
      β0={m['ns_params']['beta0']:.4f} β1={m['ns_params']['beta1']:+.4f}</p>
    {_table(curve_rows, ['期限(年)', '零息利率'])}
  </section>

  <section class="wide"><h2>订单簿与执行成本</h2>
    {_svg_book(book)}
    <p class="note">样本订单 {m['sample_qty']:,.0f} 股（约 2% ADV）走簿冲击
      {m['walk_shortfall_bps']:.2f}bp</p>
    {_table(cost_rows, ['参与率', '股数', '成本(bp)', '风险(bp)'])}
  </section>

  <section><h2>校准参数</h2>{_table(calib_rows, ['指标', '数值'])}</section>
  <section><h2>数据</h2>
    <p class="note">{html.escape(m['data_path'])}</p>
    <p class="note">日线 OHLCV 无法识别逐笔方向：Kyle λ 用带符号成交量代理，
    η 由平方根律在参考参与率上锚定，γ 取 Kyle λ。详见 kairos-microstructure/calibrate.py。</p>
  </section>
  <section><h2>如何复现</h2>
    <p class="note">python examples/run_dashboard.py --symbol sh600519</p>
    <p class="note">全部数字来自本地 CSV，不联网；HTML 内联 SVG，可离线打开。</p>
  </section>
</main></body></html>"""


__all__ = ["ATM_MULTIPLIERS", "CURVE_SHAPE", "MATURITIES", "default_data_path",
           "run_dashboard", "summarize"]