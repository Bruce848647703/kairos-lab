"""研究方向看板流水线测试（离线 CSV）。"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

os.environ.setdefault("KAIROS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from kairos_lab.pipelines import run_dashboard, summarize_dashboard  # noqa: E402


def _write_csv(path, n=400, seed=5):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.015, size=n - 1)
    prices = 50.0 * np.exp(np.cumsum(returns))
    volumes = rng.lognormal(np.log(2e6), 0.3, size=n)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("date,close,volume\n")
        for i, (p, v) in enumerate(zip(np.concatenate([[50.0], prices]), volumes)):
            fh.write(f"2021-01-{i % 28 + 1:02d},{p:.6f},{v:.0f}\n")


@pytest.fixture(scope="module")
def dashboard(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dash")
    csv_path = tmp / "sample.csv"
    _write_csv(str(csv_path))
    result = run_dashboard(symbol="TEST", data_path=str(csv_path),
                           out_dir=str(tmp / "out"), quiet=True)
    return result


def test_outputs_exist(dashboard):
    for key in ("html", "report", "metrics_json"):
        assert os.path.isfile(dashboard[key])
    assert os.path.getsize(dashboard["html"]) > 2000


def test_html_contains_three_panels(dashboard):
    text = open(dashboard["html"], encoding="utf-8").read()
    assert "波动率曲面" in text
    assert "收益率曲线" in text
    assert "订单簿与执行成本" in text
    assert "<svg" in text and "http://" not in text.split("<style>")[0]


def test_metrics_contract(dashboard):
    m = dashboard["metrics"]
    assert m["symbol"] == "TEST"
    assert 0 < m["sigma_annual"] < 2
    assert m["adv"] > 0
    assert len(m["surface_atm"]) == 5
    assert len(m["curve_zero"]) == len(m["curve_pillars"])
    assert m["cost_curve"] and m["ac"] and m["ac"][0]["cost_bps"] > 0
    assert m["sigma_intraday"] is not None or m["sigma_intraday"] is None

    with open(dashboard["metrics_json"], encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["TEST"]["symbol"] == "TEST"


def test_summary(dashboard):
    text = summarize_dashboard(dashboard)
    assert "TEST" in text and "bp" in text


def test_multi_symbol_and_interactive_controls(tmp_path):
    paths = {}
    for symbol in ("AAA", "BBB"):
        csv_path = tmp_path / f"{symbol}.csv"
        _write_csv(str(csv_path), seed=3 if symbol == "AAA" else 9)
        paths[symbol] = str(csv_path)

    result = run_dashboard(symbols=["AAA", "BBB"], data_paths=paths,
                           out_dir=str(tmp_path / "multi"), quiet=True)
    assert len(result["panels"]) == 2
    text = open(result["html"], encoding="utf-8").read()
    assert 'id="sym"' in text and 'id="q"' in text and 'id="lam"' in text
    assert 'data-symbol="AAA"' in text and 'data-symbol="BBB"' in text
    assert "<script>" in text
    assert "多标的对比" in text

    with open(result["metrics_json"], encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert set(loaded) == {"AAA", "BBB"}
