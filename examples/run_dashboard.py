"""流水线 ④：衍生品 / 利率 / 微观结构研究看板。

用法::

    python examples/run_dashboard.py --symbol sh600519
    python examples/run_dashboard.py --data path/to/ohlcv.csv --out-dir research/dashboard
    python examples/run_dashboard.py --synthetic --out-dir /tmp/lab/dashboard   # 离线自检

产物：``index.html``（内联 SVG，可离线打开）、``REPORT.md``、``metrics.json``。
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from kairos_lab.pipelines import run_dashboard, summarize_dashboard


def _write_synthetic_csv(path: str, n: int = 600, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.016, size=n - 1)
    prices = 100.0 * np.exp(np.cumsum(returns))
    volumes = rng.lognormal(mean=np.log(5e6), sigma=0.4, size=n)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("date,close,volume\n")
        for i, (price, volume) in enumerate(zip(np.concatenate([[100.0], prices]), volumes)):
            fh.write(f"2020-01-{i % 28 + 1:02d},{price:.6f},{volume:.0f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kairos Lab 研究方向看板")
    parser.add_argument("--symbol", default="sh600519")
    parser.add_argument("--symbols", default="", help="逗号分隔的多个标的，如 sh600519,sh600036")
    parser.add_argument("--data", default=None, help="OHLCV CSV 路径")
    parser.add_argument("--out-dir", default=os.path.join("research", "dashboard"))
    parser.add_argument("--synthetic", action="store_true", help="用离线合成数据自检")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data_path = args.data
    tmp = None
    if args.synthetic and not data_path:
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        tmp.close()
        _write_synthetic_csv(tmp.name)
        data_path = tmp.name

    try:
        names = [s.strip() for s in args.symbols.split(",") if s.strip()]
        symbol = "SYNTH" if (args.synthetic and not args.data) else args.symbol
        if names and data_path:
            result = run_dashboard(symbols=names,
                                   data_paths={n: data_path for n in names},
                                   out_dir=args.out_dir, quiet=args.quiet)
        elif names:
            result = run_dashboard(symbols=names, out_dir=args.out_dir, quiet=args.quiet)
        else:
            result = run_dashboard(symbol=symbol, data_path=data_path,
                                   out_dir=args.out_dir, quiet=args.quiet)
        print(f"OK  {summarize_dashboard(result)}")
    finally:
        if tmp is not None:
            os.unlink(tmp.name)


if __name__ == "__main__":
    main()
