"""示例 ③：真实日线 bar 上的父单切片执行与 TCA（TWAP / VWAP / IS 对比）。

用法::

    python examples/run_execution_demo.py
    python examples/run_execution_demo.py --symbol sz000858 --side sell --n-slices 15
    python examples/run_execution_demo.py --algos twap vwap is iceberg --parent-qty 20000
    python examples/run_execution_demo.py --synthetic          # 强制离线合成行情

默认读 ``<kairos_root>/kairos-data/data/ashare/<symbol>.csv`` 的真实前复权日线，
取最近 ``--last`` 根 bar 作为执行窗口；本地缺失时联网抓取，再失败回退合成行情。
产物写入 ``research/execution_demo/``。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kairos_lab                                        # noqa: E402  触发 sys.path 引导
from kairos_lab import data as kld                        # noqa: E402
from kairos_lab.pipelines import execution_demo as ex     # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数。"""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None,
                   help="真实行情 CSV 目录（默认 <kairos_root>/kairos-data/data/ashare）")
    p.add_argument("--symbol", default=None, help="标的代码（默认取股票池第一个）")
    p.add_argument("--dataset", default="ashare", choices=("ashare", "etf"),
                   help="使用哪个数据集的真实行情")
    p.add_argument("--last", type=int, default=60, help="取最近多少根 bar")
    p.add_argument("--side", choices=("buy", "sell"), default="buy", help="买卖方向")
    p.add_argument("--algos", nargs="+", default=["twap", "vwap"], choices=list(ex.ALGOS),
                   help="要对比的执行算法")
    p.add_argument("--n-slices", type=int, default=15, help="TWAP/VWAP/IS 的切片数")
    p.add_argument("--parent-qty", type=float, default=None,
                   help="父单数量；省略时按窗口成交量的 --participation-of-volume 自动确定")
    p.add_argument("--participation-of-volume", type=float, default=0.02,
                   help="自动确定父单量时占窗口成交量的比例")
    p.add_argument("--lot-size", type=int, default=100, help="最小交易单位（0 表示允许碎股）")
    p.add_argument("--latency", type=int, default=1, help="成交延迟（根 bar）")
    p.add_argument("--participation-rate", type=float, default=0.10,
                   help="单根 bar 的成交量参与率上限")
    p.add_argument("--slippage-bps", type=float, default=5.0, help="模拟滑点（基点）")
    p.add_argument("--commission-rate", type=float, default=3e-4, help="佣金比例")
    p.add_argument("--commission-min", type=float, default=5.0, help="单笔最低佣金")
    p.add_argument("--market-ref", choices=("close", "open"), default="close",
                   help="市价单的撮合参考价")
    p.add_argument("--display-size", type=float, default=None,
                   help="iceberg 每批显露量（默认父单量的 1/4）")
    p.add_argument("--decay", type=float, default=0.7, help="IS 算法的几何衰减系数")
    p.add_argument("--out-dir", default=None, help="产物输出目录")
    p.add_argument("--name", default="execution_demo", help="本次运行的名称")
    p.add_argument("--no-fetch", action="store_true", help="本地无数据时不联网抓取")
    p.add_argument("--synthetic", action="store_true", help="强制使用离线合成行情")
    p.add_argument("--seed", type=int, default=7, help="合成行情的随机种子")
    p.add_argument("--quiet", action="store_true", help="只打印最终摘要")
    p.add_argument("--default-out", default=here, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    """入口：取 bar → 逐算法执行 → 打印 TCA 对比。"""
    args = parse_args(argv)
    out_dir = args.out_dir or os.path.join(args.default_out, "research", args.name)

    if args.synthetic:
        bars = kld.synthetic_ohlcv(
            kld.synthetic_prices(n_assets=1, n_days=max(args.last, 30), seed=args.seed,
                                 prefix="SIM"), seed=args.seed + 1, symbol="SIM")
        symbol, source = "SIM", f"synthetic OHLCV(seed={args.seed})"
    else:
        bars, symbol, source = kld.resolve_ohlcv(
            dataset=args.dataset, symbol=args.symbol, data_dir=args.data_dir,
            last=args.last, fetch=not args.no_fetch, n_bars=max(args.last, 30),
            seed=args.seed, verbose=not args.quiet)

    if not args.quiet:
        print(f"[lab] kairos_root={kairos_lab.kairos_root()}")
        print(f"[lab] 行情: {source} | symbol={symbol} | bar 数={len(bars)}")

    result = ex.run_execution_demo(
        bars, parent_qty=args.parent_qty, symbol=symbol, side=args.side,
        compare=tuple(args.algos), n_slices=args.n_slices,
        lot_size=(args.lot_size or None), latency=args.latency,
        participation_rate=args.participation_rate, slippage_bps=args.slippage_bps,
        commission_rate=args.commission_rate, commission_min=args.commission_min,
        market_ref=args.market_ref, display_size=args.display_size, decay=args.decay,
        target_participation=args.participation_of_volume, out_dir=out_dir,
        source=source, name=args.name)

    print(ex.summarize(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
