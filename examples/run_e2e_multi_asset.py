"""示例 ②：真实跨资产 ETF 的「全天候」风险平价配置流水线（数据 → 组合 → 回测 → 风险）。

用法::

    python examples/run_e2e_multi_asset.py
    python examples/run_e2e_multi_asset.py --data-dir ../kairos-data/data/etf
    python examples/run_e2e_multi_asset.py --est-window 250 --rebalance 63
    python examples/run_e2e_multi_asset.py --optimizer mean_variance --target-vol 0.08

默认从 ``<kairos_root>/kairos-data/data/etf`` 读 9 只跨资产 ETF 的真实日线
（A 股宽基 / 海外权益 / 黄金 / 国债 / 货币）；本地缺失时联网抓取，再失败回退合成面板。
产物写入 ``research/e2e_multi_asset/``。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kairos_lab                                        # noqa: E402  触发 sys.path 引导
from kairos_lab import data as kld                        # noqa: E402
from kairos_lab.pipelines import e2e_multi_asset as ma    # noqa: E402
from kairos_lab.pipelines.e2e_equity import OPTIMIZERS    # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数。"""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None,
                   help="真实行情 CSV 目录（默认 <kairos_root>/kairos-data/data/etf）")
    p.add_argument("--out-dir", default=None, help="产物输出目录")
    p.add_argument("--name", default="e2e_multi_asset", help="本次运行的名称")
    p.add_argument("--est-window", type=int, default=120, help="协方差估计窗口（期）")
    p.add_argument("--rebalance", type=int, default=21, help="再平衡间隔（期）")
    p.add_argument("--optimizer", choices=OPTIMIZERS, default="risk_parity",
                   help="组合求解器")
    p.add_argument("--shrink-target", choices=("const_corr", "diag"), default="const_corr",
                   help="Ledoit-Wolf 收缩目标")
    p.add_argument("--risk-aversion", type=float, default=5.0, help="均值-方差的 λ")
    p.add_argument("--target-vol", type=float, default=None,
                   help="目标年化波动；启用后按波动率缩放，剩余敞口进现金池")
    p.add_argument("--weight-lag", type=int, default=1, help="权重滞后期数（防未来）")
    p.add_argument("--cost-rate", type=float, default=0.001, help="单边换手成本率")
    p.add_argument("--confidence", type=float, default=0.95, help="VaR/ES 置信度")
    p.add_argument("--n-trials", type=int, default=5, help="DSR 的试验次数")
    p.add_argument("--with-cash", action="store_true",
                   help="把货币 ETF 也纳入风险平价（默认作为现金池排除，否则它会吸走 >90% 权重）")
    p.add_argument("--start", default=None, help="样本起始日（YYYY-MM-DD）")
    p.add_argument("--end", default=None, help="样本结束日（YYYY-MM-DD）")
    p.add_argument("--max-days", type=int, default=None, help="只取最近 N 个交易日")
    p.add_argument("--no-fetch", action="store_true", help="本地无数据时不联网抓取")
    p.add_argument("--synthetic", action="store_true", help="强制使用离线合成面板")
    p.add_argument("--seed", type=int, default=11, help="合成面板的随机种子")
    p.add_argument("--quiet", action="store_true", help="只打印最终摘要")
    p.add_argument("--default-out", default=here, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    """入口：解析参数 → 取数 → 跑流水线 → 打印摘要。"""
    args = parse_args(argv)
    out_dir = args.out_dir or os.path.join(args.default_out, "research", args.name)

    if args.synthetic:
        prices = kld.synthetic_prices(n_assets=5, n_days=500, seed=args.seed,
                                      mu_low=-0.02, mu_high=0.20, vol_low=0.05,
                                      vol_high=0.28, market_vol=0.12, prefix="C")
        volumes = kld.synthetic_volumes(prices, seed=args.seed + 1)
        source = f"synthetic({prices.shape[1]}x{prices.shape[0]}, seed={args.seed})"
    else:
        prices, volumes, source = kld.resolve_panel(
            dataset="etf", data_dir=args.data_dir, fetch=not args.no_fetch,
            start=args.start, end=args.end, max_days=args.max_days,
            n_assets=5, n_days=500, seed=args.seed, verbose=not args.quiet)

    if not args.quiet:
        print(f"[lab] kairos_root={kairos_lab.kairos_root()}")
        print(f"[lab] 面板: {prices.shape[0]} 交易日 × {prices.shape[1]} 资产 "
              f"({prices.index[0].date()} ~ {prices.index[-1].date()})，来源={source}")
        print(f"[lab] 资产: {list(prices.columns)}")

    result = ma.run_multi_asset_pipeline(
        prices, volumes, est_window=args.est_window, rebalance=args.rebalance,
        cost_rate=args.cost_rate, out_dir=out_dir, optimizer=args.optimizer,
        shrink_target=args.shrink_target, risk_aversion=args.risk_aversion,
        weight_lag=args.weight_lag, target_vol=args.target_vol,
        confidence=args.confidence, n_trials=args.n_trials,
        cash_assets=None if args.with_cash else "auto", source=source, name=args.name)

    print(ma.summarize(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
