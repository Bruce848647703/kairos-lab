"""示例 ①：真实 A 股数据的端到端研究流水线（数据 → 因子 → 组合 → 回测 → 风险）。

用法::

    python examples/run_e2e_equity.py
    python examples/run_e2e_equity.py --data-dir ../kairos-data/data/ashare
    python examples/run_e2e_equity.py --lookback 20 --horizon 5 --top-quantile 0.2
    python examples/run_e2e_equity.py --optimizer mean_variance --max-days 800

默认从 ``<kairos_root>/kairos-data/data/ashare`` 读已提交的真实前复权日线；
本地缺失时联网抓取（``--no-fetch`` 关闭），再失败则回退到离线合成面板并在
输出与报告里明确标注来源。产物写入 ``research/e2e_equity/``。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kairos_lab                                        # noqa: E402  触发 sys.path 引导
from kairos_lab import data as kld                        # noqa: E402
from kairos_lab.pipelines import e2e_equity as eq         # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数。"""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None,
                   help="真实行情 CSV 目录（默认 <kairos_root>/kairos-data/data/ashare）")
    p.add_argument("--out-dir", default=None,
                   help="产物输出目录（默认 <repo>/research/<name>）")
    p.add_argument("--name", default="e2e_equity", help="本次运行的名称（决定默认输出目录）")
    p.add_argument("--lookback", type=int, default=60, help="动量因子回看期")
    p.add_argument("--horizon", type=int, default=5, help="远期收益期限，同时作为调仓间隔")
    p.add_argument("--top-quantile", type=float, default=0.3, help="做多因子分数最高的比例")
    p.add_argument("--weighting", choices=("equal", "score"), default="equal",
                   help="因子组合加权方式：组内等权 / 按分数倾斜")
    p.add_argument("--cost-rate", type=float, default=0.001, help="单边换手成本率")
    p.add_argument("--optimizer", choices=eq.OPTIMIZERS, default="risk_parity",
                   help="优化组合使用的求解器")
    p.add_argument("--est-window", type=int, default=120, help="协方差估计窗口")
    p.add_argument("--opt-rebalance", type=int, default=21, help="优化组合调仓间隔")
    p.add_argument("--risk-aversion", type=float, default=5.0, help="均值-方差的 λ")
    p.add_argument("--weight-lag", type=int, default=1, help="权重相对因子的滞后期数（防未来）")
    p.add_argument("--confidence", type=float, default=0.95, help="VaR/ES 置信度")
    p.add_argument("--n-trials", type=int, default=6, help="DSR 的试验次数")
    p.add_argument("--start", default=None, help="样本起始日（YYYY-MM-DD）")
    p.add_argument("--end", default=None, help="样本结束日（YYYY-MM-DD）")
    p.add_argument("--max-assets", type=int, default=None, help="只取前 N 个标的（快速试跑）")
    p.add_argument("--max-days", type=int, default=None, help="只取最近 N 个交易日")
    p.add_argument("--no-fetch", action="store_true", help="本地无数据时不联网抓取")
    p.add_argument("--synthetic", action="store_true", help="强制使用离线合成面板")
    p.add_argument("--seed", type=int, default=7, help="合成面板的随机种子")
    p.add_argument("--quiet", action="store_true", help="只打印最终摘要")
    p.add_argument("--default-out", default=here, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    """入口：解析参数 → 取数 → 跑流水线 → 打印摘要。"""
    args = parse_args(argv)
    out_dir = args.out_dir or os.path.join(args.default_out, "research", args.name)

    if args.synthetic:
        prices = kld.synthetic_prices(n_assets=12, n_days=600, seed=args.seed)
        volumes = kld.synthetic_volumes(prices, seed=args.seed + 1)
        source = f"synthetic({prices.shape[1]}x{prices.shape[0]}, seed={args.seed})"
    else:
        prices, volumes, source = kld.resolve_panel(
            dataset="ashare", data_dir=args.data_dir, fetch=not args.no_fetch,
            start=args.start, end=args.end, max_assets=args.max_assets,
            max_days=args.max_days, n_assets=12, n_days=600, seed=args.seed,
            verbose=not args.quiet)

    if not args.quiet:
        print(f"[lab] kairos_root={kairos_lab.kairos_root()}")
        print(f"[lab] 引导状态: {kairos_lab.ensure_paths()}")
        print(f"[lab] 面板: {prices.shape[0]} 交易日 × {prices.shape[1]} 资产 "
              f"({prices.index[0].date()} ~ {prices.index[-1].date()})，来源={source}")

    result = eq.run_equity_pipeline(
        prices, volumes, lookback=args.lookback, horizon=args.horizon,
        cost_rate=args.cost_rate, out_dir=out_dir, top_quantile=args.top_quantile,
        weighting=args.weighting, est_window=args.est_window,
        opt_rebalance=args.opt_rebalance, optimizer=args.optimizer,
        risk_aversion=args.risk_aversion, weight_lag=args.weight_lag,
        confidence=args.confidence, n_trials=args.n_trials, source=source,
        name=args.name)

    print(eq.summarize(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
