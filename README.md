# Kairos Lab

> Kairos 量化系列的 **capstone** —— 一个**自研、轻量**的端到端研究实验室：用**真实行情数据**把 `kairos_data` / `kairos_factor` / `kairos_portfolio` / `kairos_backtest` / `kairos_risk` / `kairos_execution` / `kairos_ml` 七个兄弟包串成可复现的研究流水线，并自动产出中文研究报告。

本仓库**不重新实现任何算法**。它的价值在于「串联」与「交付」：把散落在各个包里的能力
按真实研究流程组装起来，用真实数据跑出真实数字，并把每一步的输入、口径、产物都写进
`REPORT.md` / `metrics.json` / `equity.csv`，让整条链路可审计、可复现、可对比。

必需依赖只有 `numpy` 与 `pandas`；测试全部离线、固定 seed、数秒内跑完。

## 端到端流程

```
                        ┌──────────────────────────────────────────────┐
   kairos_data          │  真实 CSV 行情（38 只 A 股 / 9 只跨资产 ETF）  │
   load_ashare_panel    │  index=交易日, columns=标的；缺失自动联网抓取   │
                        └───────────────────┬──────────────────────────┘
                                            │ prices, volumes
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                   ▼                                   ▼
 ① A 股因子链                        ② 跨资产全天候                       ③ 执行与 TCA
 kairos_factor                       kairos_portfolio                    kairos_execution
  pct_change(60) 动量                 pct_change → 滚动 120 期            make_ohlcv / to_bars
  winsorize + zscore                  ledoit_wolf(const_corr)             TWAP/VWAP/IS/Iceberg
  forward_returns(5)                  risk_parity 定点迭代                  .schedule(parent, bars)
  rank_ic / ic_summary                （负相关无解 → δ=1 → 逆波动率）      SimBroker
  quantile_returns / FactorReport     target_volatility 叠加现金池          （延迟/参与率/滑点/佣金）
        │                                   │                              execute_children
        ▼ 因子分数 → 目标权重                 ▼ 目标权重面板                   │
 kairos_portfolio                      kairos_backtest                      ▼
  因子选股 top 30%                       VectorBacktester(cost_rate)        analyze → TCAReport
  ledoit_wolf + risk_parity/              vs 等权 / 逆波动率对照              滑点·成交率·价差
  mean_variance 作对照                                                      ·冲击·实现差额
        ▼                                   ▼
 kairos_backtest                         kairos_risk
  VectorBacktester(cost_rate=0.001)       historical_var / expected_shortfall
  因子组合 vs 优化组合 vs 等权基准          modified_var / tail_summary(EVT-GPD)
        ▼                                 drawdown_report / psr / dsr
 kairos_risk                              component_risk / 类别风险归集
  VaR·ES·修正VaR·回撤报告·PSR·DSR
  fit_factor_model → 因子风险分解
  component_risk → 欧拉成分风险
        ▼
 kairos_ml（可选）
  walk_forward_splits 样本外 IC
  hit_rate / signal_pnl / t_statistic
  跨包一致性校验（km.ic ≡ kf.rank_ic）
        ▼
 research/<name>/REPORT.md + metrics.json + equity.csv + weights.csv
```

## 特性

- **真实数据优先**：示例默认读 `kairos-data` 仓库已提交的前复权日线 CSV
  （38 只 A 股 + 9 只跨资产 ETF）；本地缺失时用 `kairos_data.ashare.fetch_universe`
  联网抓取；再失败（离线环境）自动回退到**固定 seed 的合成面板**，并在报告里明确标注来源。
- **零安装也能跑**：`kairos_lab/_bootstrap.py` 用 `os.path` 相对定位兄弟仓库目录并按需
  插入 `sys.path`，因此**并排 checkout 即可运行**，无需先 `pip install` 七个包；
  也可用环境变量 `KAIROS_ROOT` 指定它们的父目录。对外发布时 `pyproject.toml` 用
  `git+https://github.com/Bruce848647703/kairos-<pkg>@main` 声明依赖，`pip install .` 自动拉取。
- **严格防未来函数**：因子只用 `prices[:t]`；远期收益取 `(t, t+h]`；目标权重相对信号再
  滞后 `weight_lag` 期；`VectorBacktester` 内部还有一期成交滞后。测试里用「把价格截断到 t
  重算，结果必须逐位一致」的方式**证明**了这一点，而不是靠口头承诺。
- **永不静默降级**：优化器求解失败时在诊断表的 `solver` 列写明实际使用的回退路径
  （例如 `risk_parity(δ=1,damping=0.5)`、`inverse_vol(风险平价无解)`），报告里如实呈现。
- **产物即报告**：每条流水线都写出中文 `REPORT.md`（含真实数字与口径说明）、
  `metrics.json`（**严格合法** JSON，NaN/Inf 转 `null`）、`equity.csv`、`weights.csv`。
- **跨包一致性校验**：用 `kairos_ml.ic(method="spearman")` 与 `kairos_factor.rank_ic`
  对同一批截面各自独立计算秩 IC，实测最大绝对差 `1.1e-16`（浮点精度级别）。

## 安装

### A. 外部用户（从 git 拉取全部兄弟包）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                  # 自动安装 7 个 kairos-* 包（git+https@main）
pip install -e ".[dev]"           # 额外装 pytest / scipy / matplotlib
pip install -e ".[scipy]"         # 只需要 long-only 约束优化（SLSQP）时
```

### B. 本地开发（兄弟包并排 checkout，无需安装）

```
<某个父目录>/
  kairos-lab/          # 本仓库
  kairos-data/         # 含真实数据 CSV
  kairos-factor/
  kairos-portfolio/
  kairos-backtest/
  kairos-risk/
  kairos-execution/
  kairos-ml/
```

```bash
cd kairos-lab
python -c "import kairos_lab; print(kairos_lab.describe())"   # 查看引导状态
python -m pytest -q                                           # 直接可跑，无需 pip install
# 兄弟仓库放在别处时：
export KAIROS_ROOT=/path/to/parent-of-kairos-data
```

## 快速开始

```bash
# ① A 股：数据 → 因子 → 组合 → 回测 → 风险
python examples/run_e2e_equity.py --data-dir ../kairos-data/data/ashare

# ② 跨资产 ETF：风险平价全天候 → 回测 → 风险
python examples/run_e2e_multi_asset.py --data-dir ../kairos-data/data/etf
python examples/run_e2e_multi_asset.py --target-vol 0.03        # 叠加目标波动 + 现金池

# ③ 执行：真实日线 bar 上的 TWAP / VWAP / IS 切片与 TCA
python examples/run_execution_demo.py
python examples/run_execution_demo.py --symbol sz000858 --side sell --algos twap vwap is iceberg

make test        # 28 个离线测试，数秒全绿
make demo        # 依次跑三条流水线
```

也可以在代码里直接调用：

```python
import kairos_lab                                  # 导入即修补 sys.path
from kairos_lab import data as kld
from kairos_lab.pipelines import run_equity_pipeline

prices, volumes = kld.load_panel(dataset="ashare")  # 真实 A 股面板
res = run_equity_pipeline(prices, volumes, lookback=60, horizon=5,
                          cost_rate=0.001, top_quantile=0.3,
                          out_dir="research/e2e_equity")

res["factor"]["rank_ic_summary"]      # {'count':1865, 'mean':-0.01, 'ir':-0.036, ...}
res["backtest"]["metrics"]["factor"]  # {'total_return':3.35, 'sharpe':0.836, ...}
res["risk"]["var"], res["risk"]["es"] # 0.0242, 0.0390
res["artifacts"]["report_md"]         # research/e2e_equity/REPORT.md
```

## 真实数据结果摘要

以下数字由本仓库在上述命令下**实际跑出**（产物见 `research/*/REPORT.md`），
非手写示例。数据区间为 `kairos-data` 已提交的前复权日线。

### ① A 股因子链（38 只 A 股，2018-10-16 ~ 2026-09-23，1930 个交易日）

| 项目 | 数值 |
|---|---|
| 动量因子 `mom60` 秩 IC | **−0.0100**（ICIR −0.0362，t = −1.56，IC>0 占比 49.7%，覆盖率 96.9%） |
| 分层收益（每 5 日 → 年化） | Q1 **+54.3%** / Q2 +24.2% / Q3 +15.1% / Q4 +22.0% / Q5 +33.9% |
| 因子多头组合（top 30%，5 日调仓） | 累计 **+334.65%**，年化 +21.15%，波动 27.46%，夏普 **0.836**，最大回撤 55.03%，年化换手 21.8× |
| 因子选股 + LW 协方差 + 风险平价 | 累计 +92.43%，年化 +8.92%，波动 24.42%，夏普 0.473，回撤 55.85%，换手 13.9× |
| 全宇宙等权买入持有（基准） | 累计 +1170.71%，年化 +39.37%，波动 36.24%，夏普 1.075，回撤 29.76% |
| 风险（因子组合） | 日 VaR(95%) **0.0242**、ES(95%) **0.0390**、修正 VaR 0.0257、PSR **0.989**、DSR(6 trials) **0.844** |
| 样本外验证（26 折 walk-forward） | 样本内 IC −0.0137 → 样本外 IC −0.0148，符号保持 58%，多空命中率 50.9% |
| 跨包一致性 | `kairos_ml.ic(spearman)` vs `kairos_factor.rank_ic` 最大绝对差 **1.1e-16** |

> **诚实解读**：在这 38 只大盘股上，60 日动量的秩 IC 是**微负**的（t = −1.56，不显著），
> 分层收益呈 **Q1 最高** 的形态 —— 即 A 股短中期存在**反转**而非动量。因子多头组合
> 跑输等权基准，也说明「股票池本身是事后精选的赢家」带来的幸存者偏差远大于因子贡献。
> 报告如实呈现这些数字，不做美化；这正是一个 lab 该有的样子。

### ② 跨资产全天候（9 只 ETF，2017-01-18 ~ 2026-09-24，2353 个交易日）

| 组合 | 累计收益 | 年化 | 年化波动 | 夏普 | 最大回撤 | 年化换手 |
|---|---|---|---|---|---|---|
| **滚动风险平价**（120 期窗口 / 21 期调仓 / 107 次再平衡） | +63.86% | **+5.43%** | **4.73%** | **1.141** | **8.02%** | 1.8× |
| 等权（1/N） | +95.37% | +7.44% | 13.21% | 0.609 | 25.08% | 1.9× |
| 滚动逆波动率 | +7.91% | +0.82% | 1.02% | 0.806 | 1.58% | 0.7× |

| 风险与结构 | 数值 |
|---|---|
| 日 VaR(95%) / ES(95%) / 修正 VaR | **0.0041 / 0.0071 / 0.0046** |
| PSR / DSR(5 trials) | **1.000 / 0.987** |
| 分散化比率 / 风险有效资产数 | 1.638 / **7.67**（共 9 个资产） |
| 横截面平均相关系数 | 0.209（最低 −0.217：国债 vs 沪深300；最高 0.853：创业板 vs 中证500） |
| Ledoit-Wolf 收缩强度 δ 均值 | 0.363（其中 34/107 次触发 δ=1 全收缩，见下） |
| 期末类别权重 | 债券 91.2%、海外权益 4.1%、A 股权益 3.1%、商品 1.6%、现金 0% |
| `--target-vol 0.03` 变体 | 年化 3.73%、波动 **3.14%**、回撤 **4.15%**、夏普 1.181，平均敞口 76.8%，货币 ETF 平均承接 22% |

> **两个真实的工程结论**（都写在报告里）：
> 1. **不加杠杆的风险平价会把权重压给低波动资产**。本样本中债券 ETF 年化波动仅 ~2%，
>    权益 ~21–30%，「等风险贡献」自然让债券占到九成 —— 这是风险平价的固有属性，
>    真实全天候组合靠债券加杠杆把总波动抬到目标水平，本流水线用 `--target-vol`
>    （配合 `max_leverage>1`）或 `risk_budget` 提供出口。
> 2. **含负相关资产时风险平价可能无 long-only 内部解**。黄金/国债与权益的相关系数为负，
>    会使某资产的边际风险贡献 `(Σw)_i ≤ 0`，定点迭代把它的权重压到 0 后残差永久卡在 `1/n`。
>    本仓库先用「逆波动率起点处的边际贡献是否为正」做**O(1) 可行性预判**，不可行时依次回退
>    「相关系数全收缩 δ=1 后重解」→「逆波动率解析解」，实测把求解耗时从 8.5s 降到 1.4s，
>    且 107 次再平衡**零次**退化到等权。

### ③ 执行与 TCA（贵州茅台 sh600519，最近 60 根真实日线）

父单 **BUY 7,200**（占执行窗口成交量 1.996%），窗口 15 根 bar（2026-09-02 ~ 2026-09-22），
撮合设置 `latency=1`、参与率 ≤10%/bar、滑点 5bp、佣金万三（最低 5 元）；
决策价 1299.56、到达价 1302.80、窗口市场基准 VWAP 1285.06。

| 算法 | 子单/成交 | 成交率 | 成交均价 | 滑点 | 相对市场 VWAP | 延迟成本 | 冲击估计 | 实现差额 | 佣金 |
|---|---|---|---|---|---|---|---|---|---|
| TWAP | 15 / 15 | 100% | 1281.49 | −163.56 bp | **−27.75 bp** | +24.93 bp | ≈1.41 bp | −136.08 bp | 2,768 |
| VWAP | 15 / 15 | 100% | 1282.30 | −157.34 bp | **−21.44 bp** | +24.93 bp | ≈1.41 bp | −129.84 bp | 2,770 |
| IS（decay=0.7） | 11 / 12 | 100% | 1307.22 | +33.94 bp | **+172.48 bp** | +24.93 bp | ≈1.41 bp | +61.98 bp | 2,824 |

> 该窗口价格整体下行，因此三种算法相对到达价都「买便宜了」（滑点为负）；
> 区分算法优劣应看**相对市场 VWAP 的价差**：VWAP 切片最贴近市场自然成交节奏（−21bp），
> 前重后轻的 IS 算法为了抢速度把成交集中在窗口前段，在下跌行情里反而比基准贵 172bp。
> 延迟成本（+24.93bp）三者相同 —— 它来自决策价到到达价的漂移，与算法无关，
> 这正是执行风险的主要来源。

## 项目结构

```
kairos-lab/
  kairos_lab/
    __init__.py            导入即调用 ensure_paths() 修补 sys.path
    _bootstrap.py          兄弟包定位与按需 import（KAIROS_ROOT 可覆盖）
    data.py                真实 CSV 面板加载 / 联网抓取 / 离线合成面板 / 三级回退解析
    report.py              Markdown 表格渲染 + 严格 JSON 序列化 + 落盘
    pipelines/
      e2e_equity.py        run_equity_pipeline()      数据→因子→组合→回测→风险
      e2e_multi_asset.py   run_multi_asset_pipeline() 风险平价全天候→回测→风险
      execution_demo.py    run_execution_demo()       切片→撮合→TCA
  examples/                三个可直接运行的入口脚本（argparse，产物写 research/<name>/）
  tests/                   28 个离线测试（含防未来函数与跨包一致性验证）
  research/                运行产物（已 gitignore）
```

## API 概览

| 对象 | 说明 |
|---|---|
| `run_equity_pipeline(prices, volumes=None, lookback=60, horizon=5, cost_rate=0.001, out_dir=None, top_quantile=0.3, ...)` | A 股端到端；返回 `{meta, factor, validation, portfolio, backtest, risk, artifacts}` |
| `run_multi_asset_pipeline(prices, volumes=None, est_window=120, rebalance=21, cost_rate=0.001, out_dir=None, optimizer="risk_parity", target_vol=None, ...)` | 跨资产全天候；返回 `{meta, portfolio, static, frontier, backtest, risk, artifacts}` |
| `run_execution_demo(bars_or_prices=None, parent_qty=None, symbol=..., side="buy", compare=("twap","vwap"), ...)` | 执行与 TCA；返回 `{meta, bars, window, runs, tca, artifacts}` |
| `summarize_equity / summarize_multi_asset / summarize_execution` | 把结果压成一段可直接 `print` 的中文摘要 |
| `factor_stage / portfolio_stage / backtest_stage / risk_stage` | A 股流水线的四个阶段，可单独调用 |
| `momentum_factor / build_preprocessor / factor_weights / optimized_weights` | 因子构造、预处理、权重生成（可复用积木） |
| `solve_weights / risk_parity_chain / risk_parity_robust / risk_parity_feasible / tradable_assets` | 组合求解与三级回退链、可行性预判、停牌剔除 |
| `rolling_weights / rolling_inverse_vol / static_weights / aggregate_by_class` | 滚动/静态权重构建与类别归集 |
| `to_bar_list / pick_window / make_algo / execute_once / tca_frame` | 执行链路积木（多形态行情归一、窗口选择、单算法执行、TCA 汇总） |
| `walk_forward_validation` | kairos_ml 样本外验证 + 跨包一致性校验（可选阶段） |
| `kairos_lab.data.load_panel / resolve_panel / resolve_ohlcv / ohlcv_frame / synthetic_prices` | 数据入口 |
| `kairos_lab.report.frame_to_md / dict_to_md / to_jsonable / write_report / write_json / write_csv` | 报告工具 |
| `kairos_lab.ensure_paths / require / kairos_root / describe` | 引导与诊断 |

## 设计要点

- **数据约定**：所有「面板」都是 `DataFrame(index=日期, columns=资产)`；权重面板同型，
  每行和 ≤ 1（不满仓即持币，`VectorBacktester` 把剩余部分视为零收益现金）。
- **防未来函数的三重保险**：① 因子/协方差只用 `[:t]` 的数据；② 权重相对信号滞后
  `weight_lag`（默认 1）期；③ 回测器内部 `held = w.shift(1)`。因此实际持仓相对信号共有
  `weight_lag + 1` 期延迟 —— 这是**保守**设定，宁可低估策略表现也不引入未来信息。
- **口径差异如实标注**：`kairos_risk.sharpe_ratio` 用**几何**年化收益，
  `kairos_backtest.sharpe_ratio` 用**算术**年化收益，两者数值略有差异属正常，报告里写明。
- **量纲一致性**：`kp.target_volatility` 要求 `target_vol` 与 `cov` 同量纲，而协方差是
  每期的，因此流水线把**年化**目标波动按 `target_vol / √252` 折算后再传入
  （否则缩放永远不触发 —— 这是踩过的坑，已写成测试）。
- **停牌与退化样本**：`load_ashare_panel` 会对停牌日做前向填充，导致某些窗口里出现
  「零方差资产」，使协方差奇异、风险平价失稳。`tradable_assets()` 在优化前剔除它们。
- **可选依赖优雅降级**：`min_variance` / `mean_variance` / `max_sharpe` 的 long-only 路径
  需要 scipy-SLSQP，缺失时捕获 `ImportError` 回退等权并在 `solver` 列标注；
  `kairos_ml` 缺失时样本外验证阶段整体跳过；`efficient_frontier` 同理。
- **JSON 严格合法**：`to_jsonable` 把 NaN/±Inf 转成 `null`，`json.dumps(allow_nan=False)`，
  因此 `metrics.json` 可被任何语言的严格 JSON 解析器读取（测试里用
  `parse_constant` 断言不出现 `NaN` 字面量）。
- **展示层与计算层分离**：`report.py` 只做格式化与落盘，不参与任何数值计算；
  整数列不会被渲染成 `1865.0000`（用 `to_numpy(dtype=object)` 而非 `iterrows`）。

## 测试

```bash
make test          # 或 python -m pytest -q
```

28 个测试全部离线（固定 seed 的合成面板，6×300 与 5×420），数秒跑完，覆盖：

- 三条流水线的返回结构、指标有限性、产物落盘、报告小节齐全；
- `metrics.json` / `tca.json` 是**严格合法** JSON；
- 权重合法性（每行和 ≤ 1、非负、风险平价满仓、成分风险占比和为 1）；
- **防未来函数**：价格截断到 t 重算，因子权重 / 优化权重 / 回测收益必须与全样本逐位一致；
- **参与率约束**：逐笔成交不超过 `participation_rate × bar.volume`；
- **子单量守恒**：`sum(child.qty) == parent.qty`；
- 确定性（同参数两次运行结果一致）、错误输入的中文报错、`KAIROS_ROOT` 覆盖生效。

## 许可

MIT © 2026 Bruce848647703，见 [LICENSE](LICENSE)。

## 数据声明

本仓库**不分发任何行情数据**。真实数据由兄弟仓库 `kairos-data` 的 `data/ashare/`
与 `data/etf/` 提供（来自腾讯/新浪公开行情接口，版权归原作者与数据源所有，
仅用于研究与演示，详见该目录下的 `DATA_NOTICE.md`）；本地缺失时可用
`kairos_data.ashare.fetch_universe` 自行抓取，或用 `--synthetic` 完全离线运行。
所有回测与 TCA 结果**不构成投资建议**。

## 参考与致谢

本项目为**独立原创实现**，未复制任何第三方代码。设计思路受业界通用范式
（动量/反转因子研究、IC-IR 与分层回测、Ledoit-Wolf 协方差收缩、风险平价与目标波动率、
向量化回测与换手成本、VaR/ES 与 EVT 极值、PSR/DSR 防过拟合、TWAP/VWAP/IS 执行算法与
TCA 成本分解、walk-forward 样本外验证）启发，在此向开源量化社区致谢。
流水线编排、报告生成、求解回退链与全部胶水代码均为本仓库自研；
所有算法实现来自同系列的 `kairos-*` 兄弟包，本仓库不重复造轮子。
