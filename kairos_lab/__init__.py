"""Kairos Lab —— Kairos 量化套件的 **capstone**：真实数据端到端研究流水线。

本包不重新实现任何算法，而是把兄弟包串成可复现的研究链路：

- :mod:`kairos_lab.pipelines.e2e_equity`      数据 → 因子 → 组合 → 回测 → 风险（A 股）
- :mod:`kairos_lab.pipelines.e2e_multi_asset` 真实 ETF 面板 → 风险平价全天候 → 回测 → 风险
- :mod:`kairos_lab.pipelines.execution_demo`  父单 → TWAP/VWAP 切片 → 模拟撮合 → TCA

辅助模块：

- :mod:`kairos_lab._bootstrap` 未安装兄弟包时按相对路径修补 ``sys.path``；
- :mod:`kairos_lab.data`       真实 CSV 面板加载 / 离线合成面板；
- :mod:`kairos_lab.report`     Markdown 表格与 JSON 落盘工具。

导入本包即会调用 :func:`ensure_paths` 修补路径（幂等、不联网、无其它副作用）；
具体的流水线与兄弟包按需惰性 import，避免任一兄弟包缺失时整个 lab 不可用。
"""
from __future__ import annotations

from ._bootstrap import (
    ENV_ROOT,
    SIBLINGS,
    describe,
    ensure_paths,
    install_hint,
    kairos_root,
    repo_root,
    require,
    sibling_dir,
)

ensure_paths()

__version__ = "0.1.0"

__all__ = [
    "SIBLINGS", "ENV_ROOT", "ensure_paths", "require", "kairos_root", "repo_root",
    "sibling_dir", "install_hint", "describe", "__version__",
]
