"""本地开发引导：让未 ``pip install`` 的兄弟包也能被 import。

Kairos Lab 是 Kairos 量化套件的 capstone，运行时需要 7 个兄弟包
（``kairos_data`` / ``kairos_factor`` / ``kairos_portfolio`` / ``kairos_backtest`` /
``kairos_risk`` / ``kairos_execution`` / ``kairos_ml``）。它们既可以通过
``pyproject.toml`` 里声明的 ``git+https`` 依赖正常安装，也可能只是与本仓库
并排 checkout 在同一父目录下（开发场景）。本模块负责兼容这两种情况：

- 已安装（或已在 ``sys.modules``）→ 直接使用；
- 未安装但兄弟目录存在 → 把该目录插入 ``sys.path``，再按普通包 import。

路径全部用 ``os.path`` 相对本文件定位，**不硬编码任何绝对路径**；可用环境变量
``KAIROS_ROOT`` 覆盖兄弟仓库所在的父目录（例如把它们放在别处时）。

目录布局假设::

    <kairos_root>/
      kairos-lab/kairos_lab/_bootstrap.py     # 本文件
      kairos-data/kairos_data/__init__.py
      kairos-factor/kairos_factor/__init__.py
      ...

设计原则：只做 ``sys.path`` 修补与 import，不联网、无副作用、可重复调用。
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Dict, Optional

#: 包名 -> 兄弟仓库目录名（相对 ``kairos_root``）
SIBLINGS: Dict[str, str] = {
    "kairos_data": "kairos-data",
    "kairos_factor": "kairos-factor",
    "kairos_portfolio": "kairos-portfolio",
    "kairos_backtest": "kairos-backtest",
    "kairos_risk": "kairos-risk",
    "kairos_execution": "kairos-execution",
    "kairos_ml": "kairos-ml",
}

#: 覆盖兄弟仓库父目录的环境变量名
ENV_ROOT = "KAIROS_ROOT"

#: ``pip install`` 时使用的 git 依赖（与 pyproject.toml 保持一致，仅用于错误提示）
GIT_OWNER = "Bruce848647703"

_HERE = os.path.dirname(os.path.abspath(__file__))      # .../kairos-lab/kairos_lab
_REPO_ROOT = os.path.dirname(_HERE)                     # .../kairos-lab


def repo_root() -> str:
    """返回本仓库（kairos-lab）根目录的绝对路径。"""
    return _REPO_ROOT


def kairos_root() -> str:
    """返回兄弟仓库所在的父目录。

    优先级：环境变量 ``KAIROS_ROOT`` > 本仓库的上一级目录。
    """
    override = os.environ.get(ENV_ROOT)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.dirname(_REPO_ROOT)


def sibling_dir(pkg: str) -> Optional[str]:
    """返回某兄弟包对应仓库目录的绝对路径（不存在时返回 None）。"""
    dirname = SIBLINGS.get(pkg)
    if dirname is None:
        return None
    path = os.path.join(kairos_root(), dirname)
    return path if os.path.isdir(os.path.join(path, pkg)) else None


def _findable(pkg: str) -> bool:
    """判断 ``pkg`` 当前是否可被 import（不执行其代码）。"""
    if pkg in sys.modules:
        return True
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def ensure_paths() -> Dict[str, str]:
    """确保所有兄弟包可被 import：必要时把其仓库目录加入 ``sys.path``。

    返回 ``{包名: 状态}``，状态取值：

    - ``"loaded"``   ：已在 ``sys.modules`` 中；
    - ``"installed"``：已安装（可被 import，无需改 ``sys.path``）；
    - ``"local"``    ：本次把兄弟目录加入 ``sys.path`` 后可用；
    - ``"missing"``  ：既未安装也找不到兄弟目录。

    可重复调用，幂等。
    """
    status: Dict[str, str] = {}
    for pkg in SIBLINGS:
        if pkg in sys.modules:
            status[pkg] = "loaded"
            continue
        if _findable(pkg):
            status[pkg] = "installed"
            continue
        path = sibling_dir(pkg)
        if path is None:
            status[pkg] = "missing"
            continue
        if path not in sys.path:
            sys.path.insert(0, path)
        status[pkg] = "local"
    return status


def install_hint(pkg: str) -> str:
    """给出某兄弟包不可用时的安装指引文本。"""
    return (
        f"缺少依赖 {pkg}。两种解决方式：\n"
        f"  1) pip install \"git+https://github.com/{GIT_OWNER}/{SIBLINGS.get(pkg, pkg)}@main\"\n"
        f"  2) 把 {SIBLINGS.get(pkg, pkg)}/ 仓库 checkout 到 {kairos_root()}/ 下"
        f"（或设置环境变量 {ENV_ROOT} 指向其父目录）"
    )


def require(pkg: str):
    """import 并返回一个兄弟包；不可用时抛出带中文安装指引的 ``ImportError``。"""
    if pkg not in SIBLINGS:
        raise KeyError(f"未知的兄弟包: {pkg!r}，可选: {sorted(SIBLINGS)}")
    ensure_paths()
    if pkg in sys.modules:
        return sys.modules[pkg]
    try:
        return importlib.import_module(pkg)
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise ImportError(install_hint(pkg)) from exc


def describe() -> str:
    """返回一段人类可读的引导状态说明（用于示例/测试的诊断输出）。"""
    lines = [f"kairos_root = {kairos_root()}"]
    for pkg, state in sorted(ensure_paths().items()):
        lines.append(f"  {pkg:<18} {state}")
    return "\n".join(lines)
