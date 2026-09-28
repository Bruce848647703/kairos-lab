"""报告工具：把 numpy/pandas 结果渲染成 Markdown 表格并落盘。

三条端到端流水线都要写 ``REPORT.md`` + ``metrics.json`` + ``equity.csv``，
把这些重复的「格式化 / 序列化 / 落盘」逻辑集中在这里：

- :func:`frame_to_md` / :func:`series_to_md` / :func:`dict_to_md`：渲染 Markdown 表格；
- :func:`to_jsonable`：把嵌套的 dict / DataFrame / Series / dataclass / numpy 标量
  递归转成纯 JSON 可序列化结构（非有限浮点转 ``None``，保证输出是**严格合法** JSON）；
- :func:`write_text` / :func:`write_json` / :func:`write_report`：统一落盘。

设计原则：纯函数、无全局状态、只做展示层的事，不参与任何数值计算。
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

#: 缺失值的统一显示
NA = "—"

PathLike = Union[str, "os.PathLike[str]"]


# ---------------------------------------------------------------------------
# 标量 / 表格格式化
# ---------------------------------------------------------------------------
def is_number(value: Any) -> bool:
    """是否为可格式化的数值（bool 除外）。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    return False


def fmt(value: Any, floatfmt: str = ".4f", intfmt: str = "d") -> str:
    """把单个值格式化为字符串。

    - ``None`` / NaN / inf → :data:`NA`；
    - 整数（含 numpy 整数）→ 不带小数；
    - 其它数值 → ``floatfmt``；
    - 字符串与非数值对象 → ``str(value)``。
    """
    if value is None:
        return NA
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (np.integer,)):
        return format(int(value), intfmt)
    if isinstance(value, int):
        return format(value, intfmt)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if not math.isfinite(x):
            return NA
        return format(x, floatfmt)
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    return str(value)


def _clean_label(label: Any, fallback: str = "") -> str:
    """把索引/列标签渲染成表头文本。"""
    if label is None:
        return fallback
    if isinstance(label, pd.Timestamp):
        return str(label.date())
    return str(label)


def frame_to_md(df: pd.DataFrame, floatfmt: str = ".4f", index_label: str = "",
                max_rows: Optional[int] = None, max_cols: Optional[int] = None) -> str:
    """把 DataFrame 渲染成 Markdown 表格（含表头分隔行）。

    ``max_rows`` / ``max_cols`` 用于截断超大表格，被截断时追加省略提示行/列。
    空表返回一行斜体提示，避免生成非法 Markdown。
    """
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        return "_（无数据）_"
    rows = df.index.tolist()
    cols = list(df.columns)
    row_cut = col_cut = False
    if max_rows is not None and len(rows) > max_rows:
        rows = rows[:max_rows]
        row_cut = True
    if max_cols is not None and len(cols) > max_cols:
        cols = cols[:max_cols]
        col_cut = True
    view = df.loc[rows, cols]

    header = [_clean_label(index_label or view.index.name, "index")]
    header += [_clean_label(c) for c in view.columns]
    if col_cut:
        header.append("...")
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    # 用 to_numpy(dtype=object) 而非 iterrows：后者会把整行强制成同一 dtype，
    # 导致整数列被渲染成 "1865.0000"。
    matrix = view.to_numpy(dtype="object")
    for pos, idx in enumerate(view.index):
        cells = [_clean_label(idx)]
        cells += [fmt(v, floatfmt) for v in matrix[pos].tolist()]
        if col_cut:
            cells.append("...")
        lines.append("| " + " | ".join(cells) + " |")
    if row_cut:
        lines.append("| ... |" + " ... |" * (len(header) - 1))
    return "\n".join(lines)


def series_to_md(s: pd.Series, name: str = "value", index_label: str = "",
                 floatfmt: str = ".4f", max_rows: Optional[int] = None) -> str:
    """把 Series 渲染成两列（index / value）Markdown 表格。"""
    if s is None or len(s) == 0:
        return "_（无数据）_"
    df = s.to_frame(name=name)
    return frame_to_md(df, floatfmt=floatfmt, index_label=index_label, max_rows=max_rows)


def dict_to_md(d: Mapping[str, Any], key_label: str = "指标",
               value_label: str = "数值", floatfmt: str = ".4f") -> str:
    """把 ``{键: 标量}`` 字典渲染成两列 Markdown 表格（保持插入顺序）。"""
    if not d:
        return "_（无数据）_"
    lines = [f"| {key_label} | {value_label} |", "|---|---|"]
    for k, v in d.items():
        lines.append(f"| {k} | {fmt(v, floatfmt)} |")
    return "\n".join(lines)


def bullets(items: Iterable[str]) -> str:
    """把字符串序列渲染成 Markdown 无序列表。"""
    return "\n".join(f"- {it}" for it in items)


def code_block(text: str, lang: str = "text") -> str:
    """渲染 Markdown 代码块。"""
    return f"```{lang}\n{text}\n```"


# ---------------------------------------------------------------------------
# JSON 序列化
# ---------------------------------------------------------------------------
def _scalar(value: Any) -> Any:
    """把单个值转成 JSON 原生类型；非有限浮点转 None。"""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return str(value)


def to_jsonable(obj: Any, max_rows: int = 4000) -> Any:
    """递归地把任意结果对象转成 JSON 可序列化结构。

    支持 dict / list / tuple / set / DataFrame / Series / Index / ndarray /
    dataclass / numpy 标量 / 枚举 / pathlib 路径；其它对象退化为 ``str()``。
    DataFrame 转 ``{列名: [值...]}`` 并保留索引（``index`` 键），超过
    ``max_rows`` 行时只保留最后 ``max_rows`` 行并注明被截断。
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return _scalar(obj)
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return _scalar(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_rows) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, max_rows) for v in obj]
    if isinstance(obj, pd.DataFrame):
        df = obj
        truncated = len(df) > max_rows
        if truncated:
            df = df.iloc[-max_rows:]
        out: Dict[str, Any] = {
            "index": [_scalar(i) for i in df.index],
            "columns": [str(c) for c in df.columns],
            "data": {str(c): [_scalar(v) for v in df[c].tolist()] for c in df.columns},
        }
        if truncated:
            out["truncated_to_last_rows"] = int(max_rows)
        return out
    if isinstance(obj, pd.Series):
        return {"index": [_scalar(i) for i in obj.index],
                "name": _scalar(obj.name),
                "data": [_scalar(v) for v in obj.tolist()]}
    if isinstance(obj, (pd.Index,)):
        return [_scalar(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return [_scalar(v) for v in np.asarray(obj).ravel().tolist()]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name), max_rows)
                for f in dataclasses.fields(obj)}
    if hasattr(obj, "value") and obj.__class__.__module__.startswith("enum"):
        return _scalar(obj.value)
    if hasattr(obj, "__fspath__"):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def _as_path(path: PathLike) -> str:
    return os.fspath(path) if not isinstance(path, str) else path


def ensure_dir(path: PathLike) -> str:
    """确保目录存在并返回其路径字符串。"""
    p = _as_path(path)
    os.makedirs(p, exist_ok=True)
    return p


def write_text(path: PathLike, text: str) -> str:
    """写文本文件（自动建父目录），返回绝对路径。"""
    p = os.path.abspath(_as_path(path))
    ensure_dir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


def write_json(path: PathLike, obj: Any, indent: int = 2) -> str:
    """把对象序列化为**严格合法** JSON 并写盘，返回绝对路径。"""
    payload = to_jsonable(obj)
    text = json.dumps(payload, ensure_ascii=False, indent=indent, allow_nan=False)
    return write_text(path, text + "\n")


def write_csv(df: pd.DataFrame, path: PathLike) -> str:
    """把 DataFrame 写成 CSV（UTF-8），返回绝对路径。"""
    p = os.path.abspath(_as_path(path))
    ensure_dir(os.path.dirname(p))
    df.to_csv(p, encoding="utf-8")
    return p


def write_report(path: PathLike, title: str,
                 sections: Sequence[Tuple[str, str]],
                 preamble: Optional[str] = None) -> str:
    """按 ``(小节标题, 正文 Markdown)`` 序列拼装一份报告并写盘。

    小节标题为空字符串时正文原样追加（用于开头的元信息段）。
    """
    parts: List[str] = [f"# {title}", ""]
    if preamble:
        parts += [preamble.strip(), ""]
    for heading, body in sections:
        if heading:
            parts += [f"## {heading}", ""]
        parts += [body.strip(), ""]
    return write_text(path, "\n".join(parts).rstrip() + "\n")
