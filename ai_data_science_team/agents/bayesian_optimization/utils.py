from typing import Any, Dict


def validate_bounds(min_val: float, max_val: float) -> bool:
    """验证最小值不大于最大值"""
    return min_val <= max_val


def serialize_optimizer_state(opt: Any) -> Dict[str, Any]:
    """将优化器状态序列化为可存储结构（剔除不可序列化对象，如 GP 实例）。"""

    def _tolist(x):
        return x.tolist() if hasattr(x, "tolist") else x

    return {
        "X": _tolist(getattr(opt, "X", [])),
        "y": _tolist(getattr(opt, "y", [])),
        "bounds": _tolist(getattr(opt, "bounds", [])),
        "n_initial_points": int(getattr(opt, "n_initial_points", 5)),
    }

