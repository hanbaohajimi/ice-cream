from __future__ import annotations

from pathlib import Path

from rclpy.node import Node


def read_repo_config() -> dict:
    """从仓库根 config.yaml 读取全部配置，找不到则返回 {}。"""
    try:
        import yaml
        here = Path(__file__).resolve().parent
        for _ in range(8):
            candidate = here / "config.yaml"
            if candidate.exists():
                return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            here = here.parent
    except Exception:
        pass
    return {}


def declare_and_get_bool(node: Node, name: str, default: bool) -> bool:
    node.declare_parameter(name, default)
    return node.get_parameter(name).get_parameter_value().bool_value


def declare_and_get_int(node: Node, name: str, default: int) -> int:
    node.declare_parameter(name, default)
    return int(node.get_parameter(name).get_parameter_value().integer_value)


def declare_and_get_float(node: Node, name: str, default: float) -> float:
    node.declare_parameter(name, default)
    return float(node.get_parameter(name).get_parameter_value().double_value)


def declare_and_get_str(node: Node, name: str, default: str) -> str:
    node.declare_parameter(name, default)
    return node.get_parameter(name).get_parameter_value().string_value
