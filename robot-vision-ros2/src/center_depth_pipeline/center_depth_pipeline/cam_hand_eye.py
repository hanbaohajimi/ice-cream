"""eye-in-hand 外参 T_cam→ee（相机系 → 末端执行器系）。

与 handeye / object_base_logger 链式一致::
  link5_hmat = T_ee2base
  p_base = T_ee2base @ T_cam2ee @ p_cam

更换标定：修改仓库根目录 config.yaml 中的 hand_eye.T_cam2ee 矩阵即可，无需改代码。
若 config.yaml 不存在，则使用此文件内的硬编码备用值。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_T_from_config() -> np.ndarray | None:
    """从仓库根 config.yaml 读取 hand_eye.T_cam2ee；找不到或格式不对则返回 None。"""
    try:
        import yaml
        here = Path(__file__).resolve().parent
        for _ in range(6):
            candidate = here / "config.yaml"
            if candidate.exists():
                with open(candidate, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                rows = cfg.get("hand_eye", {}).get("T_cam2ee")
                if rows and len(rows) == 4 and all(len(r) == 4 for r in rows):
                    return np.array(rows, dtype=np.float64)
                return None
            here = here.parent
    except Exception:
        pass
    return None


# Park 方法标定备用值（相机 → 末端）；优先使用 config.yaml 中的矩阵
_T_FALLBACK = np.array(
    [
        [-0.05458395, -0.98267662, -0.177108,   0.05646747],
        [ 0.99841646, -0.05613061,  0.00373064, 0.00237631],
        [-0.01360719, -0.17662391,  0.98418435, 0.01534744],
        [ 0.0,         0.0,         0.0,         1.0       ],
    ],
    dtype=np.float64,
)

T_CAM2EE: np.ndarray = _load_T_from_config() if _load_T_from_config() is not None else _T_FALLBACK


def get_T_cam2ee() -> np.ndarray:
    """返回 T_cam→ee 的 4×4 副本。"""
    return T_CAM2EE.copy()
