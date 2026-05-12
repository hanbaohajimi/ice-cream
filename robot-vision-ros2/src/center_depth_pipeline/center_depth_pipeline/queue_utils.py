from __future__ import annotations

import queue
from typing import Any


def replace_latest(q: queue.Queue, item: Any) -> None:
    """丢弃最多一条过期数据，并将最新条目入队。"""
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass
