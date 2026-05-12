"""
sensor_msgs/Image 与 sensor_msgs/CompressedImage 和 numpy 之间的转换，不依赖 cv_bridge。

ROS Humble 的 cv_bridge Python 绑定基于 NumPy 1.x 构建；在 NumPy 2.x 下导入
会抛出 AttributeError: _ARRAY_API not found。
本模块仅使用 numpy + cv2（已是 YOLO 的依赖项）。

压缩话题命名（image_transport）：
  RGB   : <base>/image/compressed              — JPEG 或 PNG（CompressedImage）
  Depth : <base>/image_raw/compressedDepth     — 见下文

compressedDepth 编码格式（image_transport_plugins，ROS2 Humble）
─────────────────────────────────────────────────────────────────
插件在载荷之前写入 12 字节的 ConfigHeader：

  struct ConfigHeader {
      uint32_t format;        // 0 = PNG，1 = RVL
      float    depthParam[2]; // 仅对 PNG 量化有意义
  };

format = 0（PNG）：载荷 = PNG 编码的 16UC1 / 32FC1
format = 1（RVL）：载荷 = RVL 编码的 int16 流（见 _rvl_decompress）

旧版 image_transport 使用 4 字节 float 头 + PNG。
通过魔数字节自动检测两种格式。

RVL 参考：https://dl.acm.org/doi/10.1145/3084822.3084823
C 参考   ：ros-perception/image_transport_plugins, rvl_codec.h

性能
───────────
_rvl_decompress 使用首次导入时编译的 C 共享库（gcc -O3），
缓存在 ~/.cache/center_depth_pipeline/rvl_c.so。
对 640×480 帧，耗时约 0.2 ms，而纯 Python 约 300 ms。
若 gcc 不可用，则回退到优化的纯 Python 实现。
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import subprocess

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image

_log = logging.getLogger(__name__)

# ── C RVL 实现 ────────────────────────────────────────────────────────────────
# 首次导入时编译一次，缓存以避免重复编译。

_RVL_C_SRC = r"""
#include <stdint.h>
#include <string.h>

void rvl_decompress(const uint8_t *in, int16_t *out, int n)
{
    memset(out, 0, (size_t)n * 2);
    int pos = 0, idx = 0, nib;
    int32_t prev = 0;
    while (idx < n) {
        /* run of zeros */
        int zeros = 0;
        do {
            uint8_t b = in[pos >> 1];
            nib = (pos & 1) ? (b >> 4) : (b & 0xF);
            ++pos;
            zeros += (nib & 8) ? (nib & 7) : 7;
        } while (!(nib & 8));
        idx += zeros;
        if (idx >= n) break;
        /* run of nonzeros */
        do {
            uint8_t b = in[pos >> 1];
            nib = (pos & 1) ? (b >> 4) : (b & 0xF);
            ++pos;
            int v = nib & 7;
            int32_t cur = prev + ((v >> 1) ^ (-(v & 1)));
            if (cur < -32768) cur = -32768;
            if (cur >  32767) cur =  32767;
            out[idx++] = (int16_t)cur;
            prev = cur;
        } while ((nib & 8) && idx < n);
    }
}
"""

_RVL_CACHE = os.path.expanduser("~/.cache/center_depth_pipeline")
_RVL_SO    = os.path.join(_RVL_CACHE, "rvl_c.so")
_RVL_CSRC  = os.path.join(_RVL_CACHE, "rvl_c.c")


def _load_rvl_c_func():
    """编译（一次）并加载 C RVL 解码器。返回 ctypes 函数或 None。"""
    try:
        os.makedirs(_RVL_CACHE, exist_ok=True)
        # 仅在源文件发生变化时重新编译（比较 .c 与 .so 的修改时间）
        need_compile = (
            not os.path.exists(_RVL_SO)
            or not os.path.exists(_RVL_CSRC)
            or os.path.getmtime(_RVL_CSRC) >= os.path.getmtime(_RVL_SO)
        )
        if need_compile:
            with open(_RVL_CSRC, "w") as f:
                f.write(_RVL_C_SRC)
            # 先编译到临时文件，避免编译崩溃时留下损坏的 .so
            tmp_so = _RVL_SO + ".tmp"
            res = subprocess.run(
                ["gcc", "-O3", "-shared", "-fPIC", "-o", tmp_so, _RVL_CSRC],
                capture_output=True, timeout=30,
            )
            if res.returncode != 0:
                _log.warning(
                    "gcc RVL compile failed: %s — using slow Python fallback",
                    res.stderr.decode(errors="replace"),
                )
                return None
            os.replace(tmp_so, _RVL_SO)

        lib = ctypes.CDLL(_RVL_SO)
        fn  = lib.rvl_decompress
        fn.restype  = None
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
        ]
        _log.info("RVL C decoder loaded from %s", _RVL_SO)
        return fn
    except Exception as exc:
        _log.warning("Could not load C RVL decoder (%s) — using Python fallback", exc)
        return None


_rvl_c_func = _load_rvl_c_func()


def _u8_view(data) -> np.ndarray:
    """尽可能返回 uint8 零拷贝视图。"""
    try:
        return np.frombuffer(memoryview(data), dtype=np.uint8)
    except TypeError:
        return np.frombuffer(bytes(data), dtype=np.uint8)


# ── RVL 纯 Python 备选（已优化）──────────────────────────────────────────────

def _rvl_decompress_py(data: bytes, num_pixels: int) -> np.ndarray:
    """纯 Python RVL 备选实现。640×480 约 300 ms，仅在 gcc 不可用时使用。"""
    buf = np.frombuffer(data, dtype=np.uint8)
    # 将字节展开为 nibble 列表（numpy 向量化 + Python 列表快速索引）
    nib_arr = np.empty(len(buf) * 2, dtype=np.int32)
    nib_arr[0::2] = buf & 0x0F
    nib_arr[1::2] = buf >> 4
    nibs = nib_arr.tolist()
    max_nib = len(nibs)

    out = np.zeros(num_pixels, dtype=np.int16)
    pos = 0
    idx = 0
    prev = 0

    while idx < num_pixels and pos < max_nib:
        zeros = 0
        while True:
            n = nibs[pos]; pos += 1
            if n & 8:
                zeros += n & 7
                break
            zeros += 7
        idx += zeros
        if idx >= num_pixels:
            break
        while True:
            n = nibs[pos]; pos += 1
            v = n & 7
            cur = prev + ((v >> 1) ^ (-(v & 1)))
            if cur < -32768: cur = -32768
            elif cur > 32767: cur = 32767
            out[idx] = cur; prev = cur; idx += 1
            if not (n & 8) or idx >= num_pixels:
                break

    return out


def _rvl_decompress(data: bytes, num_pixels: int) -> np.ndarray:
    """将 RVL 字节流解压为 int16 数组（长度 num_pixels）。

    有 C 实现时使用编译版（约 0.2 ms），否则回退到纯 Python（约 300 ms）。
    """
    if _rvl_c_func is not None:
        in_buf  = np.frombuffer(data, dtype=np.uint8)
        out_buf = np.empty(num_pixels, dtype=np.int16)
        _rvl_c_func(
            in_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_int(num_pixels),
        )
        return out_buf
    return _rvl_decompress_py(data, num_pixels)


# ─────────────────────────── 原始 sensor_msgs/Image ──────────────────────────

def imgmsg_to_bgr8(msg: Image) -> np.ndarray:
    """将 RGB/BGR/mono8 的 sensor_msgs/Image 解码为连续 BGR uint8 数组（H,W,3）。"""
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    enc = msg.encoding
    buf = _u8_view(msg.data)
    need = step * h
    if buf.size < need:
        raise ValueError(f"Image buffer too small: {buf.size} < {need}")
    rows = buf[:need].reshape((h, step))

    if enc == "bgr8":
        return np.array(rows[:, : w * 3].reshape((h, w, 3)), copy=True, order="C")
    if enc == "rgb8":
        rgb = rows[:, : w * 3].reshape((h, w, 3))
        return np.array(rgb[:, :, ::-1], copy=True, order="C")
    if enc == "mono8":
        g = rows[:, :w]
        return np.array(np.stack([g, g, g], axis=-1), copy=True, order="C")
    raise ValueError(f"Unsupported RGB encoding '{enc}' (bgr8/rgb8/mono8 supported)")


def imgmsg_to_depth_np(msg: Image) -> tuple[np.ndarray, str]:
    """将 sensor_msgs/Image 深度图解码为（numpy 数组，编码字符串）。"""
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    enc = msg.encoding

    if enc == "16UC1":
        row_u16 = step // 2
        buf = np.frombuffer(_u8_view(msg.data), dtype=np.uint16)
        rows = buf[: row_u16 * h].reshape((h, row_u16))
        return rows[:, :w].copy(), enc

    if enc == "32FC1":
        row_f = step // 4
        buf = np.frombuffer(_u8_view(msg.data), dtype=np.float32)
        rows = buf[: row_f * h].reshape((h, row_f))
        return rows[:, :w].copy(), enc

    raise ValueError(f"Unsupported depth encoding '{enc}'")


def bgr8_to_imgmsg(frame: np.ndarray, header) -> Image:
    """将 BGR uint8（H,W,3）编码为 sensor_msgs/Image。"""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be uint8 HxWx3 BGR")
    h, w = frame.shape[:2]
    msg = Image()
    msg.header = header
    msg.height, msg.width = h, w
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = frame.tobytes()
    return msg


# ─────────────── sensor_msgs/CompressedImage（RGB，JPEG/PNG）────────────────

def compressed_imgmsg_to_bgr8(msg: CompressedImage) -> np.ndarray:
    """将 CompressedImage（JPEG 或 PNG）解码为 BGR uint8（H,W,3）。

    尽可能使用缓冲区视图（memoryview），避免每帧将完整 JPEG
    载荷拷贝到新的 ``bytes`` 对象。
    """
    buf = _u8_view(msg.data)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(
            f"cv2.imdecode failed (format='{msg.format}', "
            f"data_len={len(msg.data)})"
        )
    return np.ascontiguousarray(frame)


# ─────────────── sensor_msgs/CompressedImage（compressedDepth）──────────────

_CFG_HDR_SIZE = 12          # uint32 format + float depthParam[2]
_CFG_FMT_PNG  = 0
_CFG_FMT_RVL  = 1
_LEGACY_HDR   = 4
_PNG_MAGIC    = b'\x89PNG'


def compressed_depth_to_np(
    msg: CompressedImage,
    width: int = 0,
    height: int = 0,
) -> tuple[np.ndarray, str]:
    """将 compressedDepth 类型的 CompressedImage 解码为（uint16 H×W 数组，'16UC1'）。

    通过 ConfigHeader 或 PNG 魔数字节自动检测编码格式（PNG 或 RVL）。
    RVL 解码需要 `width` 和 `height`（从 CameraInfo 获取）。
    """
    raw = memoryview(msg.data)
    n   = len(raw)
    if n < 4:
        raise ValueError(f"compressedDepth data too short ({n} bytes)")

    cfg_format = struct.unpack_from("<I", raw, 0)[0]

    if cfg_format == _CFG_FMT_RVL:
        if width <= 0 or height <= 0:
            raise ValueError(
                "RVL-encoded compressedDepth requires image dimensions. "
                "Ensure CameraInfo has been received before depth frames."
            )
        rvl_payload = raw[4:]
        flat  = _rvl_decompress(rvl_payload, width * height)
        depth = flat.view(np.uint16).reshape((height, width))
        return np.ascontiguousarray(depth), "16UC1"

    if cfg_format == _CFG_FMT_PNG:
        png_data = raw[_CFG_HDR_SIZE:]
    elif n > 8 and raw[4:8] == _PNG_MAGIC:
        png_data = raw[_LEGACY_HDR:]
    elif n > 4 and raw[0:4] == _PNG_MAGIC:
        png_data = raw
    else:
        if width > 0 and height > 0:
            try:
                flat  = _rvl_decompress(raw, width * height)
                depth = flat.view(np.uint16).reshape((height, width))
                return np.ascontiguousarray(depth), "16UC1"
            except Exception:
                pass
        head = bytes(raw[:8])
        raise ValueError(
            f"Cannot decode compressedDepth: format='{msg.format}', "
            f"data[0:8]={head.hex()}, data_len={n}. "
            "If the camera uses RVL, make sure CameraInfo arrives before depth."
        )

    buf   = np.frombuffer(png_data, dtype=np.uint8)
    depth = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(
            f"cv2.imdecode failed for PNG depth (format='{msg.format}', "
            f"png_offset={n - len(png_data)}, data_len={n})"
        )
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.dtype == np.float32:
        depth = (depth * 1000.0).astype(np.uint16)
    else:
        depth = depth.astype(np.uint16)
    return np.ascontiguousarray(depth), "16UC1"
