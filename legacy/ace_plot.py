"""
Tiny dependency-free PNG plotter for hackathon demos.

Why: environments may not have matplotlib installed. This module renders a very
simple 2-panel line plot (reward + accuracy) using only the Python stdlib.
"""

from __future__ import annotations

import struct
import zlib
from typing import Iterable


Color = tuple[int, int, int]


def plot_training_curves(
    episodes: list[dict],
    out_png: str,
    width: int = 900,
    height: int = 700,
) -> None:
    xs = [e["episode"] for e in episodes]
    rewards = [float(e["total_reward"]) for e in episodes]
    accs = [float(e["inference_accuracy"]) for e in episodes]

    img = _new_image(width, height, (255, 255, 255))

    pad = 40
    gap = 30
    panel_h = (height - 2 * pad - gap) // 2

    top = (pad, pad, width - pad, pad + panel_h)
    bot = (pad, pad + panel_h + gap, width - pad, pad + 2 * panel_h + gap)

    _draw_panel(img, width, top, xs, rewards, (30, 90, 200), y_min=min(rewards), y_max=max(rewards))
    _draw_panel(img, width, bot, xs, accs, (30, 160, 90), y_min=0.0, y_max=1.0)

    _write_png(out_png, img, width, height)


def plot_accuracy_curve(
    values: list[float],
    out_png: str,
    width: int = 900,
    height: int = 420,
) -> None:
    xs = list(range(1, len(values) + 1))
    img = _new_image(width, height, (255, 255, 255))

    pad = 40
    rect = (pad, pad, width - pad, height - pad)
    _draw_panel(img, width, rect, xs, values, (30, 160, 90), y_min=0.0, y_max=1.0)

    _write_png(out_png, img, width, height)


def _draw_panel(
    img: bytearray,
    width: int,
    rect: tuple[int, int, int, int],
    xs: list[int],
    ys: list[float],
    color: Color,
    *,
    y_min: float,
    y_max: float,
) -> None:
    x0, y0, x1, y1 = rect

    # Border
    _rect(img, width, x0, y0, x1, y1, (220, 220, 220))

    if not xs or len(xs) != len(ys):
        return

    if y_max <= y_min:
        y_max = y_min + 1.0

    # Padding inside the panel
    ix0, iy0, ix1, iy1 = x0 + 10, y0 + 10, x1 - 10, y1 - 10

    # Grid
    for i in range(1, 5):
        gy = iy0 + (iy1 - iy0) * i // 5
        _hline(img, width, ix0, ix1, gy, (245, 245, 245))

    n = len(xs)
    for i in range(n - 1):
        tx0 = i / (n - 1) if n > 1 else 0.0
        tx1 = (i + 1) / (n - 1) if n > 1 else 1.0

        px0 = int(ix0 + tx0 * (ix1 - ix0))
        px1 = int(ix0 + tx1 * (ix1 - ix0))

        py0 = _map_y(ys[i], y_min, y_max, iy0, iy1)
        py1 = _map_y(ys[i + 1], y_min, y_max, iy0, iy1)

        _line(img, width, px0, py0, px1, py1, color)


def _map_y(v: float, vmin: float, vmax: float, top: int, bottom: int) -> int:
    t = (v - vmin) / (vmax - vmin)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return int(bottom - t * (bottom - top))


def _new_image(width: int, height: int, bg: Color) -> bytearray:
    r, g, b = bg
    img = bytearray(width * height * 3)
    for i in range(0, len(img), 3):
        img[i] = r
        img[i + 1] = g
        img[i + 2] = b
    return img


def _set_px(img: bytearray, x: int, y: int, w: int, color: Color) -> None:
    if x < 0 or y < 0:
        return
    idx = (y * w + x) * 3
    if idx < 0 or idx + 2 >= len(img):
        return
    r, g, b = color
    img[idx] = r
    img[idx + 1] = g
    img[idx + 2] = b


def _hline(img: bytearray, w: int, x0: int, x1: int, y: int, color: Color) -> None:
    if x1 < x0:
        x0, x1 = x1, x0
    for x in range(x0, x1 + 1):
        _set_px(img, x, y, w, color)


def _vline(img: bytearray, w: int, x: int, y0: int, y1: int, color: Color) -> None:
    if y1 < y0:
        y0, y1 = y1, y0
    for y in range(y0, y1 + 1):
        _set_px(img, x, y, w, color)


def _rect(img: bytearray, w: int, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
    _hline(img, w, x0, x1, y0, color)
    _hline(img, w, x0, x1, y1, color)
    _vline(img, w, x0, y0, y1, color)
    _vline(img, w, x1, y0, y1, color)


def _line(img: bytearray, w: int, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
    # Bresenham
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _set_px(img, x0, y0, w, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _write_png(path: str, rgb: bytearray, width: int, height: int) -> None:
    # Add filter byte 0 for each scanline.
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        start = y * stride
        raw.extend(rgb[start : start + stride])

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=6)

    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", ihdr))
    png.extend(chunk(b"IDAT", idat))
    png.extend(chunk(b"IEND", b""))

    with open(path, "wb") as f:
        f.write(png)
