#!/usr/bin/env python3
"""Render a plaintext CLI transcript as a dark-terminal PNG (paper screenshot)."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (13, 17, 23)
CHROME = (33, 38, 45)
FG = (201, 209, 217)
DIM = (110, 118, 129)
GREEN = (63, 185, 80)
CYAN = (88, 166, 255)
ORANGE = (210, 153, 34)
RED = (248, 81, 73)
YELLOW = (227, 179, 65)
DOT_R, DOT_Y, DOT_G = (255, 95, 86), (255, 189, 46), (39, 201, 63)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _line_color(line: str) -> tuple[int, int, int]:
    s = line.strip()
    if s.startswith("$") or s.endswith("$") or "@" in s[:40] and "$" in s:
        return GREEN
    if s.startswith("INFO:"):
        return CYAN
    if s.startswith("WARNING:") or s.startswith("ERROR:"):
        return ORANGE if s.startswith("WARNING:") else RED
    if s.startswith("---") or s.startswith("utc=") or s.startswith("url="):
        return DIM
    return FG


def render(text: str, out: Path, title: str, wrap: int = 92) -> None:
    font = _font(15)
    font_ui = _font(12)
    lines: list[str] = []
    for raw in text.rstrip("\n").splitlines() or [""]:
        if len(raw) <= wrap:
            lines.append(raw)
            continue
        rest = raw
        while len(rest) > wrap:
            lines.append(rest[:wrap])
            rest = rest[wrap:]
        lines.append(rest)

    line_h = 22
    pad_x, pad_y = 22, 18
    chrome_h = 38
    width = pad_x * 2 + wrap * 9
    height = chrome_h + pad_y * 2 + line_h * max(len(lines), 1) + 8

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, chrome_h), fill=CHROME)
    for i, color in enumerate((DOT_R, DOT_Y, DOT_G)):
        x = 18 + i * 16
        draw.ellipse((x, 13, x + 12, 25), fill=color)
    draw.text((70, 11), title, font=font_ui, fill=DIM)

    y = chrome_h + pad_y
    for line in lines:
        draw.text((pad_x, y), line.replace("\t", "    "), font=font, fill=_line_color(line))
        y += line_h

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("text_file")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--title", default="python -m evidence.analyze_client")
    p.add_argument("--wrap", type=int, default=92)
    args = p.parse_args()
    text = Path(args.text_file).read_text(encoding="utf-8")
    render(text, Path(args.output), args.title, wrap=args.wrap)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
