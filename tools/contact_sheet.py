"""Build labelled contact sheets so a human (or Claude) can hand-label ground truth
without paying to look at 46 full-resolution phone photos one at a time.

Usage: python tools/contact_sheet.py
Writes out/contact/sheet_NN.jpg
"""
from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw

SRC = pathlib.Path("data/images")
DST = pathlib.Path("out/contact")
CELL_W, CELL_H = 460, 620
COLS, ROWS = 4, 3
LABEL_H = 34


def cell(path: pathlib.Path) -> Image.Image:
    canvas = Image.new("RGB", (CELL_W, CELL_H + LABEL_H), "white")
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((CELL_W - 8, CELL_H - 8))
            canvas.paste(im, ((CELL_W - im.width) // 2, (CELL_H - im.height) // 2))
    except Exception as exc:  # corrupt / non-image
        ImageDraw.Draw(canvas).text((12, CELL_H // 2), f"UNREADABLE\n{exc}", fill="red")
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, CELL_H, CELL_W, CELL_H + LABEL_H], fill="black")
    d.text((10, CELL_H + 10), path.stem, fill="white")
    return canvas


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    paths = sorted(SRC.glob("optera_doc_*.jpg"))
    per = COLS * ROWS
    for sheet_i in range(0, len(paths), per):
        chunk = paths[sheet_i : sheet_i + per]
        sheet = Image.new("RGB", (COLS * CELL_W, ROWS * (CELL_H + LABEL_H)), "white")
        for i, p in enumerate(chunk):
            sheet.paste(cell(p), ((i % COLS) * CELL_W, (i // COLS) * (CELL_H + LABEL_H)))
        out = DST / f"sheet_{sheet_i // per + 1:02d}.jpg"
        sheet.save(out, quality=88)
        print(out)


if __name__ == "__main__":
    main()
