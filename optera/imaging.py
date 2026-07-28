"""Image preprocessing, integrity repair, hashing, and image-token accounting.

Everything in this module is free — no API calls — which is exactly why it is
the first stage of the pipeline. A document rejected or deduplicated here costs
nothing at all.
"""
from __future__ import annotations

import base64
import hashlib
import io
import pathlib
from dataclasses import dataclass

from PIL import Image, ImageFile

from .config import GEMINI_MODELS, PRICING

# Gemini bills vision in fixed tiles rather than by area: an image with both
# sides <= 384px is a flat 258 tokens, and anything larger is cropped into
# 768x768 tiles at 258 tokens each.
GEMINI_TILE_PX = 768
GEMINI_SMALL_PX = 384
GEMINI_TOKENS_PER_TILE = 258

# Some WhatsApp images arrive with the last few bytes missing. Pillow raises on
# these by default; the pixels that are present are still perfectly usable, so
# we decode what we can rather than throwing away a real invoice.
ImageFile.LOAD_TRUNCATED_IMAGES = True

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
WEBP_MAGIC = b"RIFF"


@dataclass
class Prepared:
    doc_id: str
    ok: bool
    reason: str | None
    b64: str | None
    media_type: str
    width: int
    height: int
    content_hash: str
    phash: str | None
    was_repaired: bool
    source_bytes: int


def sniff(raw: bytes) -> str | None:
    """Return a media type from magic bytes, or None if this is not an image.

    Filename extensions lie. A failed download saved as .jpg is an HTML error
    page, and sending it to a multimodal model costs money to be told so.
    """
    if raw.startswith(JPEG_MAGIC):
        return "image/jpeg"
    if raw.startswith(PNG_MAGIC):
        return "image/png"
    if raw.startswith(WEBP_MAGIC) and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def dhash(im: Image.Image, size: int = 8) -> str:
    """64-bit difference hash — cheap near-duplicate detection.

    The same page photographed twice, or the same file re-sent through WhatsApp
    at a different compression level, lands on the same or an adjacent hash.
    """
    small = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = small.tobytes()  # mode "L" -> one byte per pixel, row-major
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | int(left > right)
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _resize(im: Image.Image, max_edge: int) -> Image.Image:
    if max(im.size) <= max_edge:
        return im
    scale = max_edge / max(im.size)
    return im.resize(
        (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
        Image.Resampling.LANCZOS,
    )


def prepare(path: str | pathlib.Path, max_edge: int, quality: int = 78) -> Prepared:
    """Validate, repair, downscale, and encode one image for a model call."""
    path = pathlib.Path(path)
    doc_id = path.stem
    raw = path.read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()[:16]

    media_type = sniff(raw)
    if media_type is None:
        head = raw[:64].decode("utf-8", "replace").strip().replace("\n", " ")
        return Prepared(
            doc_id, False,
            f"not an image file (magic bytes say {raw[:4]!r}; starts with {head[:48]!r})",
            None, "", 0, 0, content_hash, None, False, len(raw),
        )

    was_repaired = False
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()  # forces full decode; truncation surfaces here
            im = im.convert("RGB")
            # A truncated file that still decoded means we recovered it.
            was_repaired = _looks_truncated(raw, media_type)
            phash = dhash(im)
            orig_w, orig_h = im.size
            small = _resize(im, max_edge)
            buf = io.BytesIO()
            small.save(buf, format="JPEG", quality=quality, optimize=True)
    except Exception as exc:
        return Prepared(
            doc_id, False, f"undecodable image: {exc}", None, "", 0, 0,
            content_hash, None, False, len(raw),
        )

    return Prepared(
        doc_id, True, None,
        base64.standard_b64encode(buf.getvalue()).decode(),
        "image/jpeg", small.width, small.height,
        content_hash, phash, was_repaired, len(raw),
    )


def _looks_truncated(raw: bytes, media_type: str) -> bool:
    if media_type == "image/jpeg":
        return not raw.rstrip().endswith(b"\xff\xd9")
    return False


# --------------------------------------------------------------------------
# Image-token accounting
# --------------------------------------------------------------------------


def image_tokens(width: int, height: int, model: str) -> int:
    """Estimate billed image tokens for a (width, height) on a given model.

    Anthropic charges roughly (width x height) / 750 tokens, after server-side
    downscaling to the model's vision tier. Both the divisor and the per-tier
    ceiling come from config.PRICING, so this tracks the published limits.

    On Anthropic models tokens scale with *area*, so halving the long edge
    quarters the image cost — up to a per-tier ceiling that caps the saving.
    Gemini instead bills in fixed 768px tiles, which makes resolution a
    step function rather than a smooth curve: dropping below 384px on both
    sides is a large win, and shrinking within a tile band buys nothing.
    """
    if model in GEMINI_MODELS:
        return _gemini_image_tokens(width, height)
    price = PRICING[model]
    w, h = float(width), float(height)
    longest = max(w, h)
    if longest > price.max_image_edge_px:
        scale = price.max_image_edge_px / longest
        w, h = w * scale, h * scale
    return min(int((w * h) / 750), price.max_image_tokens)


def _gemini_image_tokens(width: int, height: int) -> int:
    if width <= GEMINI_SMALL_PX and height <= GEMINI_SMALL_PX:
        return GEMINI_TOKENS_PER_TILE
    cols = max(1, -(-width // GEMINI_TILE_PX))  # ceil division
    rows = max(1, -(-height // GEMINI_TILE_PX))
    return cols * rows * GEMINI_TOKENS_PER_TILE


def estimate_text_tokens(text: str) -> int:
    """Offline text-token estimate (~3.6 chars/token for English + JSON).

    Only used when running without an API key. In live mode the ledger records
    the exact `usage` block the API returns, so these estimates never leak into
    a reported number that claims to be measured.
    """
    return max(1, round(len(text) / 3.6))
