"""One-off: generate PWA icons for the frontend (see 2026-07-10 PWA spec).

Draws the MusiX glyph — five rounded equalizer bars fading violet→amber on the
Studio Console dark background — and writes the icon set into frontend/public/.
Replace with real brand art later by overwriting the same files.

Usage: python -m scripts.gen_pwa_icons   (requires pillow)
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).parent.parent / "frontend" / "public"

BG = (13, 13, 16)          # colors.bg dark (#0d0d10)
BG_GLOW = (30, 26, 48)     # faint violet radial center
BARS = [                   # violet → amber sweep, matches accent/amber tokens
    (122, 111, 240),
    (140, 122, 242),
    (163, 136, 232),
    (196, 156, 170),
    (212, 165, 90),
]
# Bar heights as a fraction of glyph height (waveform silhouette).
HEIGHTS = [0.45, 0.78, 1.0, 0.62, 0.36]


def draw_icon(size: int, glyph_frac: float) -> Image.Image:
    """glyph_frac — glyph box as a fraction of the canvas (maskable needs ≤0.66)."""
    ss = 4  # supersample for clean edges
    s = size * ss
    img = Image.new("RGB", (s, s), BG)
    d = ImageDraw.Draw(img)

    # Soft radial glow behind the glyph.
    for r in range(s // 2, 0, -ss):
        t = r / (s / 2)
        col = tuple(int(g + (b - g) * t) for g, b in zip(BG_GLOW, BG))
        d.ellipse([s / 2 - r, s / 2 - r, s / 2 + r, s / 2 + r], fill=col)

    glyph = s * glyph_frac
    n = len(BARS)
    bar_w = glyph / (n * 1.7)
    gap = (glyph - n * bar_w) / (n - 1)
    x0 = (s - glyph) / 2
    for i, (col, hf) in enumerate(zip(BARS, HEIGHTS)):
        h = glyph * hf
        x = x0 + i * (bar_w + gap)
        y = (s - h) / 2
        d.rounded_rectangle([x, y, x + bar_w, y + h], radius=bar_w / 2, fill=col)

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draw_icon(192, 0.62).save(OUT / "pwa-192x192.png")
    draw_icon(512, 0.62).save(OUT / "pwa-512x512.png")
    # Maskable: glyph inside the 66% safe zone, background full-bleed.
    draw_icon(512, 0.46).save(OUT / "pwa-512x512-maskable.png")
    draw_icon(180, 0.62).save(OUT / "apple-touch-icon.png")
    print(f"icons written to {OUT}")


if __name__ == "__main__":
    main()
