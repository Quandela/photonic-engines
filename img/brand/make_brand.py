"""Draw the photonic-engines mark, light and dark: SVG + PNG (512, 128, 32) + README banner.

Run from the repository root: ``venv/bin/python img/brand/make_brand.py``. Needs ``rsvg-convert`` (librsvg) on the
PATH for the PNGs. Geometry is on a 64 x 64 grid; strokes are sized to stay readable at 32 px (1 unit = 0.5 px).

The mark is the emitter-source glyph of the frontier paper's figures: a striped source with the spin (dot + arrow)
at its output face, emitting photons.
"""

import subprocess
from pathlib import Path

from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as Path_
from matplotlib.textpath import TextPath, TextToPath

OUT = Path(__file__).resolve().parent

THEMES = {  # tile and wordmark, then the glyph's own colours
    "light": {"bg": "#ffffff", "ink": "#1f1f1f", "accent": "#2a78d6",
              "src": "#8a8f98", "src_fill": "#d9dce1", "spin": "#e0620d", "ph": "#d62728", "wire": "#4a4f58"},
    "dark": {"bg": "#10151c", "ink": "#e8ebf0", "accent": "#5b9cf0",
             "src": "#a3a9b2", "src_fill": "#3a414b", "spin": "#f07a2a", "ph": "#ef5350", "wire": "#b9bfc8"},
}
SIZES = (512, 128, 32)


def _f(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


def spin(t):
    """The spin-source glyph: striped source box with the spin (dot + arrow) at its output face, emitting photons."""
    parts = [f'<rect x="5" y="13" width="35" height="38" rx="4" fill="{t["src_fill"]}" stroke="{t["src"]}" '
             f'stroke-width="3"/>',
             f'<path d="M12 19V45M18 19V45" stroke="{t["src"]}" stroke-width="2.8"/>',
             f'<path d="M40 32H59" stroke="{t["wire"]}" stroke-width="3"/>',
             f'<circle cx="30" cy="35" r="4.6" fill="{t["spin"]}"/>',
             f'<path d="M30 45V26" stroke="{t["spin"]}" stroke-width="3.2"/>',
             f'<path d="M24.5 27.5L30 18.5L35.5 27.5Z" fill="{t["spin"]}" stroke="{t["spin"]}" stroke-width="1.2" '
             f'stroke-linejoin="round"/>',
             f'<circle cx="48" cy="32" r="4.4" fill="{t["ph"]}"/>',
             f'<circle cx="57.5" cy="32" r="3.2" fill="{t["ph"]}"/>']
    return "\n".join(parts)


def tile_svg(body, t):
    """Square icon: the mark on a rounded tile (avatar, favicon, PyPI)."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="512" height="512">\n'
            f'<rect width="64" height="64" rx="13" fill="{t["bg"]}"/>\n'
            f'<g fill="none" stroke-linecap="round">\n{body}\n</g>\n</svg>\n')


def _font():
    """Inter Medium (SIL Open Font License); DejaVu Sans if Inter is not installed."""
    for family in ("Inter", "DejaVu Sans"):
        prop = FontProperties(family=family, weight="medium")
        try:
            font_manager.findfont(prop, fallback_to_default=False)
            return prop
        except ValueError:
            continue
    return FontProperties()


def _outline(text, prop, x, size):
    """SVG path data of ``text`` set at baseline (x, 0), em = ``size`` units, y pointing down."""
    p = TextPath((0, 0), text, size=size, prop=prop)
    d, i, verts, codes = [], 0, p.vertices, p.codes
    while i < len(codes):
        c = codes[i]
        if c == Path_.MOVETO:
            d.append(f"M{_f(x + verts[i][0])} {_f(-verts[i][1])}")
            i += 1
        elif c == Path_.LINETO:
            d.append(f"L{_f(x + verts[i][0])} {_f(-verts[i][1])}")
            i += 1
        elif c == Path_.CURVE3:
            d.append("Q" + " ".join(f"{_f(x + v[0])} {_f(-v[1])}" for v in verts[i:i + 2]))
            i += 2
        elif c == Path_.CURVE4:
            d.append("C" + " ".join(f"{_f(x + v[0])} {_f(-v[1])}" for v in verts[i:i + 3]))
            i += 3
        else:  # CLOSEPOLY
            d.append("Z")
            i += 1
    return "".join(d)


def wordmark(t, size=25.5):
    """'photonic-engines' as outlines (no font needed to view the SVG); the hyphen in the spin's orange."""
    prop = _font()
    ttp = TextToPath()
    adv = [ttp.get_text_width_height_descent(s, prop, ismath=False)[0] * size / prop.get_size_in_points()
           for s in ("photonic", "photonic-")]
    return (f'<path d="{_outline("photonic", prop, 0, size)}" fill="{t["ink"]}"/>\n'
            f'<path d="{_outline("-", prop, adv[0], size)}" fill="{t["spin"]}"/>\n'
            f'<path d="{_outline("engines", prop, adv[1], size)}" fill="{t["ink"]}"/>')


def banner_svg(body, t):
    """1280 x 320 README banner: the mark at left, the name in outlined Inter Medium (DejaVu Sans if absent)."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 80" width="1280" height="320">\n'
            f'<rect width="320" height="80" fill="{t["bg"]}"/>\n'
            f'<g transform="translate(12 12) scale(0.875)" fill="none" stroke-linecap="round">\n{body}\n</g>\n'
            f'<g transform="translate(82 50)">\n{wordmark(t)}\n</g>\n</svg>\n')


def render(svg_path, png_path, size):
    """Rasterise an SVG to a PNG ``size`` px wide."""
    subprocess.run(["rsvg-convert", "-w", str(size), "-o", str(png_path), str(svg_path)], check=True)


def main():
    """Write each theme: tile SVG, its PNGs, banner SVG + PNG."""
    for theme, t in THEMES.items():
        stem = f"spin_{theme}"
        svg = OUT / f"{stem}.svg"
        svg.write_text(tile_svg(spin(t), t))
        for s in SIZES:
            render(svg, OUT / f"{stem}_{s}.png", s)
        bsvg = OUT / f"{stem}_banner.svg"
        bsvg.write_text(banner_svg(spin(t), t))
        render(bsvg, OUT / f"{stem}_banner.png", 1280)


if __name__ == "__main__":
    main()
