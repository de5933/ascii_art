"""Measure how much ink each character puts down in a font, and save it as a config.

    python make_font_config.py "Cascadia Mono"
    python make_font_config.py "Consolas" --weight 700
    python make_font_config.py --list

Each glyph is rendered on its own into a cell the size of the font's advance
width by its ascent-plus-descent, and its coverage is the mean ink across that
cell. The result is written to glyph-configs/<font-slug>.json.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import glyph_config
from glyph_config import CHARSET, ConfigError

FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
    Path.home() / ".fonts",
    Path("/usr/share/fonts"),
    Path("/Library/Fonts"),
]

FONT_SUFFIXES = {".ttf", ".otf", ".ttc", ".otc"}

DEFAULT_SIZE = 64

UPRIGHT_STYLES = {"regular", "roman", "book", "medium"}


def iter_font_files():
    seen = set()
    for directory in FONT_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            key = str(path).lower()
            if path.suffix.lower() in FONT_SUFFIXES and key not in seen:
                seen.add(key)
                yield path


def open_font(path, index, size):
    return ImageFont.truetype(str(path), size, index=index)


def describe(path, index=0):
    """(family, style) for a font file, or None if it cannot be read."""
    try:
        return open_font(path, index, 12).getname()
    except Exception:
        return None


def find_font(name):
    """Resolve a family name (or a path) to (path, index, family, style)."""
    direct = Path(name)
    if direct.is_file():
        described = describe(direct)
        if not described:
            raise ConfigError(f"cannot read font file {direct}")
        return direct, 0, described[0], described[1]

    wanted = name.strip().lower()
    fallback = None

    for path in iter_font_files():
        collection = path.suffix.lower() in {".ttc", ".otc"}
        for index in range(8 if collection else 1):
            described = describe(path, index)
            if not described:
                break
            family, style = described
            if family.lower() != wanted:
                continue
            # Prefer an upright regular face over italics and extra weights.
            if style.lower() in UPRIGHT_STYLES:
                return path, index, family, style
            fallback = fallback or (path, index, family, style)

    if fallback:
        return fallback
    raise ConfigError(f"no installed font named {name!r} (try --list)")


def apply_weight(font, weight):
    """Pin the Weight axis of a variable font."""
    if weight is None:
        return

    try:
        axes = font.get_variation_axes()
    except OSError:
        raise ConfigError("this font is not variable, so --weight does not apply")

    settings, matched = [], False
    for axis in axes:
        label = axis.get("name", b"")
        label = label.decode("utf-8", "ignore") if isinstance(label, bytes) else str(label)
        if label.lower() == "weight":
            settings.append(max(axis["minimum"], min(axis["maximum"], weight)))
            matched = True
        else:
            settings.append(axis["default"])

    if not matched:
        raise ConfigError("this font has no Weight axis")

    font.set_variation_by_axes(settings)


def measure(path, index, size, weight, chars=CHARSET):
    """Render each character alone in its cell and return {char: ink coverage}."""
    font = open_font(path, index, size)
    apply_weight(font, weight)

    advances = {font.getlength(char) for char in chars}
    cell_width = math.ceil(max(advances))
    ascent, descent = font.getmetrics()
    cell_height = ascent + descent

    if cell_width < 1 or cell_height < 1:
        raise ConfigError("font reported a degenerate cell size")

    coverage = {}
    for char in chars:
        cell = Image.new("L", (cell_width, cell_height), 0)
        ImageDraw.Draw(cell).text((0, 0), char, fill=255, font=font)
        coverage[char] = float(np.asarray(cell, dtype=np.float64).mean() / 255.0)

    return coverage, cell_width, cell_height, len(advances) == 1


def measure_strip(path, chars=CHARSET):
    """Measure a rendered strip image of equally spaced characters."""
    img = Image.open(path).convert("L")
    grey = np.asarray(img, dtype=np.float64) / 255.0
    background, ink = grey.min(), grey.max()
    if ink <= background:
        raise ConfigError(f"{path} has no contrast to measure")

    edges = [round(i * img.width / len(chars)) for i in range(len(chars) + 1)]
    coverage = {
        char: float((grey[:, edges[i]:edges[i + 1]].mean() - background) / (ink - background))
        for i, char in enumerate(chars)
    }
    return coverage, round(img.width / len(chars)), img.height


def list_fonts(monospace_only=True):
    rows = []
    for path in iter_font_files():
        described = describe(path)
        if not described:
            continue
        if monospace_only:
            try:
                font = open_font(path, 0, 32)
                if len({font.getlength(c) for c in "iMW1 ."}) != 1:
                    continue
            except Exception:
                continue
        rows.append((described[0], described[1], path.name))
    return sorted(set(rows))


def generate(name, size=DEFAULT_SIZE, weight=None, directory=None, strip=None, quiet=False):
    """Measure a font (or strip) and write its config. Returns the path written."""
    if strip:
        coverage, cell_width, cell_height = measure_strip(strip)
        source, family, style, uniform = Path(strip), name, "strip", True
        method, size = "strip-sample", None
    else:
        source, index, family, style = find_font(name)
        coverage, cell_width, cell_height, uniform = measure(source, index, size, weight)
        method = "rendered-ink-coverage"

    if not uniform and not quiet:
        print(f"warning: {family} is not monospaced; cells use its widest advance", file=sys.stderr)

    config = glyph_config.build(
        family=family,
        style=style,
        file=source,
        weight=weight,
        size=size,
        cell_width=cell_width,
        cell_height=cell_height,
        coverage=coverage,
        method=method,
    )
    return glyph_config.save(config, glyph_config.config_path(family, weight, directory))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("font", nargs="?", help="font family name, or a path to a font file")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="render size in px")
    parser.add_argument("--weight", type=int, help="weight axis value for variable fonts")
    parser.add_argument("--from-strip", metavar="IMAGE", help="measure a character strip instead")
    parser.add_argument("--dir", help="write the config somewhere other than glyph-configs/")
    parser.add_argument("--list", action="store_true", help="list monospace font families")
    parser.add_argument("--all-fonts", action="store_true", help="with --list, include all fonts")
    args = parser.parse_args()

    if args.list:
        for family, style, filename in list_fonts(not args.all_fonts):
            print(f"{family:<38} {style:<18} {filename}")
        return

    if not args.font:
        parser.error("a font name is required (or use --list)")

    try:
        written = generate(args.font, args.size, args.weight, args.dir, args.from_strip)
    except (ConfigError, OSError) as exc:
        sys.exit(f"error: {exc}")

    config = glyph_config.load(written)
    ramp = "".join(g["char"] for g in config["glyphs"])
    print(f"wrote {written}")
    print(f"  font   {config['font']['family']} ({config['font']['style']})")
    print(
        f"  cell   {config['cell']['width']}x{config['cell']['height']}px "
        f"aspect {config['cell']['aspect']:.3f}"
    )
    print(f"  ramp   {ramp}")


if __name__ == "__main__":
    main()
