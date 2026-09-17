"""Slice a character-strip image into equal blocks and report each glyph's average lightness."""

import argparse
import csv
import sys

import numpy as np
from PIL import Image

CHARSET = r'''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`~!@#$%^&*()-_=+[{]}\|;:'",<.>/?'''


def block_bounds(width, count):
    """(left, right) columns for `count` blocks, rounded so widths differ by at most 1px."""
    edges = [round(i * width / count) for i in range(count + 1)]
    return list(zip(edges[:-1], edges[1:]))


def srgb_to_linear(channel):
    """sRGB 0..1 -> linear-light 0..1."""
    return np.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055) ** 2.4)


def luminance_to_lstar(y):
    """Linear luminance 0..1 -> CIE L* 0..100."""
    f = np.where(y > 216 / 24389, np.cbrt(y), (24389 / 27 * y + 16) / 116)
    return 116 * f - 16


def analyze(path, chars):
    img = Image.open(path).convert("RGB")
    width = img.width
    count = len(chars)

    if count > width:
        raise ValueError(f"cannot make {count} blocks from an image only {width}px wide")

    srgb = np.asarray(img, dtype=np.float64) / 255.0
    # Averaging light must happen in linear space; averaging sRGB values overstates dark regions.
    luma = (srgb_to_linear(srgb) * [0.2126, 0.7152, 0.0722]).sum(axis=2)

    rows = []
    for index, (left, right) in enumerate(block_bounds(width, count)):
        block_luma = luma[:, left:right].mean()
        block_srgb = srgb[:, left:right, :].mean(axis=(0, 1))
        r, g, b = (int(round(c * 255)) for c in block_srgb)
        rows.append(
            {
                "index": index,
                "char": chars[index],
                "left": left,
                "right": right,
                "lstar": round(float(luminance_to_lstar(block_luma)), 2),
                "luminance": round(float(block_luma), 5),
                "gray": int(round(block_srgb.mean() * 255)),
                "hex": f"#{r:02X}{g:02X}{b:02X}",
            }
        )

    lo = min(r["lstar"] for r in rows)
    hi = max(r["lstar"] for r in rows)
    span = (hi - lo) or 1.0
    for row in rows:
        row["normalized"] = round((row["lstar"] - lo) / span, 4)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default="alphabet.png")
    parser.add_argument("--chars", default=CHARSET, help="characters in left-to-right order")
    parser.add_argument("--sort", action="store_true", help="order output darkest to lightest")
    parser.add_argument("--ramp", action="store_true", help="print the characters as a density ramp")
    parser.add_argument("--csv", metavar="FILE", help="also write results to this CSV file")
    args = parser.parse_args()

    try:
        rows = analyze(args.image, args.chars)
    except (OSError, ValueError) as exc:
        sys.exit(f"error: {exc}")

    display = sorted(rows, key=lambda r: r["lstar"]) if args.sort else rows

    print(f"{'#':>3}  {'char':^4}  {'L*':>6}  {'norm':>5}  {'gray':>4}  {'hex':<7}  bar")
    print("-" * 62)
    for row in display:
        bar = "#" * int(round(row["normalized"] * 28))
        print(
            f"{row['index']:>3}  {row['char']:^4}  {row['lstar']:>6.2f}  "
            f"{row['normalized']:>5.3f}  {row['gray']:>4}  {row['hex']:<7}  {bar}"
        )

    if args.ramp:
        ramp = "".join(r["char"] for r in sorted(rows, key=lambda r: r["lstar"]))
        print(f"\ndark -> light ramp:\n{ramp}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
