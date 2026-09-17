"""Render an image as ASCII art using per-font glyph brightnesses.

    python ascii_art.py photo.jpg 100
    python ascii_art.py photo.jpg 100 --font Consolas --noise 20
    python ascii_art.py photo.jpg 100 --png art.png --color truecolor

Glyph densities come from a config in glyph-configs/, measured from the real
font by make_font_config.py. If the requested font has no config yet, one is
generated on first use. Every pixel of the downscaled image is replaced by the
glyph whose weight is closest to its brightness -- ' ' for black, '@' for white.

--png writes the art to an image instead of the terminal, drawn in the same font
the ramp was measured from and left transparent everywhere the glyphs do not ink.
It takes its colours from --color exactly as the terminal does.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import glyph_config
from glyph_config import ConfigError

DEFAULT_FONT = "Cascadia Mono"

PRESETS = {
    "full": None,  # every glyph in the config
    "alnum": "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "letters": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "minimal": ".:-=+*#%@",
}


def noise_level(value):
    number = int(value)
    if not 0 <= number <= 100:
        raise argparse.ArgumentTypeError("noise must be between 0 and 100")
    return number


def srgb_to_linear(channel):
    return np.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055) ** 2.4)


def luminance_to_lstar(y):
    f = np.where(y > 216 / 24389, np.cbrt(y), (24389 / 27 * y + 16) / 116)
    return 116 * f - 16


RESET = "\033[0m"


def detect_color():
    """Pick a colour depth for the current terminal, or 'off' if it can't show any."""
    if os.environ.get("NO_COLOR"):
        return "off"
    if not sys.stdout.isatty():
        return "off"
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    # Windows Terminal handles 24-bit fine but does not advertise COLORTERM.
    if os.environ.get("WT_SESSION"):
        return "truecolor"
    if "256" in os.environ.get("TERM", ""):
        return "256"
    return "off"


def enable_windows_vt():
    """Legacy conhost prints escape codes literally until VT processing is on."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass  # Nothing to do if the console refuses; Windows Terminal needs no help.


XTERM_LEVELS = (0, 95, 135, 175, 215, 255)


def to_xterm256(r, g, b):
    """Nearest index in the xterm-256 palette: a 6x6x6 cube plus a 24-step grey ramp."""
    levels = XTERM_LEVELS

    def nearest_level(value):
        return min(range(6), key=lambda i: abs(levels[i] - value))

    ri, gi, bi = nearest_level(r), nearest_level(g), nearest_level(b)
    cube_error = sum(
        (levels[i] - v) ** 2 for i, v in ((ri, r), (gi, g), (bi, b))
    )

    grey_step = min(range(24), key=lambda i: abs((8 + i * 10) - (r + g + b) / 3))
    grey = 8 + grey_step * 10
    grey_error = (grey - r) ** 2 + (grey - g) ** 2 + (grey - b) ** 2

    if grey_error < cube_error:
        return 232 + grey_step
    return 16 + 36 * ri + 6 * gi + bi


def from_xterm256(index):
    """The RGB a terminal actually shows for a palette index."""
    if index >= 232:
        grey = 8 + (index - 232) * 10
        return grey, grey, grey

    index -= 16
    return XTERM_LEVELS[index // 36], XTERM_LEVELS[index // 6 % 6], XTERM_LEVELS[index % 6]


def cell_colour(rgb, depth):
    """The colour a cell ends up as at this depth, after any palette quantization."""
    r, g, b = (int(v) for v in rgb)
    if depth == "256":
        return from_xterm256(to_xterm256(r, g, b))
    return r, g, b


def colour_escape(rgb, depth):
    r, g, b = (int(v) for v in rgb)
    if depth == "256":
        return f"\033[38;5;{to_xterm256(r, g, b)}m"
    return f"\033[38;2;{r};{g};{b}m"


# Without --color the terminal draws glyphs in its own foreground; a PNG has no
# such default, so it stands in with white.
PNG_INK = (255, 255, 255)


def vivify(rgb):
    """Push each cell's colour to full brightness, keeping its hue and saturation."""
    peak = rgb.max(axis=2, keepdims=True)
    scale = np.where(peak > 0, 255.0 / np.maximum(peak, 1), 1.0)
    return np.clip(rgb * scale, 0, 255)


def render_plain(glyphs):
    return "\n".join("".join(row) for row in glyphs)


def render_colour(glyphs, rgb, depth, vivid):
    """Colour each glyph by its cell, emitting an escape only when the colour changes."""
    if vivid:
        rgb = vivify(rgb.astype(np.float64))

    lines = []
    for glyph_row, colour_row in zip(glyphs, rgb):
        parts, current = [], None
        for glyph, cell in zip(glyph_row, colour_row):
            # A space shows only the background, so it never needs its own colour.
            if glyph != " ":
                escape = colour_escape(cell, depth)
                if escape != current:
                    parts.append(escape)
                    current = escape
            parts.append(glyph)
        lines.append("".join(parts) + RESET)

    return "\n".join(lines)


def open_png_font(config, weight, size):
    """Open the font a config was measured from, sized for PNG output."""
    import make_font_config

    info = config.get("font") or {}
    file, family = info.get("file"), info.get("family")

    font = None
    # A collection file needs the face index only find_font knows, so skip it here.
    if file and Path(file).suffix.lower() in (".ttf", ".otf"):
        try:
            font = ImageFont.truetype(file, size)
        except OSError:
            font = None

    if font is None:
        path, index, _, _ = make_font_config.find_font(family or DEFAULT_FONT)
        font = ImageFont.truetype(str(path), size, index=index)

    make_font_config.apply_weight(font, weight)
    return font


def cell_size(font, chars):
    """Pixel size of one character cell: the widest advance by the full line height."""
    ascent, descent = font.getmetrics()
    width = math.ceil(max(font.getlength(char) for char in chars))
    return max(1, width), max(1, ascent + descent)


def render_png(glyphs, rgb, font, depth, vivid):
    """Draw the glyph grid onto a transparent canvas, one glyph per cell."""
    if depth != "off" and vivid:
        rgb = vivify(rgb.astype(np.float64))

    cell_width, cell_height = cell_size(font, set("".join("".join(row) for row in glyphs)))
    image = Image.new(
        "RGBA", (len(glyphs[0]) * cell_width, len(glyphs) * cell_height), (0, 0, 0, 0)
    )
    draw = ImageDraw.Draw(image)

    for row, (glyph_row, colour_row) in enumerate(zip(glyphs, rgb)):
        for column, (glyph, cell) in enumerate(zip(glyph_row, colour_row)):
            # A space inks nothing, so leaving it out keeps that cell transparent.
            if glyph == " ":
                continue
            colour = cell_colour(cell, depth) if depth != "off" else PNG_INK
            draw.text(
                (column * cell_width, row * cell_height), glyph, font=font, fill=colour + (255,)
            )

    return image


def resolve_config(font, weight, explicit, allow_generate=True, quiet=False):
    """Load the config for a font, measuring it on first use."""
    path = Path(explicit) if explicit else glyph_config.config_path(font, weight)

    try:
        return glyph_config.load(path)
    except ConfigError:
        if explicit or not allow_generate:
            raise

    import make_font_config

    if not quiet:
        print(f"measuring {font}...", file=sys.stderr)
    written = make_font_config.generate(font, weight=weight, quiet=quiet)
    return glyph_config.load(written)


def build_ramp(weights, allowed):
    """Return (chars, weights) sorted dark to light, one glyph per distinct weight."""
    usable = {c: w for c, w in weights.items() if allowed is None or c in allowed}
    usable[" "] = 0.0

    seen, chars, values = set(), [], []
    for char, weight in sorted(usable.items(), key=lambda kv: kv[1]):
        key = round(weight, 3)
        if key in seen:
            continue
        seen.add(key)
        chars.append(char)
        values.append(weight)

    if len(chars) < 2:
        raise ConfigError("need at least two distinct glyphs to build a ramp")

    return chars, np.array(values)


def load_image(path, background):
    """Open an image, flattening any transparency onto a solid background."""
    img = ImageOps.exif_transpose(Image.open(path))

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        shade = 255 if background == "white" else 0
        canvas = Image.new("RGBA", img.size, (shade, shade, shade, 255))
        img = Image.alpha_composite(canvas, img)

    return img.convert("RGB")


def brightness_map(img, linear, gamma, stretch):
    """Per-pixel brightness in 0..1, where 1 is the lightest tone."""
    srgb = np.asarray(img, dtype=np.float64) / 255.0
    luminance = (srgb_to_linear(srgb) * [0.2126, 0.7152, 0.0722]).sum(axis=2)
    values = luminance if linear else luminance_to_lstar(luminance) / 100.0

    if stretch:
        lo, hi = values.min(), values.max()
        if hi > lo:
            values = (values - lo) / (hi - lo)

    if gamma != 1.0:
        values = np.clip(values, 0, 1) ** (1.0 / gamma)

    return np.clip(values, 0, 1)


def apply_noise(brightness, noise, seed):
    """Blend each pixel toward an independent random value.

    `noise` is a percentage: 0 leaves the image untouched, 100 replaces every
    pixel with pure randomness, and values between mix the two proportionally.
    """
    if not noise:
        return brightness

    mix = noise / 100.0
    rng = np.random.default_rng(seed)
    return (1.0 - mix) * brightness + mix * rng.random(brightness.shape)


def to_ascii(path, width, chars, values, aspect, invert, linear, gamma, stretch,
             background, noise, seed):
    """Return (glyph grid, per-cell RGB) for the image at the requested width."""
    if width < 1:
        raise ValueError("width must be at least 1")

    img = load_image(path, background)
    height = max(1, round(img.height * (width / img.width) / aspect))
    img = img.resize((width, height), Image.LANCZOS)

    # Downscaling already averaged each cell, so these pixels are the cell colours.
    rgb = np.asarray(img, dtype=np.uint8)

    brightness = brightness_map(img, linear, gamma, stretch)
    if invert:
        brightness = 1.0 - brightness

    brightness = np.clip(apply_noise(brightness, noise, seed), 0, 1)

    # Nearest glyph weight for every pixel, vectorized over the whole image.
    indices = np.abs(brightness[:, :, None] - values[None, None, :]).argmin(axis=2)
    return np.array(chars)[indices], rgb


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", nargs="?", help="image to convert")
    parser.add_argument(
        "width", nargs="?", type=int, default=100, help="output width in characters (default 100)"
    )
    parser.add_argument("--font", default=DEFAULT_FONT, help=f"font family (default {DEFAULT_FONT})")
    parser.add_argument("--weight", type=int, help="weight axis value for variable fonts")
    parser.add_argument("--config", help="use this config file instead of looking one up")
    parser.add_argument("--charset", choices=sorted(PRESETS), default="full")
    parser.add_argument("--chars", help="explicit glyph set, overrides --charset")
    parser.add_argument("--aspect", type=float, help="override the config's cell height/width")
    parser.add_argument(
        "--invert", action="store_true", help="for dark text on a light background"
    )
    parser.add_argument(
        "--background",
        choices=("black", "white"),
        default="black",
        help="what transparent areas become (default black)",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="match raw linear luminance instead of perceptual lightness",
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="brighten (>1) or darken (<1)")
    parser.add_argument(
        "--stretch", action="store_true", help="rescale contrast to use the full glyph range"
    )
    parser.add_argument(
        "--noise",
        type=noise_level,
        default=0,
        metavar="0-100",
        help="percent randomness mixed into each pixel (0 none, 100 fully random)",
    )
    parser.add_argument("--seed", type=int, help="seed the noise for repeatable output")
    parser.add_argument(
        "--color",
        "--colour",
        dest="color",
        choices=("off", "auto", "256", "truecolor"),
        default="off",
        help="colour each glyph by its cell (auto detects what the terminal supports)",
    )
    parser.add_argument(
        "--vivid",
        action="store_true",
        help="with --color, push cell colours to full brightness so glyphs carry the shading",
    )
    parser.add_argument(
        "--png",
        metavar="FILE",
        help="write the art to this PNG with a transparent background instead of printing it",
    )
    parser.add_argument(
        "--png-size", type=int, default=20, metavar="PX", help="font size for --png (default 20)"
    )
    parser.add_argument("--show-ramp", action="store_true", help="print the ramp and exit")
    parser.add_argument("--list-configs", action="store_true", help="list measured fonts")
    args = parser.parse_args()

    if args.list_configs:
        for path in glyph_config.available():
            config = glyph_config.load(path)
            print(f"{config['font']['family']:<30} {path.name}")
        return

    try:
        config = resolve_config(args.font, args.weight, args.config)
        allowed = set(args.chars) if args.chars else PRESETS[args.charset]
        chars, values = build_ramp(glyph_config.weights(config), allowed)
    except ConfigError as exc:
        sys.exit(f"error: {exc}")

    if args.show_ramp:
        print("".join(chars))
        return

    if not args.image:
        parser.error("an image is required")

    if args.png and args.png_size < 1:
        parser.error("--png-size must be at least 1")

    depth = detect_color() if args.color == "auto" else args.color
    if depth != "off" and not args.png:
        enable_windows_vt()

    try:
        glyphs, rgb = to_ascii(
            args.image, args.width, chars, values,
            args.aspect or glyph_config.aspect(config),
            args.invert, args.linear, args.gamma, args.stretch,
            args.background, args.noise, args.seed,
        )
    except (OSError, ValueError) as exc:
        sys.exit(f"error: {exc}")

    if args.png:
        try:
            font = open_png_font(config, args.weight, args.png_size)
            render_png(glyphs, rgb, font, depth, args.vivid).save(args.png)
        except (ConfigError, OSError, ValueError) as exc:
            sys.exit(f"error: {exc}")
        print(f"wrote {args.png}", file=sys.stderr)
    elif depth == "off":
        print(render_plain(glyphs))
    else:
        print(render_colour(glyphs, rgb, depth, args.vivid))


if __name__ == "__main__":
    main()
