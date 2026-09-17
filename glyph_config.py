"""The glyph-brightness config format shared by the generator and the renderer.

A config records how much ink each character puts on the page in one specific
font, so an ASCII-art renderer can pick the glyph whose density matches a pixel.

    {
      "format": "glyph-brightness",
      "version": 1,
      "font":   {"family": ..., "style": ..., "file": ..., "weight": ..., "size": ...},
      "cell":   {"width": px, "height": px, "aspect": height / width},
      "measurement": {"method": ..., "normalized_to": "@", "generated": ISO-8601},
      "glyphs": [{"char": " ", "coverage": 0.0, "weight": 0.0}, ...]
    }

`coverage` is the raw fraction of the cell covered by ink. `weight` is that
value rescaled so the densest glyph is 1.0 and a blank cell is 0.0 -- the
0..1 scale a renderer matches pixel brightness against. Glyphs are listed
darkest-first.
"""

import json
import re
from pathlib import Path

FORMAT = "glyph-brightness"
VERSION = 1

CHARSET = r'''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`~!@#$%^&*()-_=+[{]}\|;:'",<.>/?'''

CONFIG_DIR = Path(__file__).resolve().parent / "glyph-configs"


class ConfigError(Exception):
    """A config file is missing, malformed, or from an incompatible version."""


def slugify(family, weight=None):
    slug = re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-") or "font"
    return f"{slug}-w{weight}" if weight else slug


def config_path(family, weight=None, directory=None):
    root = Path(directory) if directory else CONFIG_DIR
    return root / f"{slugify(family, weight)}.json"


def build(family, style, file, weight, size, cell_width, cell_height, coverage, method):
    """Assemble a config dict from raw per-character coverage measurements."""
    peak = max(coverage.values())
    if peak <= 0:
        raise ConfigError(f"no glyph in {family!r} registered any ink")

    densest = max(coverage, key=coverage.get)
    glyphs = [{"char": " ", "coverage": 0.0, "weight": 0.0}]
    glyphs += [
        {"char": char, "coverage": round(value, 6), "weight": round(value / peak, 6)}
        for char, value in coverage.items()
    ]
    glyphs.sort(key=lambda g: g["weight"])

    return {
        "format": FORMAT,
        "version": VERSION,
        "font": {
            "family": family,
            "style": style,
            "file": str(file),
            "weight": weight,
            "size": size,
        },
        "cell": {
            "width": cell_width,
            "height": cell_height,
            "aspect": round(cell_height / cell_width, 6),
        },
        "measurement": {
            "method": method,
            "normalized_to": densest,
            "generated": _timestamp(),
        },
        "glyphs": glyphs,
    }


def _timestamp():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load(path):
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"no config at {path}")
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}")

    if config.get("format") != FORMAT:
        raise ConfigError(f"{path} is not a {FORMAT} config")
    if config.get("version") != VERSION:
        raise ConfigError(f"{path} is version {config.get('version')}, expected {VERSION}")
    if not config.get("glyphs"):
        raise ConfigError(f"{path} lists no glyphs")

    return config


def weights(config):
    """{char: weight} from a loaded config."""
    return {g["char"]: g["weight"] for g in config["glyphs"]}


def aspect(config):
    return config["cell"]["aspect"]


def available(directory=None):
    root = Path(directory) if directory else CONFIG_DIR
    return sorted(root.glob("*.json")) if root.is_dir() else []
