"""Fetch PBR surface textures for the table the board stands on.

Why this exists: every material in this generator was procedural noise, and the
table -- the largest surface in frame after the board itself -- was a flat colour
with nothing on it at all. Noise makes a surface *non-uniform*; it does not make it
a material. Real wood has figure that runs, joins between boards, wear along an
edge, and a varnish that pools unevenly over all of it, and none of that is
reachable from a noise node.

The evidence that this matters is indirect but strong. Diffusion restyling of these
renders was tried and abandoned -- it moves geometry, which breaks the labels -- but
what it *did* to the images before breaking them was instructive: the first thing it
changed, every time, was to put real grain on the table, and that alone made the
frames read as photographs. This module supplies that directly, at render time,
where the labels stay exact.

Source is Poly Haven, same as the HDRIs, and the textures are equally **CC0**. Maps
are curated by name rather than pulled by category, for the same reason the HDRIs
are: a category query returns bark, rusted shutters and climbing walls, and a chess
board does not stand on any of those.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

API = "https://api.polyhaven.com"
LICENSE = "CC0 1.0"
SOURCE_URL = "https://polyhaven.com/textures"

#: The maps actually used. Diffuse and roughness carry most of the look; the normal
#: map supplies relief that a bump node approximates badly on a large flat surface
#: seen at a grazing angle -- which is the whole lower half of this dataset.
#: Displacement and AO are deliberately skipped: displacement needs subdivision the
#: table does not have, and Cycles computes occlusion properly on its own.
MAPS: tuple[str, ...] = ("Diffuse", "Rough", "nor_gl")

#: Curated Poly Haven texture slugs, grouped by the kind of surface they stand in
#: for. Everything here is something a board could plausibly sit on: a wooden table,
#: a laminate desk, a cloth, a stone worktop.
CURATED: dict[str, tuple[str, ...]] = {
    "wood": (
        "oak_veneer_01",
        "plank_flooring_02",
        "plank_flooring_03",
        "dark_wooden_planks",
        "laminate_floor_02",
        "brown_planks_05",
    ),
    "parquet": (
        "diagonal_parquet",
        "herringbone_parquet",
    ),
    "cloth": (
        "denim_fabric",
        "cotton_jersey",
        "brown_leather",
        "leather_white",
    ),
    "stone": (
        "concrete_floor",
        "brushed_concrete",
    ),
}

#: 2k is the right size for a table that fills a third of a 640px frame. 1k smears
#: visibly at grazing angles; 4k quadruples disk and load time for detail no camera
#: in this dataset resolves.
DEFAULT_RESOLUTION = "2k"


class TextureError(RuntimeError):
    """Raised when a texture cannot be listed or fetched."""


def all_slugs() -> list[str]:
    """Every curated slug, in a stable order."""
    return [slug for group in CURATED.values() for slug in group]


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "chesssight/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        raise TextureError(f"cannot read {url}: {error}") from error


def map_urls(slug: str, resolution: str = DEFAULT_RESOLUTION) -> dict[str, str]:
    """Direct JPG URLs for each wanted map of one texture."""
    files = _get_json(f"{API}/files/{slug}")
    urls = {}
    for name in MAPS:
        entry = files.get(name, {}).get(resolution, {}).get("jpg")
        if not entry or "url" not in entry:
            raise TextureError(f"{slug} has no {name} jpg at {resolution}")
        urls[name] = entry["url"]
    return urls


def credits(slug: str) -> dict:
    """Author and category metadata, recorded alongside the downloaded maps."""
    info = _get_json(f"{API}/info/{slug}")
    return {
        "slug": slug,
        "name": info.get("name", slug),
        "authors": list(info.get("authors", {})),
        "categories": info.get("categories", []),
    }


def texture_sets(root: Path, resolution: str = DEFAULT_RESOLUTION) -> list[dict]:
    """Every complete texture set under ``root``, as map-name -> absolute path.

    A set missing any of :data:`MAPS` is skipped rather than half-loaded: a material
    wired to a diffuse map but no roughness silently renders shinier than intended,
    which is worse than not using the texture at all.
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        return []

    # Enumerate from the diffuse maps: one per texture, so the slug falls out by
    # stripping a known suffix rather than by guessing where the name ends. Slugs
    # themselves contain underscores and digits, so splitting on those does not work.
    suffix = f"_Diffuse_{resolution}.jpg"
    found = []
    for path in sorted(root.glob(f"*{suffix}")):
        slug = path.name[: -len(suffix)]
        paths = {}
        for name in MAPS:
            candidate = root / f"{slug}_{name}_{resolution}.jpg"
            if candidate.is_file():
                paths[name] = str(candidate)
        if len(paths) == len(MAPS):
            found.append({"slug": slug, "maps": paths})
    return found


def download(
    out_dir: Path,
    *,
    slugs: list[str] | None = None,
    resolution: str = DEFAULT_RESOLUTION,
    progress: Callable[[str], None] = print,
    skip_existing: bool = True,
) -> dict:
    """Fetch the curated textures into ``out_dir`` and write a credits file."""
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = slugs if slugs is not None else all_slugs()

    entries = []
    downloaded = skipped = 0
    for index, slug in enumerate(wanted, start=1):
        targets = {name: out_dir / f"{slug}_{name}_{resolution}.jpg" for name in MAPS}
        if skip_existing and all(path.is_file() for path in targets.values()):
            progress(f"[chesssight] {index}/{len(wanted)} {slug}: already present")
            skipped += 1
            names = [p.name for p in targets.values()]
            entries.append(credits(slug) | {"files": names})
            continue

        urls = map_urls(slug, resolution)
        progress(f"[chesssight] {index}/{len(wanted)} {slug} ({len(MAPS)} maps)")
        for name, url in urls.items():
            target = targets[name]
            if skip_existing and target.is_file():
                continue
            request = urllib.request.Request(
                url, headers={"User-Agent": "chesssight/0.1"}
            )
            # Written to a temporary name and renamed, so an interrupted fetch
            # cannot leave a truncated JPG that Blender loads as a pink error.
            partial = target.with_suffix(".jpg.part")
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    partial.write_bytes(response.read())
            except (urllib.error.URLError, OSError) as error:
                partial.unlink(missing_ok=True)
                raise TextureError(f"cannot fetch {slug} {name}: {error}") from error
            partial.replace(target)
        downloaded += 1
        entries.append(credits(slug) | {"files": [p.name for p in targets.values()]})

    (out_dir / "credits.json").write_text(
        json.dumps(
            {
                "source": SOURCE_URL,
                "license": LICENSE,
                "note": (
                    "Poly Haven textures are CC0: no attribution is required. "
                    "Credits recorded for provenance."
                ),
                "resolution": resolution,
                "maps": MAPS,
                "textures": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "total": len(entries),
        "dir": str(out_dir),
    }
