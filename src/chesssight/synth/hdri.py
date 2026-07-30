"""Fetch HDRI environment maps for image-based lighting.

Why this exists: the detector trained on renders reached mAP 0.37 on real
photographs against 0.85 once real photographs were mixed in. Board *geometry*
transferred fine; piece *appearance* did not. Appearance under light is what an
environment map fixes -- the procedural sun-plus-fills rig produces clean,
directional, physically simple light, while a real room throws colour from painted
walls, soft window gradients, multiple mismatched fixtures and reflections that
land differently on every curved piece. That mismatch is exactly the kind of thing
a classifier latches onto.

Source is Poly Haven, whose HDRIs are all **CC0** -- no attribution required,
which makes this the one asset class in the project with no licence to propagate.
Credits are recorded anyway, for the same reason everything else here is.

The maps are curated by name rather than pulled by category. A category filter
returns abandoned factories and Christmas photo studios along with the useful
ones, and lighting a chessboard by a derelict boiler room is domain *noise*, not
domain randomisation. Everything below is a room a game could plausibly be played
in: halls, classrooms, offices, cafés, lounges, plus a few deliberately plain
rooms so the set is not all dramatic.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

API = "https://api.polyhaven.com"
LICENSE = "CC0 1.0"
SOURCE_URL = "https://polyhaven.com/hdris"

#: Curated Poly Haven HDRI slugs, grouped by the kind of venue they stand in for.
#: Mixed daylight and artificial light on purpose: tournaments happen in halls with
#: windows *and* in rooms lit only by fixtures, and ChessReD contains both.
CURATED: dict[str, tuple[str, ...]] = {
    "halls": (
        "school_hall",
        "events_hall_interior",
        "marry_hall",
        "old_hall",
        "studio_country_hall",
        "music_hall_02",
    ),
    "rooms": (
        "combination_room",
        "church_meeting_room",
        "empty_play_room",
        "small_empty_room_1",
        "small_empty_room_2",
        "small_empty_room_4",
    ),
    "offices": (
        "unfinished_office",
        "unfinished_office_night",
        "cayley_interior",
        "newman_lobby",
    ),
    "social": (
        "comfy_cafe",
        "newman_cafeteria",
        "lythwood_lounge",
        "wooden_lounge",
        "anniversary_lounge",
        "warm_restaurant",
    ),
}

#: 1k is enough to *light* a scene, but low camera elevations see the world behind
#: the table, so the map is also a visible backdrop. 2k keeps that from smearing
#: at 512px without the 24 MB/map that 4k costs.
DEFAULT_RESOLUTION = "2k"


class HdriError(RuntimeError):
    """Raised when an HDRI cannot be listed or fetched."""


def all_slugs() -> list[str]:
    """Every curated slug, in a stable order."""
    return [slug for group in CURATED.values() for slug in group]


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "chesssight/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        raise HdriError(f"cannot read {url}: {error}") from error


def file_url(slug: str, resolution: str = DEFAULT_RESOLUTION) -> tuple[str, int]:
    """Direct ``.hdr`` URL and byte size for one map at one resolution."""
    files = _get_json(f"{API}/files/{slug}")
    hdri = files.get("hdri", {})
    if resolution not in hdri:
        raise HdriError(
            f"{slug} has no {resolution} variant; available: {sorted(hdri)}"
        )
    entry = hdri[resolution].get("hdr")
    if not entry or "url" not in entry:
        raise HdriError(f"{slug} has no .hdr at {resolution}")
    return entry["url"], int(entry.get("size", 0))


def credits(slug: str) -> dict:
    """Author and category metadata, recorded alongside the downloaded maps."""
    info = _get_json(f"{API}/info/{slug}")
    return {
        "slug": slug,
        "name": info.get("name", slug),
        "authors": list(info.get("authors", {})),
        "categories": info.get("categories", []),
        "whitebalance": info.get("whitebalance"),
    }


def download(
    out_dir: Path,
    *,
    slugs: list[str] | None = None,
    resolution: str = DEFAULT_RESOLUTION,
    progress: Callable[[str], None] = print,
    skip_existing: bool = True,
) -> dict:
    """Fetch the curated maps into ``out_dir`` and write a credits file."""
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = slugs if slugs is not None else all_slugs()

    entries = []
    downloaded = skipped = 0
    for index, slug in enumerate(wanted, start=1):
        target = out_dir / f"{slug}_{resolution}.hdr"
        if skip_existing and target.is_file():
            progress(f"[chesssight] {index}/{len(wanted)} {slug}: already present")
            skipped += 1
            entries.append(credits(slug) | {"file": target.name})
            continue

        url, size = file_url(slug, resolution)
        progress(f"[chesssight] {index}/{len(wanted)} {slug} ({size // 1024} KB)")
        request = urllib.request.Request(url, headers={"User-Agent": "chesssight/0.1"})
        # Written to a temporary name and renamed, so an interrupted fetch cannot
        # leave a truncated .hdr that Blender would load as a black world.
        partial = target.with_suffix(".hdr.part")
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                partial.write_bytes(response.read())
        except (urllib.error.URLError, OSError) as error:
            partial.unlink(missing_ok=True)
            raise HdriError(f"cannot fetch {slug}: {error}") from error
        partial.replace(target)
        downloaded += 1
        entries.append(credits(slug) | {"file": target.name})

    (out_dir / "credits.json").write_text(
        json.dumps(
            {
                "source": SOURCE_URL,
                "license": LICENSE,
                "note": (
                    "Poly Haven HDRIs are CC0: no attribution is required. "
                    "Credits recorded for provenance."
                ),
                "resolution": resolution,
                "maps": entries,
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
