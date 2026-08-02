"""Fixtures for tests that actually launch Blender.

These are marked ``blender`` and skipped when no ``blender`` binary is on PATH, so
CI -- which has neither Blender nor a GPU -- stays green.

Output goes under ``$HOME`` rather than ``tmp_path``: the Blender here is a snap
package, and snap confinement makes paths outside the home directory unreliable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from chesssight.data.dataset import DatasetWriter
from chesssight.data.schema import DatasetMeta
from chesssight.synth import jobs
from chesssight.synth.config import GeneratorConfig

BLENDER = shutil.which("blender")
ENTRY_SCRIPT = (
    Path(__file__).resolve().parents[2] / "src" / "chesssight" / "blender" / "entry.py"
)

pytestmark = pytest.mark.skipif(BLENDER is None, reason="blender not on PATH")


@pytest.fixture(scope="session")
def render_root() -> Path:
    root = Path(
        os.environ.get(
            "CHESSSIGHT_TEST_ROOT", Path.home() / ".cache" / "chesssight" / "pytest"
        )
    )
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def run_blender(shard: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    """Render a shard, returning the completed process for assertion."""
    assert BLENDER is not None  # guarded by the module-level skipif
    return subprocess.run(
        [
            BLENDER,
            "--background",
            "--factory-startup",
            "--python",
            str(ENTRY_SCRIPT),
            "--",
            "--shard",
            str(shard),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def make_run(
    root: Path,
    *,
    count: int = 2,
    engine: str = "BLENDER_EEVEE",
    samples: int = 8,
    resolution: tuple[int, int] = (192, 144),
    seed: int = 4242,
    sampler=None,
    overrides: dict | None = None,
) -> tuple[GeneratorConfig, DatasetWriter, Path]:
    """Plan a tiny run and write its single shard."""
    payload: dict = {
        "count": count,
        "master_seed": seed,
        "render": {
            "engine": engine,
            "samples": samples,
            "resolution": list(resolution),
            "image_format": "PNG",
        },
        "positions": {"pgn_paths": [], "weight_pgn": 0.0, "weight_random": 1.0},
        "output": {"root": str(root.parent), "run_name": root.name},
    }
    payload.update(overrides or {})
    config = GeneratorConfig.model_validate(payload)
    writer = DatasetWriter(root)
    writer.initialise(
        DatasetMeta(
            name=root.name,
            created_at="2026-07-29T00:00:00Z",
            source="synthetic",
            master_seed=seed,
        )
    )
    specs = list(jobs.iter_jobs(config, writer, sampler=sampler))
    shard = jobs.write_shards(root, jobs.shard(specs, 1))[0]
    return config, writer, shard
