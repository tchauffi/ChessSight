"""Enforce the project/Blender import boundary.

Blender's bundled Python has numpy but no pydantic, PIL, yaml, typer, chess or
zstandard. Anything reachable from ``chesssight.blender`` that imports one of those
fails at *render* time, deep inside a subprocess, with the traceback buried in a
shard log -- which is exactly how it was found once already.

This walks the import graph statically, so the same mistake fails in CI instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
BLENDER_PACKAGE = SRC / "chesssight" / "blender"

#: Installed in this venv but absent from Blender's Python.
FORBIDDEN_ON_BLENDER_SIDE = {
    "pydantic",
    "pydantic_core",
    "yaml",
    "typer",
    "click",
    "PIL",
    "chess",
    "zstandard",
    "rich",
}

#: Third-party modules Blender does provide.
ALLOWED_THIRD_PARTY = {
    "bpy",
    "bmesh",
    "mathutils",
    "bpy_extras",
    "numpy",
    "addon_utils",
}


def module_name(path: Path) -> str:
    return ".".join(path.relative_to(SRC).with_suffix("").parts)


def imported_modules(path: Path) -> set[str]:
    """Top-level module names imported by a file, including inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, stays inside the package
                continue
            if node.module:
                found.add(node.module.split(".")[0])
    return found


def chesssight_imports(path: Path) -> set[str]:
    """Fully-qualified ``chesssight.*`` modules imported by a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if not node.module.startswith("chesssight"):
                continue
            for alias in node.names:
                candidate = f"{node.module}.{alias.name}"
                if (SRC / Path(*candidate.split("."))).with_suffix(".py").is_file():
                    found.add(candidate)
                else:
                    found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("chesssight"):
                    found.add(alias.name)
    return found


def path_for(module: str) -> Path | None:
    candidate = (SRC / Path(*module.split("."))).with_suffix(".py")
    return candidate if candidate.is_file() else None


def reachable_from_blender() -> set[Path]:
    """Every project file the Blender side can pull in, transitively."""
    queue = sorted(BLENDER_PACKAGE.rglob("*.py"))
    seen: set[Path] = set()
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        for module in chesssight_imports(current):
            target = path_for(module)
            if target and target not in seen:
                queue.append(target)
    return seen


BLENDER_FILES = sorted(BLENDER_PACKAGE.rglob("*.py"))
REACHABLE = sorted(reachable_from_blender())


def test_the_blender_package_is_not_empty():
    assert len(BLENDER_FILES) >= 5


@pytest.mark.parametrize("path", REACHABLE, ids=lambda p: module_name(p))
def test_no_forbidden_imports_reachable_from_blender(path: Path):
    offenders = imported_modules(path) & FORBIDDEN_ON_BLENDER_SIDE
    assert not offenders, (
        f"{module_name(path)} imports {sorted(offenders)}, which Blender's bundled "
        f"Python does not have. It is reachable from chesssight.blender, so this "
        f"would fail at render time inside a subprocess."
    )


def test_reachable_set_includes_the_shared_modules():
    # Guards the walker itself: if it stopped following imports, the test above
    # would pass vacuously.
    names = {module_name(path) for path in REACHABLE}
    assert "chesssight.blender.scene" in names
    assert "chesssight.data.fen" in names
    assert "chesssight.synth.profiles" in names


def test_the_validated_manifest_module_is_not_reachable():
    # chesssight.synth.assets is pydantic-backed on purpose; the Blender side must
    # go through chesssight.synth.asset_spec instead.
    names = {module_name(path) for path in REACHABLE}
    assert "chesssight.synth.assets" not in names
    assert "chesssight.synth.asset_spec" in names


def test_shared_modules_use_only_blender_safe_third_party():
    import sys

    stdlib = set(sys.stdlib_module_names)
    for path in REACHABLE:
        for module in imported_modules(path):
            if module.startswith("chesssight") or module in stdlib:
                continue
            assert module in ALLOWED_THIRD_PARTY, (
                f"{module_name(path)} imports {module!r}, which is neither stdlib "
                f"nor known to exist in Blender's Python"
            )
