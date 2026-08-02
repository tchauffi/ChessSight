"""Blender-side rendering code.

Everything in this package runs inside Blender's own bundled Python (3.13 here), not
in the project's uv venv. It may import **stdlib and numpy only** -- Blender ships
numpy 2.3.4 but has no ``pydantic``, ``PIL``, ``yaml`` or ``scipy``. It may also
import the deliberately import-light project modules ``chesssight.data.fen`` and
``chesssight.synth.profiles``.

This package is excluded from mypy (see ``[tool.mypy]`` in ``pyproject.toml``)
because ``bpy`` does not exist in the venv or in CI. Its correctness is covered by
the integration tests instead, which are skipped when no ``blender`` binary is found.
"""
