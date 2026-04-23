"""Backwards-compatible entry point.

Project metadata, dependencies, and extras are declared in ``pyproject.toml``.
This shim exists only so that older ``pip install`` flows (pip < 21.3) and
``pip install -e .`` on systems without PEP 517 support still work.
"""

from setuptools import setup

setup()
