#!/usr/bin/env python3
"""Legacy compatibility entry point for the canonical ASTEROID-NEO module."""

from __future__ import annotations

from asteroid_neo import *  # noqa: F401,F403 - preserve historical script imports
from asteroid_neo import main


if __name__ == "__main__":
    raise SystemExit(main())
