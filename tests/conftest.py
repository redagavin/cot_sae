# ABOUTME: Shared pytest fixtures for the test suite.
# ABOUTME: Provides reusable synthetic data and configuration for tests.

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
