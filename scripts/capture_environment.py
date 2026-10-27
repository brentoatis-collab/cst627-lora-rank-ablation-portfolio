#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import capture_environment

capture_environment(ROOT / "artifacts/environment/environment.json")
print("Wrote artifacts/environment/environment.json")
