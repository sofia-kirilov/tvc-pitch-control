"""Run every stage in order. Figures go to figures/."""

import pathlib
import subprocess
import sys

stages = sorted((pathlib.Path(__file__).parent / "stages").glob("0*.py"))
for script in stages:
    print(f"\n=== {script.name} ===")
    subprocess.run([sys.executable, str(script)], check=True)
