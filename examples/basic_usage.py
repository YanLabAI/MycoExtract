"""Run the manuscript workflow on the first parsed paper in the inventory."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("papers_dir", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts/run_all_papers_full_extraction.py"),
        "--all-papers-dir", str(args.papers_dir.resolve()),
        "--limit", "1",
        "--output-root", "analysis_outputs/example",
    ]
    if args.inventory_only:
        command.append("--inventory-only")
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    sys.exit(main())
