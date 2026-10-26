"""Re-run the pipeline and compare with tests/fixtures/golden_values.json ("was / is / difference").

Prints every number whose relative difference exceeds the tolerance (default 1e-9, i.e. rounding
only) and a one-line summary. Exit code 1 if anything differs. Never overwrites the golden file.

    python -m scripts.compare_golden --run fixture
    python -m scripts.compare_golden --run full
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

GOLDEN = Path("tests/fixtures/golden_values.json")


def flatten(node, prefix=""):
    """Nested dict / list -> {"a.b.0": value}."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from flatten(v, f"{prefix}{k}.")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from flatten(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1], node


def compare(was, now, rel_tol):
    """Return the list of (key, was, now, rel_diff) for values that differ beyond tolerance."""
    a, b = dict(flatten(was)), dict(flatten(now))
    rows = []
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key, "<missing>"), b.get(key, "<missing>")
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y))
        if numeric:
            if not math.isclose(x, y, rel_tol=rel_tol, abs_tol=1e-9):
                rows.append((key, x, y, abs(x - y) / max(abs(x), 1e-12)))
        elif x != y:
            rows.append((key, x, y, None))
    return rows, len(a)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", choices=["fixture", "full"], required=True)
    parser.add_argument("--rel-tol", type=float, default=1e-9)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "now.json"
        subprocess.run(
            [sys.executable, "-m", "scripts.capture_golden", "--run", args.run, "--out", str(out)],
            check=True,
            capture_output=True,
        )
        now = json.loads(out.read_text(encoding="utf8"))[args.run]
    was = json.loads(GOLDEN.read_text(encoding="utf8"))[args.run]
    rows, total = compare(was, now, args.rel_tol)
    print(f"{args.run}: {total} numbers compared, {len(rows)} differ (rel. tolerance {args.rel_tol:g})")
    for key, x, y, rel in rows[:40]:
        print(f"  {key}: was {x} / is {y} / rel. diff {rel}")
    sys.exit(1 if rows else 0)


if __name__ == "__main__":
    main()
