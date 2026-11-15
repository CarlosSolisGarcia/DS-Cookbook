"""
Execute every ```python code block of the cookbook to check that the examples run.

Blocks inside one Markdown file are concatenated in order and executed in a
fresh subprocess (so later blocks may use variables defined earlier in the
same recipe — exactly how you would run them in a notebook).

A block whose first line is `# not-run` is skipped (used for fragments that
need external services, e.g. an MLflow server, or that are templates).

Usage (from the repository root):
    python tools/validate_code_blocks.py                 # all recipes
    python tools/validate_code_blocks.py 03_regression   # one folder / file
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```", re.S | re.M)
PRELUDE = (
    "import matplotlib\nmatplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\nplt.show = lambda *a, **k: plt.close('all')\n"
    "import warnings\nwarnings.filterwarnings('ignore', category=FutureWarning)\n"
)


def blocks_of(md: Path) -> list[str]:
    out = []
    for b in BLOCK.findall(md.read_text(encoding="utf-8")):
        if b.lstrip().startswith("# not-run"):
            continue
        out.append(b)
    return out


def run(md: Path, timeout: int = 900) -> tuple[bool, str, float]:
    code = blocks_of(md)
    if not code:
        return True, "no code", 0.0
    src = PRELUDE + "\n\n".join(code)
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "recipe.py"
        script.write_text(src, encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
        t0 = time.time()
        p = subprocess.run([sys.executable, str(script)], cwd=tmp, env=env,
                           capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
    ok = p.returncode == 0
    return ok, (p.stderr[-3000:] if not ok else f"{len(code)} blocks"), dt


def main() -> int:
    targets = sys.argv[1:] or ["."]
    files: list[Path] = []
    for t in targets:
        p = (ROOT / t).resolve()
        files += [p] if p.is_file() else sorted(p.rglob("*.md"))
    files = [f for f in files if "tools" not in f.parts]
    failed = 0
    for f in files:
        ok, msg, dt = run(f)
        if msg == "no code":
            continue
        print(f"{'PASS' if ok else 'FAIL'}  {f.relative_to(ROOT)}  ({dt:.1f}s)  {msg if ok else ''}")
        if not ok:
            failed += 1
            print(msg)
    print(f"\n{failed} file(s) failed")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
