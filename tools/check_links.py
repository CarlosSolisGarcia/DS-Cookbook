"""Check that every relative Markdown link in the cookbook points to an existing file or folder.

Usage (from the repository root):  python tools/check_links.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def main() -> int:
    broken = []
    for md in sorted(ROOT.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        text = re.sub(r"```.*?```", "", text, flags=re.S)          # ignore code blocks
        for target in LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = target.split("#")[0]
            if not path:
                continue
            if not (md.parent / path).resolve().exists():
                broken.append((md.relative_to(ROOT), target))
    for f, t in broken:
        print(f"BROKEN  {f}  →  {t}")
    print(f"{len(broken)} broken link(s)")
    return int(bool(broken))


if __name__ == "__main__":
    sys.exit(main())
