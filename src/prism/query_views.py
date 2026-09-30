"""Deterministic query views (no LLM). The raw query is always kept as its own view.

APPS problem statements are long: narrative + Input/Output spec + worked
examples + notes. A 512-token encoder truncates the tail, which is often the
I/O spec and the starter-code signature — the parts that most resemble code.
"""

from __future__ import annotations

import re

_HEADER = re.compile(r"^\s*[-#*=_~`]*\s*([A-Za-z][A-Za-z /]{0,28}?)\s*\d*\s*[:#*=_~`-]*\s*$")

DROP_SECTIONS = {
    "example", "examples", "example input", "example output", "sample input", "sample output",
    "sample", "samples", "explanation", "note", "notes", "subtasks", "for example",
    "sample test", "sample tests", "test cases", "good luck!",
}
KEEP_SECTIONS = {
    "input", "output", "constraints", "input format", "output format", "input/output",
    "task", "problem statement", "problem", "statement", "description",
}
_CODE_FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)
_NUMERIC_LINE = re.compile(r"^[\s\d.,\-+\[\]()]*$")


def _section_name(line: str) -> str | None:
    m = _HEADER.match(line)
    if not m:
        return None
    name = m.group(1).strip().lower()
    return name if name in DROP_SECTIONS or name in KEEP_SECTIONS else None


def core_view(text: str) -> str:
    """Statement + I/O spec + constraints + starter code; drops examples/notes/numeric lines."""
    code_blocks = _CODE_FENCE.findall(text)
    body = _CODE_FENCE.sub("\n", text)
    out: list[str] = []
    dropping = False
    for line in body.split("\n"):
        sec = _section_name(line)
        if sec is not None:
            dropping = sec in DROP_SECTIONS
            if not dropping:
                out.append(line)
            continue
        if dropping or _NUMERIC_LINE.match(line):
            continue
        out.append(line)
    core = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    starter = "\n".join(b.strip() for b in code_blocks if b.strip())
    return f"{core}\n\n{starter}".strip() if starter else core


def starter_code(text: str) -> str:
    """Only the fenced code blocks (e.g. `def solve(...)` / `class Solution`)."""
    return "\n".join(b.strip() for b in _CODE_FENCE.findall(text) if b.strip())


VIEWS = {"raw": lambda t: t, "core": core_view}
