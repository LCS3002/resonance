"""Prompt version loader — always returns the latest vN.txt for a given stage.

File format (api/prompts/<stage>/v1.txt):

    ## SYSTEM
    <system prompt content>

    ## USER
    <user prompt template — use $variable placeholders>

Usage:
    system, user = load_prompt("s1_concept")
    system, user = render("s1_concept", brand_name="Pocket", target_emotion="aspirational", ...)
"""
from __future__ import annotations

import re
from pathlib import Path
from string import Template

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(stage: str) -> tuple[str, str]:
    """Return (system, user_template) for the latest version of *stage*.

    user_template contains $variable placeholders (string.Template style).
    Raises FileNotFoundError if no prompt files exist for the stage.
    """
    stage_dir = _PROMPTS_DIR / stage
    files = sorted(stage_dir.glob("v*.txt"), key=_version_key)
    if not files:
        raise FileNotFoundError(
            f"No prompt files for stage '{stage}' in {stage_dir}. "
            f"Create {stage_dir}/v1.txt with ## SYSTEM and ## USER sections."
        )
    return _parse(files[-1])


def render(stage: str, **kwargs: str) -> tuple[str, str]:
    """Load the latest prompt for *stage* and substitute template variables.

    Returns (system, user_content) ready to pass to _claude.messages.create().
    Unknown $variables are left as-is (safe_substitute).
    """
    system, user_tmpl = load_prompt(stage)
    return system, Template(user_tmpl).safe_substitute(**kwargs)


def latest_version(stage: str) -> str:
    """Return the filename of the latest prompt version for *stage*."""
    stage_dir = _PROMPTS_DIR / stage
    files = sorted(stage_dir.glob("v*.txt"), key=_version_key)
    if not files:
        raise FileNotFoundError(f"No prompt files for stage '{stage}'")
    return files[-1].name


# ── Internals ──────────────────────────────────────────────────────────────────

def _version_key(path: Path) -> int:
    m = re.search(r"v(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def _parse(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    parts: dict[str, list[str]] = {"SYSTEM": [], "USER": []}
    current: str | None = None
    for line in text.splitlines(keepends=True):
        if line.strip() == "## SYSTEM":
            current = "SYSTEM"
        elif line.strip() == "## USER":
            current = "USER"
        elif current is not None:
            parts[current].append(line)
    return (
        "".join(parts["SYSTEM"]).strip(),
        "".join(parts["USER"]).strip(),
    )
