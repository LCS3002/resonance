"""Drop-in replacement for anthropic.Anthropic() that routes through the
local Claude Code CLI (`claude -p`).

Unsets ANTHROPIC_API_KEY before spawning the subprocess so the invalid env
key doesn't poison the call. Uses Claude Code's own session auth instead.

Usage:  from .claude_local import local_claude as _claude
        resp = _claude.messages.create(model=..., max_tokens=..., messages=[...], system=...)
        text = resp.content[0].text
"""

from __future__ import annotations

import os
import subprocess
import logging

logger = logging.getLogger(__name__)

_TIMEOUT = 120  # seconds per call


class _Content:
    def __init__(self, text: str):
        self.text = text


class _Response:
    def __init__(self, text: str):
        self.content = [_Content(text)]


class _Messages:
    def create(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        messages: list[dict],
        system: str | None = None,
    ) -> _Response:
        # Build full prompt — prepend system block if provided
        user_content = messages[-1]["content"] if messages else ""
        if system:
            full_prompt = f"[SYSTEM]\n{system}\n[/SYSTEM]\n\n{user_content}"
        else:
            full_prompt = user_content

        # Inherit env but strip the bad API key so Claude Code uses its own auth
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

        cmd = ["claude", "--model", model, "-p", full_prompt]
        logger.debug(f"claude_local -> model={model} prompt_len={len(full_prompt)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            env=env,
        )

        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            logger.error(f"claude_local subprocess error: {err}")
            raise RuntimeError(f"claude CLI error: {err[:200]}")

        return _Response(result.stdout.strip())


class _LocalClaude:
    def __init__(self):
        self.messages = _Messages()


local_claude = _LocalClaude()
