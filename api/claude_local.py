"""Drop-in replacement for anthropic.Anthropic() that routes through the
local Claude Code CLI (`claude -p`).

Unsets ANTHROPIC_API_KEY so the invalid env key doesn't poison the call.
Uses Claude Code's own session auth instead.

Two interfaces:
  _claude.messages.create(...)   — sync, for LangGraph nodes (run in thread executors)
  _claude.messages.acreate(...)  — async, for FastAPI route handlers and SSE generators
"""

from __future__ import annotations

import asyncio
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


def _build_prompt(messages: list[dict], system: str | None) -> str:
    user_content = messages[-1]["content"] if messages else ""
    if system:
        return f"[SYSTEM]\n{system}\n[/SYSTEM]\n\n{user_content}"
    return user_content


def _clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}


class _Messages:
    def create(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        messages: list[dict],
        system: str | None = None,
    ) -> _Response:
        """Synchronous call — use in LangGraph nodes (run via run_in_executor)."""
        full_prompt = _build_prompt(messages, system)
        logger.debug(f"claude_local.create model={model} len={len(full_prompt)}")

        result = subprocess.run(
            ["claude", "--model", model, "-p", full_prompt],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            env=_clean_env(),
        )

        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            logger.error(f"claude_local subprocess error: {err}")
            raise RuntimeError(f"claude CLI error: {err[:200]}")

        return _Response(result.stdout.strip())

    async def acreate(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        messages: list[dict],
        system: str | None = None,
    ) -> _Response:
        """Async call — use in FastAPI handlers and SSE generators.

        Uses asyncio.create_subprocess_exec so it never blocks the event loop,
        allowing SSE chunks to flush between awaits.
        """
        full_prompt = _build_prompt(messages, system)
        logger.debug(f"claude_local.acreate model={model} len={len(full_prompt)}")

        proc = await asyncio.create_subprocess_exec(
            "claude", "--model", model, "-p", full_prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_clean_env(),
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"claude CLI timed out after {_TIMEOUT}s")

        if proc.returncode != 0:
            err = stderr.decode().strip() or stdout.decode().strip()
            logger.error(f"claude_local async subprocess error: {err}")
            raise RuntimeError(f"claude CLI error: {err[:200]}")

        return _Response(stdout.decode().strip())


class _LocalClaude:
    def __init__(self):
        self.messages = _Messages()


local_claude = _LocalClaude()
