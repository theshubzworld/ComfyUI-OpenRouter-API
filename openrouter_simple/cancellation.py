from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class NodeTimeoutError(RuntimeError):
    """Raised when the user-selected node deadline expires."""


def _model_management():
    try:
        import comfy.model_management as model_management
    except ImportError:
        return None
    return model_management


def processing_interrupted() -> bool:
    model_management = _model_management()
    return bool(model_management and model_management.processing_interrupted())


def raise_native_interrupt() -> None:
    model_management = _model_management()
    if model_management is not None:
        model_management.throw_exception_if_processing_interrupted()
    raise asyncio.CancelledError("ComfyUI execution interrupted")


class NodeDeadline:
    """One monotonic deadline shared by preprocessing and the paid chat request."""

    def __init__(self, timeout_seconds: float, *, poll_interval: float = 0.1):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.started = time.monotonic()
        self.expires_at = self.started + float(timeout_seconds)
        self.poll_interval = max(0.02, min(float(poll_interval), 0.25))

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    def checkpoint(self) -> None:
        if processing_interrupted():
            raise_native_interrupt()
        if self.remaining <= 0:
            raise NodeTimeoutError("OpenRouter node deadline expired")

    async def _watch(self) -> str:
        while True:
            if processing_interrupted():
                return "interrupted"
            remaining = self.remaining
            if remaining <= 0:
                return "timeout"
            await asyncio.sleep(min(self.poll_interval, remaining))

    async def run(
        self,
        awaitable: Awaitable[T],
        *,
        on_cancel: Callable[[], Awaitable[None]] | None = None,
    ) -> T:
        """Race work against Stop and deadline, then fully settle cancelled work."""

        self.checkpoint()
        work = asyncio.ensure_future(awaitable)
        watcher = asyncio.create_task(self._watch())
        try:
            done, _ = await asyncio.wait({work, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if work in done:
                return work.result()

            reason = watcher.result()
            work.cancel()
            if on_cancel is not None:
                with contextlib.suppress(Exception):
                    await on_cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await work
            if reason == "interrupted":
                raise_native_interrupt()
            raise NodeTimeoutError("OpenRouter node deadline expired")
        except asyncio.CancelledError:
            work.cancel()
            if on_cancel is not None:
                with contextlib.suppress(Exception):
                    await on_cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await work
            raise
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher


async def terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=0.75)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=0.75)


async def run_process(
    deadline: NodeDeadline,
    *args: str,
    stdin: int | None = None,
    stdout: int | None = asyncio.subprocess.PIPE,
    stderr: int | None = asyncio.subprocess.PIPE,
) -> tuple[bytes, bytes]:
    """Run a child process with bounded Stop/timeout cleanup and capped diagnostics."""

    deadline.checkpoint()
    process = await asyncio.create_subprocess_exec(
        *args,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )
    out, err = await deadline.run(process.communicate(), on_cancel=lambda: terminate_process(process))
    if process.returncode:
        detail = (err or out or b"").decode("utf-8", errors="replace")[-2000:].strip()
        raise RuntimeError(f"Media encoder failed (exit {process.returncode}): {detail}")
    return out or b"", err or b""
