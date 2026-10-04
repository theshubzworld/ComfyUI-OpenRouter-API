import asyncio
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from openrouter_api.cancellation import NodeDeadline, NodeTimeoutError, run_process


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OSError):
        return False
    return True


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_kills_and_reaps_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "pid"
            code = "import os,pathlib,time; pathlib.Path(%r).write_text(str(os.getpid())); time.sleep(30)" % str(pid_path)
            started = time.monotonic()
            with self.assertRaises(NodeTimeoutError):
                await run_process(NodeDeadline(0.25, poll_interval=0.02), sys.executable, "-c", code)
            self.assertLess(time.monotonic() - started, 1.5)
            pid = int(pid_path.read_text())
            self.assertFalse(process_exists(pid), "cancelled child process was left alive")

    async def test_stop_signal_kills_encoder_and_surfaces_as_cancelled(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "pid"
            code = "import os,pathlib,time; pathlib.Path(%r).write_text(str(os.getpid())); time.sleep(30)" % str(pid_path)
            started = time.monotonic()

            def interrupted():
                return time.monotonic() - started > 0.15

            with mock.patch("openrouter_api.cancellation.processing_interrupted", side_effect=interrupted):
                with self.assertRaises(asyncio.CancelledError):
                    await run_process(NodeDeadline(5, poll_interval=0.02), sys.executable, "-c", code)
            self.assertLess(time.monotonic() - started, 1.0)
            pid = int(pid_path.read_text())
            self.assertFalse(process_exists(pid), "interrupted child process was left alive")


if __name__ == "__main__":
    unittest.main()
