#!/usr/bin/env python3
"""Detach COMMAND into a new session, then exec.

macOS has no `setsid(1)`, and `os.setsid()` fails if this process is already a
process-group leader (the usual fate of `cmd &`). Fork first, then setsid in
the child, then exec.

    DAEMONIZE_PID_FILE=logs/pipeline.pid DAEMONIZE_LOG=logs/pipeline.log \
      .venv/bin/python experiments/gp_tuning_20260902/daemonize.py bash SCRIPT

The parent writes the child's pid and exits. The child is the session leader.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: daemonize.py COMMAND [ARGS...]", file=sys.stderr)
        return 64
    pid_path = os.environ.get("DAEMONIZE_PID_FILE")
    log_path = os.environ.get("DAEMONIZE_LOG")
    child = os.fork()
    if child > 0:
        if pid_path:
            with open(pid_path, "w", encoding="utf-8") as handle:
                handle.write(f"{child}\n")
        else:
            print(child)
        return 0
    os.setsid()
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    if log_path:
        log_fd = os.open(
            log_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        os.close(log_fd)
    os.execvp(sys.argv[1], sys.argv[1:])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
