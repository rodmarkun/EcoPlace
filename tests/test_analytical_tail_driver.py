"""A denied monitor must leave no runaway tail simulator or silent retry."""

from __future__ import annotations

import pytest

from experiments.gp_tuning_20260902.recovery import read_sealed
from experiments.validation_20260909 import run_analytical_tails as driver


def test_monitor_failure_reaps_owned_child_and_preserves_failed_attempt(tmp_path, monkeypatch):
    class Child:
        pid = 12345
        killed = False
        waited = False

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True

        def wait(self):
            self.waited = True
            return -9

    child = Child()
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **kw: child)

    def denied(*args):
        raise PermissionError("test monitor unavailable")

    monkeypatch.setattr(driver.psutil, "Process", denied)
    with pytest.raises(PermissionError):
        driver.guarded(tmp_path, {}, {}, 600.0)
    assert child.killed and child.waited
    usage = read_sealed(next((tmp_path / "runs").glob("*.usage.json")))
    assert "PermissionError" in usage["stop_reason"]
    with pytest.raises(ValueError, match="no automatic retry"):
        driver.guarded(tmp_path, {}, {}, 600.0)
