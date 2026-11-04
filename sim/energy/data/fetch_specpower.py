"""Re-pin SPECpower_ssj2008 entries from spec.org (TODO 3.1; plan §4.1).

Build-time tool, **never** imported by the simulator (the simulator stays free of
network dependencies — web access happens only here, at pinning time). It downloads
the pinned result pages, parses the published performance/power table into joint
(target load, ssj_ops, average active power) tuples per machine, attaches a full
provenance record (URL, access date, content sha256, exact fields transcribed), and
writes ``specpower.json`` next to this script.

Run: ``uv run python sim/energy/data/fetch_specpower.py``

The pinned machine list below is the complete enumeration; adding a machine means
adding its result URL here and re-running (the JSON is never hand-edited).
"""

from __future__ import annotations

import datetime
import hashlib
import html as html_mod
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "specpower.json"
BASE = "https://www.spec.org/power_ssj2008/results/"

# machine_id -> result page (relative to BASE). Every entry is a complete published
# SPECpower_ssj2008 report; multi-node reports are rejected by the parser below
# (whole-machine joint tuples must describe one physical machine).
PINNED = {
    "specpower-lenovo-sr665v3-2023": "res2023q4/power_ssj2008-20230926-01323.html",
    "specpower-supermicro-as1115sv-2023": "res2023q4/power_ssj2008-20230912-01312.html",
    "specpower-fujitsu-rx100s7-2011": "res2011q2/power_ssj2008-20110601-00380.html",
    "specpower-asus-esc4000e11-2023": "res2023q4/power_ssj2008-20231024-01328.html",
    "specpower-h3c-r6900g6-2023": "res2023q4/power_ssj2008-20230831-01311.html",
    "specpower-ibm-x3550m3-2011": "res2011q2/power_ssj2008-20110406-00368.html",
}


def _rows(text: str) -> list[list[str]]:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S)
    out = []
    for r in rows:
        cells = [
            html_mod.unescape(re.sub(r"<[^>]+>", "", c)).strip()
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)
        ]
        if cells and any(cells):
            out.append(cells)
    return out


def _num(s: str) -> float:
    return float(s.replace(",", "").replace("%", ""))


def parse_result_page(text: str, url: str, accessed: str) -> dict:
    rows = _rows(text)
    kv: dict[str, str] = {}
    for c in rows:
        for i in range(0, len(c) - 1, 2):
            if isinstance(c[i], str) and c[i].endswith(":"):
                kv.setdefault(c[i], c[i + 1])
    if kv.get("Test Method:") != "Single Node":
        raise ValueError(
            f"{url}: Test Method is {kv.get('Test Method:')!r}; only Single Node reports "
            "are accepted (a whole-machine tuple must describe one physical machine)"
        )
    title, metric = rows[0][0], rows[0][1]
    m = re.search(r"SPECpower_ssj2008\s*=\s*([\d,]+)", metric)
    if not m:
        raise ValueError(f"{url}: cannot parse overall metric from {metric!r}")
    load_levels = []
    active_idle_w = None
    for c in rows:
        # main table rows: [target, actual, ssj_ops, power, (ratio)]
        if len(c) >= 4 and re.match(r"^\d+%$", c[0]) and "," in c[2] and "%" in c[1]:
            load_levels.append(
                {
                    "target_load_pct": _num(c[0]),
                    "actual_load_pct": _num(c[1]),
                    "ssj_ops": _num(c[2]),
                    "avg_active_power_w": _num(c[3]),
                }
            )
        if c[0] == "Active Idle" and len(c) == 4 and active_idle_w is None:
            # ['Active Idle', ssj_ops(=0), power, ratio(=0)]
            active_idle_w = _num(c[2])
    load_levels = load_levels[:10]
    if len(load_levels) != 10 or active_idle_w is None:
        raise ValueError(f"{url}: expected 10 load levels + Active Idle")
    return {
        "title": title,
        "vendor": kv.get("Hardware Vendor:", title.split()[0]),
        "model": kv.get("Model:", title),
        "cpu": kv.get("CPU Name:"),
        "cpu_enabled": kv.get("CPU(s) Enabled:"),
        "memory_gb": kv.get("Memory Amount (GB):"),
        "test_method": kv.get("Test Method:"),
        "test_date": kv.get("Test Date:"),
        "hardware_availability": kv.get("Hardware Availability:"),
        "publication": kv.get("Publication:"),
        "overall_ssj_ops_per_watt": _num(m.group(1)),
        "load_levels": load_levels,
        "active_idle_w": active_idle_w,
        "provenance": {
            "source_name": "SPECpower_ssj2008 published result (SPEC)",
            "url": url,
            "doi": None,
            "version_or_date_accessed": accessed,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "field_transcribed": (
                "Performance/Power table: Target Load, Actual Load, ssj_ops, "
                "Average Active Power (W); Active Idle power; overall ssj_ops/watt; "
                "hardware summary fields"
            ),
            "status": "VERIFIED",
            "verification_method": (
                "downloaded from spec.org and machine-parsed by "
                "sim/energy/data/fetch_specpower.py (no hand transcription)"
            ),
        },
    }


def main() -> None:
    accessed = datetime.datetime.now(datetime.UTC).date().isoformat()
    entries = {}
    for machine_id, rel in sorted(PINNED.items()):
        url = BASE + rel
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
            text = resp.read().decode("utf-8", errors="replace")
        entries[machine_id] = parse_result_page(text, url, accessed)
        print(f"pinned {machine_id}: {entries[machine_id]['title']}")
    payload = {
        "schema": "specpower-joint-tuples-v1",
        "note": (
            "Joint (throughput, power) tuples per published machine; SPECpower load "
            "levels are consumed jointly, never as independent curves (plan §4.1). "
            "Generated by fetch_specpower.py; do not hand-edit."
        ),
        "entries": entries,
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({len(entries)} machines)")


if __name__ == "__main__":
    main()
