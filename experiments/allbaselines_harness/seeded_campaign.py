"""Campaign CLI shim for the harness-local fixed initialization archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evolution import run_campaign

from .initialization import read_seed_archive, seeded_campaign_runner


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed-archive", type=Path, required=True)
    parser.add_argument("--seed-archive-size", type=int, required=True)
    # The seed archive's immutable shell is the *run's* evolution seed, not the module
    # default: the dev-continuum family starts from evolution/task/initial_tight.py.
    # Consume it here to validate the archive, then hand it straight back to the driver.
    parser.add_argument("--initial", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    archive = read_seed_archive(
        args.seed_archive,
        expected_size=args.seed_archive_size,
        initial_path=args.initial,
    )
    remaining = [*remaining, "--initial", str(args.initial)]
    run_campaign.CampaignRunner = seeded_campaign_runner(archive)
    sys.argv = [sys.argv[0], *remaining]
    return run_campaign.main()


if __name__ == "__main__":
    raise SystemExit(main())
