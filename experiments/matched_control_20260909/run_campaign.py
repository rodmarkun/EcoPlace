"""Use the ordinary campaign CLI with a fail-closed paired initialization barrier."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evolution import run_campaign
from experiments.allbaselines_harness.initialization import read_seed_archive
from experiments.matched_transport_20260909.client import DurableOpenRouterClient

from .runner import InitializationPending, certify_pair, matched_campaign_runner


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pair-receipt', type=Path, required=True)
    parser.add_argument('--certify-pair', type=Path, nargs=2)
    parser.add_argument('--seed-archive', type=Path)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--proposal-journal', type=Path)
    parser.add_argument('--proposal_engine')
    args, remaining = parser.parse_known_args()
    if args.certify_pair:
        if (remaining or args.seed_archive or args.initial
                or args.proposal_journal or args.proposal_engine):
            parser.error('certification accepts only the pair and receipt paths')
        certify_pair(*args.certify_pair, args.pair_receipt)
        return 0
    if args.seed_archive is None or args.initial is None:
        parser.error('campaign requires --seed-archive and --initial')
    archive = read_seed_archive(args.seed_archive, expected_size=12, initial_path=args.initial)
    if args.proposal_journal is not None and args.proposal_engine != 'openrouter':
        parser.error('--proposal-journal requires --proposal_engine openrouter')
    if args.proposal_journal is not None:
        def durable_client(config):
            return DurableOpenRouterClient(config, journal=args.proposal_journal)
        run_campaign.OpenRouterCompletionClient = durable_client
    elif args.proposal_engine == 'openrouter':
        parser.error('corrected OpenRouter campaigns require --proposal-journal')
    if args.proposal_engine is not None:
        remaining = [*remaining, '--proposal_engine', args.proposal_engine]
    run_campaign.CampaignRunner = matched_campaign_runner(archive, args.pair_receipt)
    sys.argv = [sys.argv[0], *remaining, '--initial', str(args.initial)]
    try:
        return run_campaign.main()
    except InitializationPending as exc:
        print(str(exc))
        return 10  # Explicit checkpoint, not campaign completion or engine failure.


if __name__ == '__main__':
    raise SystemExit(main())
