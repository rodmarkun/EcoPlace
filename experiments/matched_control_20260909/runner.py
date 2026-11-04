"""Sequential campaign seam with a paired initialization barrier.

This is an execution control, not a scientific registration or launch authorization.
The ordinary campaign CLI retains its scientific gates and cost controls.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evolution.campaign import EventArchive
from experiments.allbaselines_harness.initialization import seeded_campaign_runner
from experiments.gp_tuning_20260902.recovery import canonical, digest, read_sealed

from .control import SequentialControl

SCHEMA = 'evolved-continuum-sequential-initialization-v1'


class InitializationPending(RuntimeError):
    """Both arms must reach the barrier and be certified before adaptive proposals."""


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _normalize_ids(value: Any, ids: dict[str, int]) -> Any:
    if isinstance(value, str) and value in ids:
        return {'candidate_seq': ids[value]}
    if isinstance(value, dict):
        return {key: _normalize_ids(item, ids) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_ids(item, ids) for item in value]
    return value


def initialization_receipt(archive: Path) -> dict:
    """Verify only the immutable initialization prefix; later appends are permitted."""
    manifest = json.loads((archive / 'manifest.json').read_text())
    context = manifest['provenance'].get('matched_sequential_control')
    if not isinstance(context, dict) or context.get('schema') != SCHEMA:
        raise ValueError('archive does not carry the sequential control')
    size = context['initialization_size']
    if size != 12 or context['plan']['proposals_per_call'] != 1:
        raise ValueError('invalid sequential control configuration')
    rows, events = [], []
    previous = '0' * 64
    with (archive / 'events.jsonl').open() as stream:
        for index, line in enumerate(stream):
            event = json.loads(line)
            body = {key: value for key, value in event.items() if key != 'event_sha256'}
            if (line != canonical(event) + '\n' or digest(body) != event['event_sha256']
                    or event['event_seq'] != index
                    or event['previous_event_sha256'] != previous):
                raise ValueError('initialization event chain mismatch')
            events.append(event)
            previous = event['event_sha256']
            payload = event['payload']
            if event['type'] not in {'candidate_created', 'candidate_completed'}:
                raise ValueError('unexpected event before initialization barrier')
            if payload['proposal_seq'] >= size or payload.get('shared_call_id') is not None:
                raise ValueError('adaptive or batched allocation before initialization barrier')
            if event['type'] == 'candidate_completed':
                rows.append(payload)
                if len(rows) == size:
                    break
    EventArchive._validate_events(events)
    if len(rows) != size:
        raise ValueError('paired initialization is incomplete')
    ids = {row['candidate_id']: row['proposal_seq'] for row in rows}
    normalized = []
    for seq, row in enumerate(rows):
        metadata = row['proposal']['metadata']
        if (row['proposal_seq'] != seq or metadata.get('phase') != 'initialization'
                or metadata.get('seed_archive_index') != seq
                or metadata.get('seed_archive_payload_sha256') != context['seed_archive_sha256']):
            raise ValueError('initialization source provenance differs')
        source_hash = hashlib.sha256(row['proposal']['source'].encode()).hexdigest()
        if source_hash != context['source_sha256'][seq]:
            raise ValueError('initialization source differs from declared archive')
        evaluation = (row.get('evaluation') or {}).get('evaluation')
        normalized.append(_normalize_ids({
            'proposal_seq': seq, 'source_sha256': source_hash,
            'island_id': row['island_id'], 'island_index': row['island_index'],
            'fitness': None if evaluation is None else evaluation['fitness'],
            'instance_metrics': [] if evaluation is None else evaluation['instances'],
            'evaluation': row['evaluation'], 'archive_assessment': row['archive_assessment'],
            'decision': row['decision'], 'evaluator_work': row['evaluator_work'],
            'parent_id': row['parent_id'], 'inspiration_ids': row['inspiration_ids'],
            'selection_seed': row['selection_seed'],
        }, ids))
    effective_campaign = {key: value for key, value in manifest['campaign'].items()
                          if key not in {'arm', 'operator', 'proposal_engine'}}
    return {'archive': str(archive.resolve()), 'manifest_sha256': digest(manifest),
            'arm': manifest['campaign']['arm'],
            'context': {'declared': context, 'effective_campaign': effective_campaign},
            'prefix_sha256': previous, 'records': normalized}


def certify_pair(left: Path, right: Path, output: Path) -> dict:
    """Create an exclusive, deterministic receipt after both arms reach the barrier."""
    receipts = sorted([initialization_receipt(left), initialization_receipt(right)],
                      key=lambda item: item['arm'])
    if [item['arm'] for item in receipts] != ['gp', 'llm']:
        raise ValueError('pair must contain exactly one GP and one LLM arm')
    a, b = receipts
    if a['context'] != b['context']:
        raise ValueError('paired effective configurations differ')
    SequentialControl().validate_initialization(a['records'], b['records'])
    if a['records'] != b['records']:
        raise ValueError('paired initialization evaluation, selection, or accounting differs')
    payload = {'schema': SCHEMA, 'boundary': 'Execution parity only; not a preregistration.',
               'arms': receipts}
    envelope = canonical({'payload': payload, 'payload_sha256': digest(payload)}) + '\n'
    # Exclusive publication refuses to overwrite an earlier study receipt.
    try:
        with output.open('x') as stream:
            stream.write(envelope)
    except FileExistsError:
        if output.read_text() != envelope:
            raise ValueError('different paired receipt already exists') from None
    return payload


def verify_pair(receipt_path: Path, archive: Path) -> None:
    payload = read_sealed(receipt_path)
    if payload.get('schema') != SCHEMA or len(payload.get('arms', [])) != 2:
        raise ValueError('invalid paired initialization receipt')
    receipts = payload['arms']
    if [item['arm'] for item in receipts] != ['gp', 'llm']:
        raise ValueError('invalid paired arms')
    if str(archive.resolve()) not in {item['archive'] for item in receipts}:
        raise ValueError('paired receipt does not cover this archive')
    if receipts[0]['context'] != receipts[1]['context']:
        raise ValueError('paired effective configurations differ')
    SequentialControl().validate_initialization(receipts[0]['records'], receipts[1]['records'])
    if receipts[0]['records'] != receipts[1]['records']:
        raise ValueError('paired initialization records differ')
    for expected in receipts:
        if initialization_receipt(Path(expected['archive'])) != expected:
            raise ValueError('paired initialization changed after certification')


def matched_campaign_runner(seed_archive: dict, receipt_path: Path):
    base = seeded_campaign_runner(seed_archive)

    class MatchedCampaignRunner(base):
        def __init__(self, **kwargs):
            control = SequentialControl()
            control.validate_plan(kwargs['plan'])
            if len(seed_archive['sources']) != control.initialization_size:
                raise ValueError('sequential study requires exactly 12 initialization policies')
            plan = kwargs['plan']
            if (plan.candidate_budget <= control.initialization_size
                    or plan.evaluator_work.max_microwork
                    < (control.initialization_size + 1)
                    * plan.evaluator_work.max_candidate_microwork):
                raise ValueError('budget must leave work for an adaptive proposal')
            provenance = dict(kwargs['provenance'])
            provenance['matched_sequential_control'] = _jsonable({
                'schema': SCHEMA, 'initialization_size': control.initialization_size,
                'seed_archive_sha256': seed_archive['payload_sha256'],
                'source_sha256': [hashlib.sha256(s.encode()).hexdigest()
                                  for s in seed_archive['sources']],
                'definition': kwargs['definition'], 'plan': plan, 'task': kwargs['task'],
                'ladder': kwargs['ladder'], 'islands': kwargs['islands'],
                'effective_constraints': {
                    profile.island_id: profile.constraints(kwargs['task'].constraints)
                    for profile in kwargs['islands'].profiles
                },
                'feedback_profile': kwargs['feedback_profile'],
                'run_purpose': kwargs.get('run_purpose', 'scientific-full-work'),
                'evaluation_seed_mode': kwargs.get('evaluation_seed_mode', 'campaign-derived'),
                'terminal_budget_policy': kwargs.get(
                    'terminal_budget_policy', 'carry-to-work-limit'
                ),
            })
            kwargs['provenance'] = provenance
            super().__init__(**kwargs)

        def _check_barrier(self, seq):
            if seq >= 12:
                if not receipt_path.exists():
                    raise InitializationPending('initialization complete; paired receipt required')
                verify_pair(receipt_path, self.archive.root)

        def _allocate(self, proposal_seq):
            self._check_barrier(proposal_seq)
            return super()._allocate(proposal_seq)

        def _proposal(self, created, *, initial_offspring=None):
            self._check_barrier(created['proposal_seq'])
            return super()._proposal(created, initial_offspring=initial_offspring)

    return MatchedCampaignRunner
