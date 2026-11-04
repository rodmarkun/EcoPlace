"""Durable, single-dispatch OpenRouter transport for the corrected study.

An unresolved dispatch stops the process; no exactly-once provider guarantee is assumed.
Historical transport and archives remain unchanged.
"""
from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from evolution.llm_operator import LLMCompletion
from evolution.openrouter_engine import (
    DispatchFailure,
    EmptyCompletion,
    HTTPStatusFailure,
    OpenRouterCompletionClient,
)
from experiments.gp_tuning_20260902.recovery import canonical, digest


class UnresolvedDispatch(SystemExit):
    """Stop outside proposal retry handlers; provider outcome requires reconciliation."""


class SettledEmptyCompletion(RuntimeError):
    """Billed response without content; cost is settled and the proposal simply failed."""

    def __init__(self, record):
        super().__init__(
            'OpenRouter returned empty completion content (settled): '
            f"finish_reason={record['finish_reason']!r}, "
            f"completion_tokens={record['completion_tokens']}, "
            f"reasoning_tokens={record['reasoning_tokens']}"
        )
        self.cost_microusd = record['cost_microusd']
        self.archive_metadata = {'empty_completion': dict(record)}


class SettledHTTPError(RuntimeError):
    """Definite HTTP error status; its reservation is charged and only the proposal fails."""

    def __init__(self, record):
        super().__init__(f"OpenRouter HTTP error {record['status']} (settled)")
        self.cost_microusd = record['cost_microusd']
        self.status = record['status']
        self.archive_metadata = {'http_error': dict(record)}


class SettledUnknownDispatch(RuntimeError):
    """No provider response; charged the full reservation and only the proposal fails."""

    def __init__(self, record):
        super().__init__(f"OpenRouter dispatch without response (settled): {record['reason']}")
        self.cost_microusd = record['cost_microusd']
        self.archive_metadata = {'unknown_dispatch': dict(record)}


def reconcile_unknown(journal: Path, config, reason: str) -> dict | None:
    """Append a charged settlement for a pending reservation; never edits earlier events.

    Returns the settlement record, or None when the journal has no pending reservation.
    """
    probe = object.__new__(DurableOpenRouterClient)
    probe.journal = Path(journal).resolve()
    probe.config = config
    with probe._locked() as stream:
        events = probe._read(stream)
        last = events[-1]['payload'] if events else None
        if not last or last['type'] != 'reserved':
            return None
        record = {'cost_microusd': last['reservation_microusd'], 'reason': reason}
        probe._append(stream, events, {'type': 'settled_unknown',
                                       'request_sha256': last['request_sha256'],
                                       'settlement': record})
        return {'request_sha256': last['request_sha256'], **record}


# Transport versions whose journals this version continues (additive changes only: new
# settlement event types). A journal keeps the identity it was created with, so request keys
# and archived-response replay are unchanged. dcff0d47...: version at the A+B launch
# (commit 14cb048); P2 amendment 02 of experiments/corrected_ab_20260925.
COMPATIBLE_TRANSPORT_SHA256 = (
    'dcff0d4749ec5c9ab806f154e68143434079f900c6985371c83aa8d2dfdd0505',
)


class DurableOpenRouterClient(OpenRouterCompletionClient):
    def __init__(self, config, *, journal: Path):
        if config.max_attempts != 1:
            raise ValueError('durable transport requires one HTTP attempt per dispatch')
        super().__init__(config)
        self.journal = journal.resolve()
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self.transport_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.running_transport_sha = self.transport_sha
        self.config_sha = digest({'config': dataclasses.asdict(config),
                                  'transport_source_sha256': self.transport_sha})
        self.operator_name = 'openrouter_llm'
        with self._locked() as stream:
            events = self._read(stream)
            if not events:
                self._append(stream, events, {'type': 'configuration',
                                             'config_sha256': self.config_sha,
                                             'schema': 'evolved-durable-openrouter-v1'})
            else:
                recorded = events[0]['payload'].get('config_sha256')
                for predecessor in COMPATIBLE_TRANSPORT_SHA256:
                    candidate = digest({'config': dataclasses.asdict(config),
                                        'transport_source_sha256': predecessor})
                    if recorded != self.config_sha and recorded == candidate:
                        self.transport_sha, self.config_sha = predecessor, candidate
            self.engine_descriptor = {'engine': 'openrouter', 'model_id': config.model_id,
                                      'transport': 'durable-single-dispatch-v1',
                                      'transport_source_sha256': self.transport_sha,
                                      'transport_config_sha256': self.config_sha}
            self._state(events)

    @contextmanager
    def _locked(self):
        # The lock covers dispatch and durable settlement, including replay across processes.
        with self.journal.open('a+b') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield stream
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read(stream):
        stream.seek(0)
        events, previous = [], '0' * 64
        for i, line in enumerate(stream):
            try:
                record = json.loads(line)
                body = {k: v for k, v in record.items() if k != 'sha256'}
                valid = (line == (canonical(record) + '\n').encode()
                         and record['seq'] == i and record['previous_sha256'] == previous
                         and record['sha256'] == digest(body))
            except (ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                raise ValueError('durable transport journal is torn or inconsistent')
            events.append(record)
            previous = record['sha256']
        return events

    def _append(self, stream, events, payload):
        body = {'seq': len(events), 'previous_sha256': events[-1]['sha256'] if events else '0' * 64,
                'payload': payload}
        record = {**body, 'sha256': digest(body)}
        stream.seek(0, os.SEEK_END)
        stream.write((canonical(record) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
        # Persist the directory entry as well as the file contents on first creation.
        directory = os.open(self.journal.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        events.append(record)

    def _state(self, events):
        if not events or events[0]['payload'] != {
                'type': 'configuration', 'config_sha256': self.config_sha,
                'schema': 'evolved-durable-openrouter-v1'}:
            raise ValueError('durable transport configuration differs')
        pending, completed, spent = None, {}, 0
        self._settled_empty = {}
        self._settled_http = {}
        self._settled_unknown = {}
        for event in events[1:]:
            row = event['payload']
            if row['type'] == 'reserved':
                request = row['request']
                key = digest(request)
                if (pending is not None or key in completed or row['request_sha256'] != key
                        or request['config_sha256'] != self.config_sha
                        or row['reservation_microusd'] != self.config.max_call_reservation_microusd
                        or spent + row['reservation_microusd'] > self.config.max_total_microusd):
                    raise ValueError('durable transport reservation differs')
                pending = row
            elif row['type'] == 'completed':
                if pending is None or row['request_sha256'] != pending['request_sha256']:
                    raise ValueError('durable transport completion lacks matching reservation')
                completion = row['completion']
                cost = completion['cost_microusd']
                if (type(cost) is not int or not 0 <= cost <= pending['reservation_microusd']
                        or not isinstance(completion['content'], str)
                        or not completion['content'].strip()):
                    raise ValueError('durable transport completion is invalid')
                completed[row['request_sha256']] = completion
                spent += cost
                pending = None
            elif row['type'] == 'settled_empty':
                if pending is None or row['request_sha256'] != pending['request_sha256']:
                    raise ValueError('durable transport settlement lacks matching reservation')
                record = row['settlement']
                cost = record['cost_microusd']
                if (type(cost) is not int or not 0 <= cost <= pending['reservation_microusd']
                        or set(record) != {'cost_microusd', 'finish_reason', 'prompt_tokens',
                                           'completion_tokens', 'reasoning_tokens'}):
                    raise ValueError('durable transport settlement is invalid')
                self._settled_empty[row['request_sha256']] = record
                spent += cost
                pending = None
            elif row['type'] == 'settled_http_error':
                if pending is None or row['request_sha256'] != pending['request_sha256']:
                    raise ValueError('durable transport settlement lacks matching reservation')
                record = row['settlement']
                if (set(record) != {'status', 'cost_microusd'}
                        or type(record['status']) is not int
                        or record['cost_microusd'] != pending['reservation_microusd']):
                    raise ValueError('durable transport settlement is invalid')
                self._settled_http[row['request_sha256']] = record
                spent += record['cost_microusd']
                pending = None
            elif row['type'] == 'settled_unknown':
                if pending is None or row['request_sha256'] != pending['request_sha256']:
                    raise ValueError('durable transport settlement lacks matching reservation')
                record = row['settlement']
                if (set(record) != {'cost_microusd', 'reason'}
                        or record['cost_microusd'] != pending['reservation_microusd']):
                    raise ValueError('durable transport settlement is invalid')
                self._settled_unknown[row['request_sha256']] = record
                spent += record['cost_microusd']
                pending = None
            else:
                raise ValueError('unknown durable transport event')
        if pending is not None:
            self.ledger.spent_microusd = spent
            self.ledger.reserved_microusd = pending['reservation_microusd']
            raise UnresolvedDispatch(
                'Unresolved OpenRouter dispatch in durable journal; automatic retry refused. '
                'Reconcile provider outcome and accounting before resuming.'
            )
        self.ledger.spent_microusd = spent
        self.ledger.reserved_microusd = 0
        self.ledger._check()
        return completed

    def complete(self, *, model, prompt, generation, seed):
        if (model != self.config.model_id or generation != self.config.generation
                or not isinstance(prompt, str) or type(seed) is not int):
            raise ValueError('completion request differs from pinned config or input types')
        request = {'config_sha256': self.config_sha, 'model': model, 'prompt': prompt,
                   'generation': generation, 'seed': seed}
        key = digest(request)
        with self._locked() as stream:
            events = self._read(stream)
            completed = self._state(events)
            if key in completed:
                return LLMCompletion(**completed[key])
            if key in self._settled_empty:
                raise SettledEmptyCompletion(self._settled_empty[key])
            if key in self._settled_http:
                raise SettledHTTPError(self._settled_http[key])
            if key in self._settled_unknown:
                raise SettledUnknownDispatch(self._settled_unknown[key])
            reservation = self.config.max_call_reservation_microusd
            if self.ledger.spent_microusd + reservation > self.config.max_total_microusd:
                raise ValueError('max_api_costs exceeded')
            self._append(stream, events, {'type': 'reserved', 'request_sha256': key,
                                         'request': request, 'reservation_microusd': reservation})
            try:
                completion = super().complete(model=model, prompt=prompt,
                                              generation=generation, seed=seed)
                self._append(stream, events, {'type': 'completed', 'request_sha256': key,
                                             'completion': dataclasses.asdict(completion)})
            except EmptyCompletion as exc:
                # The provider answered and billed, but produced no final content. This is a
                # known outcome: settle its cost durably and fail only this proposal.
                record = {'cost_microusd': exc.cost_microusd,
                          'finish_reason': exc.finish_reason,
                          'prompt_tokens': exc.prompt_tokens,
                          'completion_tokens': exc.completion_tokens,
                          'reasoning_tokens': exc.reasoning_tokens}
                try:
                    self._append(stream, events, {'type': 'settled_empty',
                                                 'request_sha256': key, 'settlement': record})
                except Exception:
                    raise UnresolvedDispatch(
                        'OpenRouter empty completion could not be settled durably; automatic '
                        'retry refused. Inspect the journal and reconcile the provider outcome.'
                    ) from None
                self._state(events)
                raise SettledEmptyCompletion(record) from None
            except DispatchFailure:
                # No provider response (network failure or timeout). The outcome is unknown,
                # so charge the whole reservation and fail only this proposal.
                record = {'cost_microusd': reservation, 'reason': 'no provider response'}
                try:
                    self._append(stream, events, {'type': 'settled_unknown',
                                                 'request_sha256': key, 'settlement': record})
                except Exception:
                    raise UnresolvedDispatch(
                        'OpenRouter dispatch failure could not be settled durably; automatic '
                        'retry refused. Inspect the journal and reconcile the provider outcome.'
                    ) from None
                self._state(events)
                raise SettledUnknownDispatch(record) from None
            except HTTPStatusFailure as exc:
                # A definite error status is a known provider outcome. Charge the whole
                # reservation (billing is not assumed to be zero) and fail only this proposal.
                record = {'status': exc.status, 'cost_microusd': reservation}
                try:
                    self._append(stream, events, {'type': 'settled_http_error',
                                                 'request_sha256': key, 'settlement': record})
                except Exception:
                    raise UnresolvedDispatch(
                        'OpenRouter HTTP error could not be settled durably; automatic retry '
                        'refused. Inspect the journal and reconcile the provider outcome.'
                    ) from None
                self._state(events)
                raise SettledHTTPError(record) from None
            except Exception:
                # The provider may have accepted the call even when the local response failed.
                # SystemExit bypasses operator retry/repair handlers that catch Exception.
                raise UnresolvedDispatch(
                    'OpenRouter dispatch did not reach durable completion; automatic retry '
                    'refused. Inspect the journal and reconcile the provider outcome.'
                ) from None
            self._state(events)
            return completion
