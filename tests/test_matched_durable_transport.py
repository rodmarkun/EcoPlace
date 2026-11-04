"""Exercise the real completion parser/accounting with deterministic fake transport."""
from __future__ import annotations

import dataclasses
import json
import os
import ssl
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from evolution.openrouter_engine import load_openrouter_engine_config
from experiments.matched_transport_20260909.client import (
    DurableOpenRouterClient,
    SettledEmptyCompletion,
    SettledHTTPError,
    SettledUnknownDispatch,
    UnresolvedDispatch,
    reconcile_unknown,
)

ROOT = Path(__file__).resolve().parents[1]


def config():
    base = load_openrouter_engine_config(ROOT / 'evolution/openrouter_engine.json')
    return dataclasses.replace(base, max_attempts=1)


def response(client):
    return {'model': client.config.model_id,
            'choices': [{'message': {'content': 'valid fake completion'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20}}


def complete(client, seed=1):
    return client.complete(model=client.config.model_id, prompt='pilot request',
                           generation=client.config.generation, seed=seed)


def test_response_replays_without_new_dispatch_or_double_charge(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'local-pilot-only')
    calls = []
    client = DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')
    monkeypatch.setattr(client, '_dispatch', lambda request: calls.append(1) or response(client))
    first = complete(client)
    assert len(calls) == 1
    assert client.ledger.spent_microusd == first.cost_microusd > 0
    restarted = DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')
    monkeypatch.setattr(restarted, '_dispatch', lambda request: pytest.fail('unexpected dispatch'))
    assert complete(restarted) == first
    assert restarted.ledger.spent_microusd == first.cost_microusd
    assert 'local-pilot-only' not in (tmp_path / 'journal.jsonl').read_text()


def test_uncertain_dispatch_stops_outside_operator_retry_handlers(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'local-pilot-only')
    client = DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')

    def fail(request):
        raise RuntimeError('connection lost after provider might have accepted request')

    monkeypatch.setattr(client, '_dispatch', fail)
    with pytest.raises(UnresolvedDispatch):
        complete(client)
    assert not issubclass(UnresolvedDispatch, Exception)
    with pytest.raises(UnresolvedDispatch):
        DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')


def test_actual_process_exit_after_dispatch_leaves_non_retryable_intent(tmp_path):
    journal = tmp_path / 'journal.jsonl'
    script = '''
import dataclasses, os, sys
from pathlib import Path
from evolution.openrouter_engine import load_openrouter_engine_config
from experiments.matched_transport_20260909.client import DurableOpenRouterClient
config = dataclasses.replace(load_openrouter_engine_config('evolution/openrouter_engine.json'),
                             max_attempts=1)
os.environ['OPENROUTER_API_KEY'] = 'local-pilot-only'
client = DurableOpenRouterClient(config, journal=Path(sys.argv[1]))
def interrupt(request):
    os._exit(37)
client._dispatch = interrupt
client.complete(model=config.model_id, prompt='interruption pilot',
                generation=config.generation, seed=1)
'''
    run = subprocess.run([sys.executable, '-c', script, str(journal)], cwd=ROOT, timeout=30,
                         capture_output=True)
    assert run.returncode == 37, run.stderr.decode()
    with pytest.raises(UnresolvedDispatch):
        DurableOpenRouterClient(config(), journal=journal)
    assert len(journal.read_text().splitlines()) == 2  # Configuration and durable intent.


def test_transport_retries_and_changed_price_config_are_refused(tmp_path):
    cfg = config()
    with pytest.raises(ValueError, match='one HTTP attempt'):
        DurableOpenRouterClient(dataclasses.replace(cfg, max_attempts=2),
                                journal=tmp_path / 'retry.jsonl')
    DurableOpenRouterClient(cfg, journal=tmp_path / 'journal.jsonl')
    with pytest.raises(ValueError, match='configuration differs'):
        DurableOpenRouterClient(dataclasses.replace(cfg, prompt_microusd_per_million=1),
                                journal=tmp_path / 'journal.jsonl')


def test_cost_cap_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'local-pilot-only')
    cfg = dataclasses.replace(config(), max_total_microusd=100, max_call_reservation_microusd=100,
                              prompt_microusd_per_million=1_000_000,
                              completion_microusd_per_million=1_000_000)
    client = DurableOpenRouterClient(cfg, journal=tmp_path / 'journal.jsonl')
    monkeypatch.setattr(client, '_dispatch', lambda request: response(client))
    first = complete(client)
    assert first.cost_microusd == 30
    restarted = DurableOpenRouterClient(cfg, journal=tmp_path / 'journal.jsonl')
    monkeypatch.setattr(restarted, '_dispatch', lambda request: pytest.fail('over-budget dispatch'))
    with pytest.raises(ValueError, match='max_api_costs'):
        complete(restarted, seed=2)
    assert complete(restarted) == first  # A previously paid response remains replayable.


def test_torn_or_modified_journal_is_not_silently_recovered(tmp_path):
    journal = tmp_path / 'journal.jsonl'
    DurableOpenRouterClient(config(), journal=journal)
    original = journal.read_text()
    journal.write_text(original + '{"partial":')
    with pytest.raises(ValueError, match='torn or inconsistent'):
        DurableOpenRouterClient(config(), journal=journal)
    row = json.loads(original)
    row['payload']['config_sha256'] = 'f' * 64
    journal.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='torn or inconsistent'):
        DurableOpenRouterClient(config(), journal=journal)


def test_response_received_but_not_persisted_cannot_be_automatically_retried(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'local-pilot-only')
    journal = tmp_path / 'journal.jsonl'
    client = DurableOpenRouterClient(config(), journal=journal)
    monkeypatch.setattr(client, '_dispatch', lambda request: response(client))
    append = client._append

    def fail_completion(stream, events, payload):
        if payload['type'] == 'completed':
            raise OSError('simulated response persistence failure')
        append(stream, events, payload)

    monkeypatch.setattr(client, '_append', fail_completion)
    with pytest.raises(UnresolvedDispatch):
        complete(client)
    with pytest.raises(UnresolvedDispatch):
        DurableOpenRouterClient(config(), journal=journal)


def test_concurrent_client_cannot_dispatch_under_another_clients_lock(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'local-pilot-only')
    journal = tmp_path / 'journal.jsonl'
    client = DurableOpenRouterClient(config(), journal=journal)

    def dispatch(request):
        with pytest.raises(BlockingIOError):
            DurableOpenRouterClient(config(), journal=journal)
        return response(client)

    monkeypatch.setattr(client, '_dispatch', dispatch)
    assert complete(client).cost_microusd > 0


def test_https_endpoint_and_subprocess_restart_replay_exactly_one_request(tmp_path):
    key, cert = tmp_path / 'key.pem', tmp_path / 'cert.pem'
    tls_config = tmp_path / 'tls.cnf'
    tls_config.write_text(
        '[req]\ndistinguished_name=dn\nx509_extensions=ext\nprompt=no\n'
        '[dn]\nCN=127.0.0.1\n[ext]\nsubjectAltName=IP:127.0.0.1\n'
        'basicConstraints=critical,CA:TRUE\n'
    )
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', str(key), '-out', str(cert), '-days', '1',
                    '-config', str(tls_config)], check=True, capture_output=True, timeout=30)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == '/api/v1/chat/completions'
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(request)
            raw = {'model': request['model'],
                   'choices': [{'message': {'content': 'persist this exact fake response'},
                                'finish_reason': 'stop'}],
                   'usage': {'prompt_tokens': 10, 'completion_tokens': 20}}
            body = json.dumps(raw).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cfg = dataclasses.replace(config(), endpoint=f'https://127.0.0.1:{server.server_port}/api/v1')
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(dataclasses.asdict(cfg)))
    env = {**os.environ, 'OPENROUTER_API_KEY': 'local-pilot-only', 'SSL_CERT_FILE': str(cert),
           'NO_PROXY': '127.0.0.1', 'no_proxy': '127.0.0.1'}
    script = '''
import dataclasses, json, sys
from pathlib import Path
from evolution.openrouter_engine import load_openrouter_engine_config
from experiments.matched_transport_20260909.client import DurableOpenRouterClient
config = load_openrouter_engine_config(sys.argv[1])
client = DurableOpenRouterClient(config, journal=Path(sys.argv[2]))
result = client.complete(model=config.model_id, prompt='local TLS pilot',
                         generation=config.generation, seed=123)
print(json.dumps({'completion': dataclasses.asdict(result), 'ledger': client.ledger.to_dict()}))
'''
    outputs = []
    try:
        for _ in range(2):
            run = subprocess.run([sys.executable, '-c', script, str(config_path),
                                  str(tmp_path / 'journal.jsonl')], cwd=ROOT, env=env,
                                 check=True, capture_output=True, text=True, timeout=30)
            outputs.append(json.loads(run.stdout))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert len(requests) == 1
    assert outputs[0] == outputs[1]
    assert outputs[0]['ledger']['spent_microusd'] == outputs[0]['completion']['cost_microusd'] > 0


def reasoning_only(client):
    return {'model': client.config.model_id,
            'choices': [{'message': {'content': None}, 'finish_reason': 'length'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 30,
                      'completion_tokens_details': {'reasoning_tokens': 30}}}


def test_billed_empty_completion_settles_and_replays_without_blocking(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'local-pilot-only')
    calls = []
    client = DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')
    monkeypatch.setattr(client, '_dispatch',
                        lambda request: calls.append(1) or reasoning_only(client))
    with pytest.raises(SettledEmptyCompletion) as caught:
        complete(client)
    # An ordinary Exception: operator handlers record a failed proposal instead of halting.
    assert isinstance(caught.value, Exception)
    cost = caught.value.cost_microusd
    assert 0 < cost == client.ledger.spent_microusd
    assert client.ledger.reserved_microusd == 0
    assert caught.value.archive_metadata['empty_completion']['finish_reason'] == 'length'

    restarted = DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')
    monkeypatch.setattr(restarted, '_dispatch', lambda request: pytest.fail('unexpected dispatch'))
    with pytest.raises(SettledEmptyCompletion):
        complete(restarted)
    assert restarted.ledger.spent_microusd == cost
    monkeypatch.setattr(restarted, '_dispatch',
                        lambda request: calls.append(1) or response(restarted))
    later = complete(restarted, seed=2)
    assert len(calls) == 2
    assert restarted.ledger.spent_microusd == cost + later.cost_microusd
    kinds = [json.loads(line)['payload']['type']
             for line in (tmp_path / 'journal.jsonl').read_text().splitlines()]
    assert kinds == ['configuration', 'reserved', 'settled_empty', 'reserved', 'completed']



def test_definite_http_error_settles_at_full_reservation_and_replays(tmp_path, monkeypatch):
    from evolution.openrouter_engine import HTTPStatusFailure

    monkeypatch.setenv('OPENROUTER_API_KEY', 'local-pilot-only')
    client = DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')

    def refuse(request):
        raise HTTPStatusFailure(429)

    monkeypatch.setattr(client, '_dispatch', refuse)
    with pytest.raises(SettledHTTPError) as caught:
        complete(client)
    assert isinstance(caught.value, Exception) and caught.value.status == 429
    reservation = client.config.max_call_reservation_microusd
    assert caught.value.cost_microusd == reservation == client.ledger.spent_microusd
    restarted = DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')
    monkeypatch.setattr(restarted, '_dispatch', lambda request: pytest.fail('unexpected dispatch'))
    with pytest.raises(SettledHTTPError):
        complete(restarted)
    assert restarted.ledger.spent_microusd == reservation


def test_no_response_settles_at_full_reservation_and_replays(tmp_path, monkeypatch):
    from evolution.openrouter_engine import DispatchFailure

    monkeypatch.setenv('OPENROUTER_API_KEY', 'local-pilot-only')
    client = DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')
    def fail(request):
        raise DispatchFailure('OpenRouter request failed')

    monkeypatch.setattr(client, '_dispatch', fail)
    with pytest.raises(SettledUnknownDispatch) as caught:
        complete(client)
    reservation = client.config.max_call_reservation_microusd
    assert caught.value.cost_microusd == reservation == client.ledger.spent_microusd
    restarted = DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')
    monkeypatch.setattr(restarted, '_dispatch', lambda request: pytest.fail('unexpected dispatch'))
    with pytest.raises(SettledUnknownDispatch):
        complete(restarted)


def test_integrity_failures_after_a_response_still_stop(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'local-pilot-only')
    client = DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')
    def mismatch(request):
        raise ValueError('provider differs from pin')

    monkeypatch.setattr(client, '_dispatch', mismatch)
    with pytest.raises(UnresolvedDispatch):
        complete(client)


def test_crash_pending_reservation_is_reconciled_by_appending(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'local-pilot-only')
    client = DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')

    def lost(request):
        raise KeyboardInterrupt  # process dies mid-request: reservation stays pending

    monkeypatch.setattr(client, '_dispatch', lost)
    with pytest.raises(KeyboardInterrupt):
        complete(client)
    before = (tmp_path / 'journal.jsonl').read_text()
    with pytest.raises(UnresolvedDispatch):
        DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')
    record = reconcile_unknown(tmp_path / 'journal.jsonl', config(), 'host reboot')
    assert record['cost_microusd'] == config().max_call_reservation_microusd
    assert (tmp_path / 'journal.jsonl').read_text().startswith(before)
    restarted = DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')
    assert restarted.ledger.spent_microusd == record['cost_microusd']
    assert reconcile_unknown(tmp_path / 'journal.jsonl', config(), 'again') is None


def test_launch_version_journals_continue_with_their_recorded_identity(tmp_path, monkeypatch):
    import hashlib

    from experiments.gp_tuning_20260902.recovery import canonical, digest
    from experiments.matched_transport_20260909 import client as module

    monkeypatch.setenv('OPENROUTER_API_KEY', 'local-pilot-only')
    predecessor = module.COMPATIBLE_TRANSPORT_SHA256[0]
    recorded = digest({'config': dataclasses.asdict(config()),
                       'transport_source_sha256': predecessor})
    body = {'seq': 0, 'previous_sha256': '0' * 64,
            'payload': {'type': 'configuration', 'config_sha256': recorded,
                        'schema': 'evolved-durable-openrouter-v1'}}
    (tmp_path / 'journal.jsonl').write_text(canonical({**body, 'sha256': digest(body)}) + '\n')
    client = DurableOpenRouterClient(config(), journal=tmp_path / 'journal.jsonl')
    assert client.config_sha == recorded
    assert client.engine_descriptor['transport_source_sha256'] == predecessor
    assert client.running_transport_sha == hashlib.sha256(
        Path(module.__file__).read_bytes()).hexdigest()
    monkeypatch.setattr(client, '_dispatch', lambda request: response(client))
    assert complete(client).content
