"""Local HTTPS + actual CLI: paired barrier and durable response/campaign-ledger replay."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import ssl
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from evolution.openrouter_engine import load_openrouter_engine_config
from experiments.gp_tuning_20260902.recovery import read_sealed
from experiments.gp_tuning_20260902.run_recovered import write_sealed
from experiments.matched_control_20260909.real_pilot import INITIAL, ROOT, SEED_ARCHIVE


@pytest.mark.parametrize('failure_mode', ['saved-response', 'lost-response'])
def test_cli_replays_or_stops_without_duplicate_dispatch(tmp_path, failure_mode):
    key, cert = tmp_path / 'key.pem', tmp_path / 'cert.pem'
    tls = tmp_path / 'tls.cnf'
    tls.write_text('[req]\ndistinguished_name=dn\nx509_extensions=ext\nprompt=no\n'
                   '[dn]\nCN=127.0.0.1\n[ext]\nsubjectAltName=IP:127.0.0.1\n'
                   'basicConstraints=critical,CA:TRUE\n')
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', str(key), '-out', str(cert), '-days', '1', '-config', str(tls)],
                   check=True, capture_output=True, timeout=30)
    requests = []
    response_mode = {'drop': False}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path == '/api/v1/chat/completions'
            requests.append(request)
            if response_mode['drop']:
                self.close_connection = True
                return  # Simulate acceptance followed by a lost response, without a retry.
            prompt = request['messages'][0]['content']
            parent = prompt.split('PARENT:\n', 1)[1].split('# EVOLVE-BLOCK-END', 1)[0]
            parent += '# EVOLVE-BLOCK-END\n'
            old = re.search(r'^def .+:$', parent, re.MULTILINE).group()
            new = f"# local fake proposal {request['seed']}\n{old}"
            if 'PROPOSAL MODE: DIFF' in prompt:
                content = f'<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE'
            else:
                content = '```python\n' + parent.replace(old, new, 1) + '```'
            raw = {'model': request['model'],
                   'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
                   'usage': {'prompt_tokens': 10, 'completion_tokens': 20}}
            body = json.dumps(raw).encode()
            self.send_response(200)
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
    config = dataclasses.replace(
        load_openrouter_engine_config(ROOT / 'evolution/openrouter_engine_glm53.json'),
        endpoint=f'https://127.0.0.1:{server.server_port}/api/v1', max_attempts=1,
        request_timeout_s=10., reasoning_effort=None,
    )
    engine = tmp_path / 'engine.json'
    engine.write_text(json.dumps(dataclasses.asdict(config)))
    plan = json.loads((ROOT / 'evolution/campaigns.json').read_text())
    plan.update(candidate_budget=100, shared_numeric_probability=0., proposals_per_call=1)
    plan['campaigns'][0].update(root_seed=9113512321037399041, evaluation_replicates=1)
    plan['evaluator_work'].update(max_microwork=20_000_000,
                                   checkpoints_microwork=[0, 20_000_000])
    campaign_config = tmp_path / 'campaign.json'
    campaign_config.write_text(json.dumps(plan))
    pair = tmp_path / 'pair.json'
    journal = tmp_path / 'proposal-journal.jsonl'
    env = {**os.environ, 'OPENROUTER_API_KEY': 'local-pilot-only', 'SSL_CERT_FILE': str(cert),
           'NO_PROXY': '127.0.0.1', 'no_proxy': '127.0.0.1'}
    env.pop('MATCHED_CLI_PILOT_CRASH', None)

    def run(arm, expected, *, crash=False):
        command = [sys.executable, '-m', 'experiments.matched_cli_pilot_20260909.worker',
                   '--pair-receipt', str(pair), '--seed-archive', str(SEED_ARCHIVE),
                   '--initial', str(INITIAL), '--campaign_id', 'c00', '--arm', arm,
                   '--archive_dir', str(tmp_path / arm), '--campaign_config', str(campaign_config),
                   '--task_config',
                   str(ROOT / 'evolution/task/config_dev_continuum_tight_calibrated.json'),
                   '--development']
        if arm == 'llm':
            command += ['--proposal_engine', 'openrouter',
                        '--openrouter_engine_config', str(engine),
                        '--proposal-journal', str(journal)]
        child_env = {**env, 'MATCHED_CLI_PILOT_CRASH': 'after-durable-response'} if crash else env
        result = subprocess.run(command, cwd=ROOT, env=child_env, capture_output=True,
                                text=True, timeout=60)
        assert result.returncode == expected, result.stdout + result.stderr
        return result

    def events(arm):
        path = tmp_path / arm / 'events.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()]

    try:
        for arm in ('gp', 'llm'):
            run(arm, 10)
            assert len([e for e in events(arm) if e['type'] == 'candidate_completed']) == 12
        assert requests == []
        certification = subprocess.run([
            sys.executable, '-m', 'experiments.matched_control_20260909.run_campaign',
            '--pair-receipt', str(pair), '--certify-pair',
            str(tmp_path / 'gp'), str(tmp_path / 'llm'),
        ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        assert certification.returncode == 0, certification.stderr
        assert read_sealed(pair)['arms'][0]['records'] == read_sealed(pair)['arms'][1]['records']
        if failure_mode == 'lost-response':
            response_mode['drop'] = True
            failed = run('llm', 1)
            assert 'automatic retry refused' in failed.stderr
            assert len(requests) == 1
            response_mode['drop'] = False
            refused = run('llm', 1)
            assert 'Unresolved OpenRouter dispatch' in refused.stderr
            assert len(requests) == 1
            assert events('llm')[-1]['type'] == 'candidate_created'
            assert json.loads(journal.read_text().splitlines()[-1])['payload']['type'] == 'reserved'
            write_sealed(tmp_path / 'PILOT_RESULT.json', {
                'complete': True, 'mode': failure_mode, 'http_requests': 1,
                'initialization_model_calls': 0, 'restarted_without_duplicate_dispatch': True,
                'outcome': 'CLI stops and keeps unresolved reservation; no automatic recovery',
                'boundary': 'Controlled evaluator and local HTTPS, dummy key; no live model calls',
            })
            return
        run('llm', 23, crash=True)
        assert len(requests) == 1
        assert events('llm')[-1]['type'] == 'candidate_created'
        saved_request = requests[0]
        run('llm', 0)
        assert requests.count(saved_request) == 1
        completed = [e['payload'] for e in events('llm') if e['type'] == 'candidate_completed']
        assert completed[12]['proposal']['valid']
        journal_events = [json.loads(line)['payload'] for line in journal.read_text().splitlines()]
        settled = [e['completion']['cost_microusd'] for e in journal_events
                   if e['type'] == 'completed']
        costs = [a['metadata'].get('api_costs_microusd', 0)
                 for row in completed for a in row['proposal']['attempts']]
        assert sum(costs) == sum(settled) > 0
        assert all(a['metadata']['cost_basis'] == 'pinned-token-prices'
                   for row in completed for a in row['proposal']['attempts']
                   if a['metadata'].get('api_costs_microusd', 0) > 0)
        before = len(requests)
        archive_before = (tmp_path / 'llm/events.jsonl').read_bytes()
        run('llm', 0)
        assert len(requests) == before
        assert (tmp_path / 'llm/events.jsonl').read_bytes() == archive_before
        write_sealed(tmp_path / 'PILOT_RESULT.json', {
            'complete': True, 'mode': failure_mode, 'http_requests': len(requests),
            'initialization_model_calls': 0, 'restarted_without_duplicate_dispatch': True,
            'campaign_attempt_cost_microusd': sum(costs),
            'transport_cost_microusd': sum(settled), 'completed_restart_archive_unchanged': True,
            'boundary': 'Controlled evaluator and local HTTPS, dummy key; no live model calls',
        })
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
