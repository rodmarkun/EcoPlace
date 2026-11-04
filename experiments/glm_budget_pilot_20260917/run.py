"""Three-call generation-cost pilot; no simulation or scientific outcome selection."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
import urllib.request
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import placementscript as ps
from shinka.database import Program

from analysis.tracking import derive_seed
from evolution.llm_operator import OpenWeightLLMOperator
from evolution.openrouter_engine import load_openrouter_engine_config
from evolution.profiles import evolution_profile
from experiments.matched_transport_20260909.client import DurableOpenRouterClient

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = HERE / 'artifacts'
JOURNAL = ROOT / 'experiments/api_budget_20260917/journal.jsonl'
PIN = ROOT / 'prereg/CORRECTED_STUDIES_20260917_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def bound(prompt, config):
    # UTF-8 byte count upper-bounds byte-based text tokens; 1024 covers chat framing.
    n = len(prompt.encode()) + 1024
    cost = math.ceil((n * config.prompt_microusd_per_million
                     + config.max_output_tokens * config.completion_microusd_per_million) / 1e6)
    if cost > config.max_call_reservation_microusd:
        raise ValueError('prompt/output bound exceeds reserved cost')
    return cost


def check_receipt(raw, config):
    usage = raw['usage']
    a, b = usage['prompt_tokens'], usage['completion_tokens']
    if type(a) is not int or type(b) is not int or min(a, b) < 0:
        raise ValueError('invalid usage')
    cost = math.ceil((a * config.prompt_microusd_per_million
                     + b * config.completion_microusd_per_million) / 1e6)
    billed = Decimal(str(usage['cost']))
    if not billed.is_finite() or billed < 0:
        raise ValueError('invalid provider cost')
    billed_micro = int((billed * 1000000).to_integral_value(rounding=ROUND_CEILING))
    if billed_micro > cost or cost > config.max_call_reservation_microusd:
        raise ValueError('provider cost outside conservative accounting bound')
    if raw.get('provider', '').lower() != 'z.ai':
        raise ValueError('provider differs from pin')
    return cost


class ReceiptClient(DurableOpenRouterClient):
    def _dispatch(self, request):
        raw = super()._dispatch(request)
        # Preserve raw provider usage/cost before stricter parser checks.
        save(self.receipt_path, {'response': raw, 'elapsed_seconds': time.monotonic()-self.started})
        check_receipt(raw, self.config)
        return raw

    def complete(self, *, model, prompt, generation, seed):
        bound(prompt, self.config)
        return super().complete(model=model, prompt=prompt, generation=generation, seed=seed)


def prepare():
    OUT.mkdir()
    config = json.loads((PIN / 'engine.json').read_text())
    config['max_total_microusd'] = 60000000
    save(OUT / 'engine.json', config)
    cfg = load_openrouter_engine_config(OUT / 'engine.json')
    a = json.loads((PIN / 'initialization/A00.json').read_text())
    # Read initial sources only, never final scores. Stored archives use sealed envelopes.
    entries = a.get('payload', a)['entries']
    parent = Program(id='cost-pilot-parent', code=entries[0]['source'], timestamp=0)
    inspiration = Program(id='cost-pilot-inspiration', code=entries[1]['source'], timestamp=0)
    requests = []
    occupied = {r['seed'] for r in json.loads((PIN / 'SEEDS.json').read_text())['enumerated_seed_records']}
    for i, mode in enumerate(('diff', 'rewrite', 'crossover')):
        seed = derive_seed(20260917, 'glm-budget-generation-pilot', str(i))
        assert seed not in occupied
        occupied.add(seed)
        prompt = OpenWeightLLMOperator._prompt(parent, (inspiration,), mode,
                    profile=evolution_profile('deep-search-feedback'), proposals_per_call=1)
        requests.append({'mode':mode, 'seed':seed, 'prompt':prompt, 'parent':parent.code,
                         'reservation_bound_microusd':bound(prompt, cfg)})
    save(OUT / 'requests.json', requests)
    save(OUT / 'PLAN.json', {
        'purpose':'initial-context generation cost pilot; not full adaptive campaign or G0 acceptance',
        'model':'z-ai/glm-5.3', 'max_calls':3, 'pilot_reserved_ceiling_usd':0.6,
        'project_total_ceiling_usd':60, 'automatic_retries':False,
        'limitations':'Three modes on one initial parent; no long feedback, repairs, B context or simulator timings. No inference of policy quality or full-run affordability.',
        'sources':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
          ROOT/'evolution/openrouter_engine.py', ROOT/'evolution/llm_operator.py',
          ROOT/'experiments/matched_transport_20260909/client.py',
          ROOT/'docs/API_BUDGET_20260917.json', PIN/'initialization/A00.json', PIN/'SEEDS.json']},
        'inputs':{p.name:sha(p) for p in (OUT/'engine.json',OUT/'requests.json')},
    })
    print('Prepared exactly three calls, maximum $0.60 reserved; aggregate cap $60.', flush=True)


def run():
    plan = json.loads((OUT/'PLAN.json').read_text())
    for name, h in plan['sources'].items():
        assert sha(ROOT/name) == h, name
    for name, h in plan['inputs'].items():
        assert sha(OUT/name) == h, name
    cfg = load_openrouter_engine_config(OUT/'engine.json')
    # Unauthenticated, unbilled model preflight before constructing any paid request.
    with urllib.request.urlopen('https://openrouter.ai/api/v1/models/z-ai/glm-5.3/endpoints', timeout=30) as r:
        endpoints = json.load(r)
    eligible = [e for e in endpoints['data']['endpoints']
                if e['provider_name'].lower()=='z.ai'
                and Decimal(e['pricing']['prompt']) <= Decimal('0.0000014')
                and Decimal(e['pricing']['completion']) <= Decimal('0.0000044')]
    if not eligible:
        raise ValueError('pinned provider/prices unavailable')
    if not (OUT/'endpoints.json').exists():
        save(OUT/'endpoints.json', endpoints)
    client = ReceiptClient(cfg, journal=JOURNAL)
    requests = json.loads((OUT/'requests.json').read_text())
    for i, req in enumerate(requests):
        result = OUT/f'result-{i}.json'
        client.receipt_path = OUT/f'provider-{i}.json'
        client.started = time.monotonic()
        # Completed journal entries replay free; uncertain dispatches block even on restart.
        c = client.complete(model=cfg.model_id, generation=cfg.generation,
                            prompt=req['prompt'], seed=req['seed'])
        if not result.exists():
            valid, error = False, None
            try:
                source = OpenWeightLLMOperator._apply_completion(req['parent'], c.content, req['mode'])
                ps.compile(source)
                valid = True
                (OUT/f'policy-{i}.ps').write_text(source)
            except Exception as exc:
                error = str(exc)[:1000]
            save(result, {'mode':req['mode'], 'completion':dataclasses.asdict(c),
                 'compiled':valid, 'validation_error':error})
        print(json.dumps({'call':i+1, 'mode':req['mode'], 'cost_usd':c.cost_microusd/1e6,
                          'aggregate_accounted_usd':client.ledger.spent_microusd/1e6}), flush=True)
    print('Pilot complete. No scientific campaigns launched.', flush=True)


if __name__ == '__main__':
    import sys
    {'prepare':prepare, 'run':run}[sys.argv[1]]()
