"""Actual campaign CLI with controlled evaluator and explicit crash injection for a local pilot."""
from __future__ import annotations

import hashlib
import os
import runpy

import evolution.campaign as campaign
from evolution.evaluation import PolicyEvaluation, policy_identity_for_source
from evolution.islands import ArchiveAssessment, DecisionCost
from evolution.ladder import LadderResult
from experiments.matched_transport_20260909.client import DurableOpenRouterClient
from sim.energy import CandidateFitness


def evaluate(source, task, ladder, proposal_seq):
    del task, ladder
    identity = policy_identity_for_source(source)
    fitness = CandidateFitness(
        feasible=True, infeasibility_vector=(0., 0., 0., 0.),
        summed_primitive_shortfall=0., normalized_energy=1. + proposal_seq / 100.,
        proposal_seq=proposal_seq, n_instances=1, n_bundles=3,
    )
    fields = {'policy_name': identity['name'], 'policy_version': identity['version'],
              'policy_identity_sha256': identity['identity_sha256'],
              'policy_artifact_sha256': identity['artifact_sha256']}
    return LadderResult(
        promoted_to_l2=True, evaluation=PolicyEvaluation(fitness=fitness, instances=(), **fields),
        trace=({'rung': 'L2', 'decision': 'fitness-of-record',
                'source_sha256': hashlib.sha256(source.encode()).hexdigest()},),
        budget_registration_status='development-provisional',
        budget_registration_source='controlled CLI/transport pilot; no simulator claim', **fields,
    )


def assess(source, task, islands):
    del task, islands
    cost = DecisionCost(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                        probe_sha256='a' * 64, score_batch_fuel=100, score_batch_memory_bytes=64,
                        calibrated_latency_us=2., calibration_id='controlled-cli-pilot',
                        calibration_status='development-provisional')
    return ArchiveAssessment(loc=10, decision_cost=cost, loc_penalty=0.,
                             decision_latency_penalty=0., total_penalty=0.)


def main():
    # Only evaluation and assessment are controlled. Parsing, archive, pairing, proposal
    # generation/validation, HTTPS, journal, budgets and CLI resume use production paths.
    original_init = campaign.CampaignRunner.__init__

    def initialize(self, **kwargs):
        kwargs.update(evaluate_fn=evaluate, assess_fn=assess)
        return original_init(self, **kwargs)

    campaign.CampaignRunner.__init__ = initialize
    original = DurableOpenRouterClient.complete

    def completion(self, **kwargs):
        result = original(self, **kwargs)
        if os.environ.get('MATCHED_CLI_PILOT_CRASH') == 'after-durable-response':
            os._exit(23)
        return result

    DurableOpenRouterClient.complete = completion
    runpy.run_module('experiments.matched_control_20260909.run_campaign', run_name='__main__')


if __name__ == '__main__':
    main()
