"""Open-weight Qwen proposal operator over Shinka's common seeded contract."""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import placementscript as ps
from shinka.core import ProposalOffspring
from shinka.database import Program
from shinka.edit import apply_full_patch

from .model import ApiCostLedger, ModelTrack
from .model import host_handshake_sha256 as _host_handshake_sha256
from .operators import apply_scoped_diff, validate_scoped_edit
from .profiles import MATCHED_PROFILE, EvolutionProfile, evolution_profile


@dataclass(frozen=True)
class LLMCompletion:
    content: str
    cost_microusd: int | None
    prompt_tokens: int
    completion_tokens: int
    raw_response: dict[str, Any]


class CompletionClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        prompt: str,
        generation: dict[str, Any],
        seed: int,
    ) -> LLMCompletion: ...


class VLLMCompletionClient:
    """Minimal OpenAI-compatible client for the isolated pinned proposal host."""

    def __init__(
        self,
        endpoint: str,
        *,
        max_request_cost_microusd: int,
        revision: str,
        weight_set_sha256: str,
        timeout_s: float = 300.0,
    ) -> None:
        self.base_endpoint = endpoint.rstrip("/")
        parsed = urlsplit(self.base_endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("completion endpoint must be a pathless HTTP(S) origin")
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        port = "" if parsed.port is None else f":{parsed.port}"
        self.origin = f"{parsed.scheme}://{host}{port}"
        self.endpoint = self.base_endpoint + "/v1/chat/completions"
        self.max_request_cost_microusd = max_request_cost_microusd
        self.revision = revision
        self.weight_set_sha256 = weight_set_sha256
        self._bound_handshake_sha256: str | None = None
        self.timeout_s = timeout_s

    def _open(self, request: urllib.request.Request):
        response = urllib.request.urlopen(request, timeout=self.timeout_s)
        final = urlsplit(response.geturl())
        host = final.hostname
        if host is None:
            response.close()
            raise ValueError("proposal response has no final HTTP origin")
        if ":" in host:
            host = f"[{host}]"
        port = "" if final.port is None else f":{final.port}"
        final_origin = f"{final.scheme}://{host}{port}"
        if final_origin != self.origin:
            response.close()
            raise ValueError("proposal endpoint redirected to a different origin")
        return response

    def fetch_handshake(self) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_endpoint + "/evolved-continuum/handshake",
            method="GET",
        )
        with self._open(request) as response:
            raw = json.loads(response.read())
        if not isinstance(raw, dict):
            raise ValueError("proposal endpoint handshake must be a JSON object")
        return raw

    def bind_handshake(self, handshake: dict[str, Any]) -> str:
        digest = _host_handshake_sha256(handshake)
        self._bound_handshake_sha256 = digest
        return digest

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        generation: dict[str, Any],
        seed: int,
    ) -> LLMCompletion:
        if self._bound_handshake_sha256 is None:
            raise RuntimeError("completion client must bind a verified live handshake")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "seed": seed,
            "request_id": f"{model}:{seed}",
            "max_cost_microusd": self.max_request_cost_microusd,
            "expected_revision": self.revision,
            "expected_weight_set_sha256": self.weight_set_sha256,
            "expected_host_handshake_sha256": self._bound_handshake_sha256,
            **generation,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, sort_keys=True).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._open(request) as response:
            raw = json.loads(response.read())
        expected_identity = {
            "model": model,
            "revision": self.revision,
            "weight_set_sha256": self.weight_set_sha256,
            "host_handshake_sha256": self._bound_handshake_sha256,
        }
        if any(raw.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("proposal response model/weight/host identity differs from pin")
        usage = raw.get("usage", {})
        # Provider request ids and wall-clock fields are deliberately excluded from
        # the scientific archive. The normalized semantic response is complete for
        # replay and stable under an idempotent model+seed request.
        normalized_response = {
            "model": raw.get("model", model),
            "revision": raw["revision"],
            "weight_set_sha256": raw["weight_set_sha256"],
            "host_handshake_sha256": raw["host_handshake_sha256"],
            "choices": [
                {
                    "index": choice.get("index"),
                    "finish_reason": choice.get("finish_reason"),
                    "message": choice.get("message"),
                }
                for choice in raw.get("choices", [])
            ],
            "usage": usage,
            "cost_microusd": raw.get("cost_microusd"),
        }
        return LLMCompletion(
            content=raw["choices"][0]["message"]["content"],
            cost_microusd=raw.get("cost_microusd"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            raw_response=normalized_response,
        )


PROPOSAL_MODES = ("diff", "rewrite", "crossover")
_LINE_ERROR = re.compile(r"\(line (\d+)\)")
_MAX_REPAIR_EXCERPT_CHARS = 1200
_BATCH_BEGIN = re.compile(
    r"(?m)^<<<EVOLVED_CONTINUUM_PROPOSAL:(\d+):BEGIN>>>[ \t]*\r?$"
)
_BATCH_END = re.compile(
    r"(?m)^<<<EVOLVED_CONTINUUM_PROPOSAL:(\d+):END>>>[ \t]*\r?$"
)

_VALID_CONSTRUCT_EXEMPLARS = """VALID PLACEMENTSCRIPT EXEMPLARS:
```python
out: list[Move] = []
for pod in state.pods:
    if len(out) < budget and not pod.stateful:
        out.append(move_pod(pod.id, target))
return out
```
```python
cpu_cap = float(max(node.cpu_capacity, 1))
cpu_after = float(node.cpu_allocated + pod.cpu_request) / cpu_cap
if cpu_after > 1.0:
    return -1.0
return 1.0 - cpu_after
```"""

# The load-time structural limits (spec §8.1). A live K=3 probe against the tightened
# seed returned 3/3 correctly delimited slots and **0/3 loadable policies**: every one
# failed on "block nesting too deep (max 8 levels)". The prompt listed the language
# subset but never these bounds, so the model had no way to know it was writing programs
# the loader would refuse -- a large, entirely avoidable share of the measured 67%
# parsed-to-loadable funnel loss.
_STRUCTURAL_LIMITS = (
    "Hard load-time limits, enforced before the policy runs: block nesting depth at most "
    "8 (every if/elif/else/for body counts one level, and the function body is level 1, "
    "so flatten with combined `and` conditions and early returns instead of stacking "
    "guards); expression nesting depth at most 32; at most 64 functions; static call "
    "chains at most 16 deep. Every expression statement must be an assignment, augmented "
    "assignment, or `.append(...)`; bare expressions are rejected. Call `float(...)` only "
    "on integer values, never on values that are already floats."
)

_COMMON_REJECTIONS = """COMMON REJECTIONS — copy the ACCEPTED shape, never REJECTED:
1 NUMERIC TYPES
REJECTED: `x = node.power_idle_w + 1`; `q = 3 / 2`;
`f = float(node.power_idle_w)`; `x: int = 1.0`; `max(node.cpu_capacity, 1.0)`
ACCEPTED: `x = node.power_idle_w + 1.0`; `q = 3 // 2`;
`f = node.power_idle_w`; `x: float = 1.0`; `max(node.cpu_capacity, 1)`
Ints and floats never mix. `/` requires two floats; `//` handles ints. `float(i)` converts
an int only. An annotation must match its value. `min`/`max` take exactly two values of
one numeric type.
2 STATEMENTS
REJECTED: `move_pod(pod.id, target)`
ACCEPTED: `move = move_pod(pod.id, target)` or `out.append(move_pod(pod.id, target))`
No other bare expression statement is legal.
3 NESTING
REJECTED (too deep):
`for pod in state.pods:\n    if movable:\n        if fits:\n            if useful:\n`
`                out.append(move_pod(pod.id, target))`
ACCEPTED (flat):
`for pod in state.pods:\n    eligible = movable and fits and useful\n    if eligible:\n`
`        out.append(move_pod(pod.id, target))`
Combine/invert guards or return early from a helper. `continue` is not a PlacementScript statement.
4 STATEMENT METHODS
REJECTED: `node.cpu_allocated(pod.cpu_request)`
ACCEPTED: `cpu_after = node.cpu_allocated + pod.cpu_request`
The only method allowed as a statement is list `.append(...)`; record fields are read, not called.
5 COMPLETE POLICY/PATCH
REJECTED: prose, an unfenced rewrite, or an incomplete patch fragment.
ACCEPTED rewrite: one complete policy inside ` ```python ... ``` `.
ACCEPTED diff: complete `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` blocks with
exact parent text.
6 BATCH DELIMITERS
REJECTED: a missing, duplicate, unmatched, nested, or code-body delimiter.
ACCEPTED: exactly one numbered BEGIN/END pair around each requested body, in order."""

ABLATION_PROMPT_HINT = (
    "SEMANTIC ABLATION PROBE: Deliberately try one reductive probe: delete a scoring "
    "term, use a ratio's raw numerator, or replace a capacity with a constant. This is "
    "legitimate exploration, but every PlacementScript constraint still applies."
)


# What the candidate is actually being scored on. Until 2026-08-25 the prompt said only
# "improve the parent as a cloud-edge placement scorer and bounded move proposer" and
# never named the objective, the constraints, the direction of the score, or the fields
# available on the observation records. A proposal operator that is not told the task is
# doing syntax-directed search with extra steps, which is precisely the comparison the
# GP arm already represents.
#
# This is a description of the *task family* (plan §5 and the §4 observation schema), not
# coaching about any scenario: it states the objective and the API and leaves every
# trade-off for the model to find. It reaches the LLM arm only -- the deterministic
# operators have no prompt -- and is therefore part of what "the LLM proposal operator"
# means in the LLM-vs-GP contrast, and is recorded as such.
_TASK_BRIEF = (
    "TASK. `score(pod, node, ctx) -> float` ranks the nodes that native Filter plugins "
    "have already found feasible for one pod; the scheduler binds the pod to the "
    "**highest**-scoring node, so a cost must be returned negated. `propose_moves(state, "
    "graph, budget) -> list[Move]` is consumed by a fixed rebalancing controller that "
    "executes at most `budget` moves per window.\n"
    "OBJECTIVE. Minimize total modeled energy over a fixed horizon under fixed offered "
    "load, subject to: per-application SLO satisfaction at or above target measured over "
    "*offered* requests (so drops, timeouts and requests still in flight at the horizon "
    "all count against it), goodput at or above target, drop and timeout rate caps, and a "
    "cumulative disruption budget over migration count, bytes moved and unavailability "
    "seconds. A candidate that violates any constraint ranks below every feasible one, so "
    "energy is worth nothing if the constraints are missed.\n"
    "OBSERVATIONS. `ctx.nodes[i]` carries `tier`, `cpu_capacity`, `cpu_allocated`, "
    "`mem_capacity`, `mem_allocated`, `power_idle_w`, `power_max_w`, `suspended`, "
    "`suspendable` and `latency_to_ms` (a per-node vector of link latencies). `ctx.pods[i]` "
    "carries `service`, `app`, `cpu_request`, `mem_request`, `stateful`, `has_pdb`, "
    "`image_mb` and `current_node` (-1 when unplaced). `ctx.graph` holds the pod-level "
    "call graph as parallel `src`/`dst`/`rate`/`payload_kb` lists over `ctx.pods` indices, "
    "including the edges the pod being scored will carry once it is serving."
)


@dataclass(frozen=True)
class ValidationRepair:
    error: str
    offending_excerpt: str

    def __post_init__(self) -> None:
        if not self.error or len(self.error) > 2000:
            raise ValueError("repair error must be non-empty and bounded")
        if not self.offending_excerpt or len(self.offending_excerpt) > (
            _MAX_REPAIR_EXCERPT_CHARS
        ):
            raise ValueError("repair excerpt must be non-empty and bounded")


@dataclass(frozen=True)
class ParsedBatchProposal:
    """One requested batch slot; malformed slots do not poison their siblings."""

    batch_index: int
    content: str | None
    error: str | None

    @property
    def valid(self) -> bool:
        return self.content is not None and self.error is None


def parse_batched_proposals(
    response: str,
    proposals_per_call: int,
) -> tuple[ParsedBatchProposal, ...]:
    """Parse exactly the requested numbered slots, failing each slot independently."""
    if (
        not isinstance(proposals_per_call, int)
        or isinstance(proposals_per_call, bool)
        or proposals_per_call < 2
    ):
        raise ValueError("batched proposal parsing requires proposals_per_call >= 2")
    begins: dict[int, list[re.Match[str]]] = {}
    ends: dict[int, list[re.Match[str]]] = {}
    for match in _BATCH_BEGIN.finditer(response):
        begins.setdefault(int(match.group(1)), []).append(match)
    for match in _BATCH_END.finditer(response):
        ends.setdefault(int(match.group(1)), []).append(match)
    all_markers = sorted(
        [*_BATCH_BEGIN.finditer(response), *_BATCH_END.finditer(response)],
        key=lambda match: match.start(),
    )
    parsed: list[ParsedBatchProposal] = []
    seen_content: dict[str, int] = {}
    for batch_index in range(1, proposals_per_call + 1):
        slot_begins = begins.get(batch_index, [])
        slot_ends = ends.get(batch_index, [])
        if not slot_begins and not slot_ends:
            parsed.append(
                ParsedBatchProposal(
                    batch_index,
                    None,
                    "model returned fewer than the requested number of delimited proposals",
                )
            )
            continue
        if len(slot_begins) != 1 or len(slot_ends) != 1:
            parsed.append(
                ParsedBatchProposal(
                    batch_index,
                    None,
                    "proposal slot has duplicate or unmatched delimiters",
                )
            )
            continue
        begin = slot_begins[0]
        end = slot_ends[0]
        if begin.end() >= end.start():
            parsed.append(
                ParsedBatchProposal(
                    batch_index,
                    None,
                    "proposal slot delimiters are reversed or empty",
                )
            )
            continue
        nested = [
            marker
            for marker in all_markers
            if begin.end() <= marker.start() < end.start()
        ]
        if nested:
            parsed.append(
                ParsedBatchProposal(
                    batch_index,
                    None,
                    "proposal slot contains a nested batch delimiter",
                )
            )
            continue
        content = response[begin.end() : end.start()].strip()
        if not content:
            parsed.append(
                ParsedBatchProposal(batch_index, None, "proposal slot body is empty")
            )
            continue
        content_sha256 = hashlib.sha256(content.encode()).hexdigest()
        duplicate_of = seen_content.get(content_sha256)
        if duplicate_of is not None:
            parsed.append(
                ParsedBatchProposal(
                    batch_index,
                    None,
                    f"proposal duplicates batch slot {duplicate_of}",
                )
            )
            continue
        seen_content[content_sha256] = batch_index
        parsed.append(ParsedBatchProposal(batch_index, content, None))
    return tuple(parsed)


def _validation_excerpt(source: str | None, error: str, fallback: str) -> str:
    """Return deterministic local lines around a parser/type error."""
    if source is not None:
        match = _LINE_ERROR.search(error)
        if match is not None:
            lines = source.splitlines()
            line = int(match.group(1))
            start = max(0, line - 3)
            end = min(len(lines), line + 2)
            excerpt = "\n".join(
                f"{index + 1}: {lines[index]}" for index in range(start, end)
            )
            if excerpt:
                return excerpt[:_MAX_REPAIR_EXCERPT_CHARS]
    excerpt = fallback.strip()
    if not excerpt:
        excerpt = "<completion produced no inspectable source>"
    return excerpt[:_MAX_REPAIR_EXCERPT_CHARS]


class OpenWeightLLMOperator:
    name = "open_weight_qwen"

    def __init__(
        self,
        track: ModelTrack,
        client: CompletionClient,
        cost_ledger: ApiCostLedger,
        *,
        proposal_modes: tuple[str, ...] = PROPOSAL_MODES,
        profile: EvolutionProfile | None = None,
        ablation_prompt_hint_probability: float = 0.0,
        proposals_per_call: int = 1,
    ) -> None:
        if not proposal_modes or any(mode not in PROPOSAL_MODES for mode in proposal_modes):
            raise ValueError("LLM proposal modes must be a non-empty registered subset")
        if len(proposal_modes) != len(set(proposal_modes)):
            raise ValueError("LLM proposal modes must not contain duplicates")
        if (
            not isinstance(ablation_prompt_hint_probability, (int, float))
            or isinstance(ablation_prompt_hint_probability, bool)
            or not 0.0 <= ablation_prompt_hint_probability <= 1.0
        ):
            raise ValueError("ablation prompt hint probability must be in [0, 1]")
        if (
            not isinstance(proposals_per_call, int)
            or isinstance(proposals_per_call, bool)
            or proposals_per_call < 1
        ):
            raise ValueError("proposals_per_call must be a positive integer")
        self.track = track
        self.client = client
        self.cost_ledger = cost_ledger
        self.proposal_modes = proposal_modes
        self.profile = profile or evolution_profile(MATCHED_PROFILE)
        self.ablation_prompt_hint_probability = float(
            ablation_prompt_hint_probability
        )
        self.proposals_per_call = proposals_per_call
        self.name = getattr(client, "operator_name", self.name)
        self.engine_descriptor = getattr(client, "engine_descriptor", None)
        self.usage_ledger = getattr(client, "usage_ledger", None)

    def _is_subscription_engine(self) -> bool:
        return (self.engine_descriptor is not None
                and self.engine_descriptor.get("engine") == "codex-cli-subscription")

    def _completion_cost_basis(self, completion: LLMCompletion) -> str:
        if self._is_subscription_engine():
            return "subscription-no-marginal-call-charge"
        if completion.cost_microusd is None:
            return "conservative-reservation"
        return getattr(self.client, "cost_basis", "host-reported")

    @staticmethod
    def _prompt(
        parent: Program,
        inspirations: tuple[Program, ...],
        mode: str,
        *,
        profile: EvolutionProfile | None = None,
        repair: ValidationRepair | None = None,
        ablation_probe: bool = False,
        proposals_per_call: int = 1,
    ) -> str:
        selected_profile = profile or evolution_profile(MATCHED_PROFILE)
        inspiration_text = "\n\n".join(
            f"INSPIRATION {index + 1}:\n{program.code}"
            for index, program in enumerate(inspirations)
        )
        feedback = str(parent.text_feedback).strip()
        feedback_text = f"\n\nSIMULATOR FEEDBACK:\n{feedback}" if feedback else ""
        if proposals_per_call == 1:
            format_instruction = {
                "diff": (
                    "Return only Shinka SEARCH/REPLACE blocks (<<<<<<< SEARCH, =======, "
                    ">>>>>>> REPLACE) whose SEARCH text occurs inside the EVOLVE-BLOCK."
                ),
                "rewrite": "Return only one complete policy inside a ```python code fence.",
                "crossover": (
                    "Combine useful parent and inspiration structure; return only one complete "
                    "policy inside a ```python code fence."
                ),
            }[mode]
            batch_contract = ""
        else:
            item_format = {
                "diff": (
                    "Shinka SEARCH/REPLACE blocks (<<<<<<< SEARCH, =======, >>>>>>> REPLACE) "
                    "whose SEARCH text occurs inside the EVOLVE-BLOCK"
                ),
                "rewrite": "one complete policy inside a ```python code fence",
                "crossover": (
                    "one complete policy inside a ```python code fence, combining useful "
                    "parent and inspiration structure"
                ),
            }[mode]
            format_instruction = (
                f"Return exactly {proposals_per_call} proposals; each delimited body must contain "
                f"only {item_format}."
            )
            batch_contract = (
                "\nBATCH DELIMITER CONTRACT. Emit exactly the requested number of blocks, "
                "numbered consecutively from 1, and no text outside them. For a 2-slot call "
                "the exact line-anchored format below instantiates "
                "<<<EVOLVED_CONTINUUM_PROPOSAL:N:BEGIN>>> and its matching END:\n"
                "<<<EVOLVED_CONTINUUM_PROPOSAL:1:BEGIN>>>\n<proposal 1 body>\n"
                "<<<EVOLVED_CONTINUUM_PROPOSAL:1:END>>>\n"
                "<<<EVOLVED_CONTINUUM_PROPOSAL:2:BEGIN>>>\n<proposal 2 body>\n"
                "<<<EVOLVED_CONTINUUM_PROPOSAL:2:END>>>\n"
                "Each BEGIN and matching END appears exactly once. Delimiter text must never "
                "appear inside a code body. "
                "Every block must pursue a genuinely different structural idea—not a literal "
                "tweak, parameter variation, or cosmetic rewrite of another block. Diversity "
                "between proposals is the purpose of this batch.\n"
            )
        structural_instruction = (
            " Prefer a substantial structural rewrite that changes policy behavior; "
            "do not spend this proposal on a one-literal or one-line micro-edit."
            if selected_profile.structural_rewrite_bias_enabled
            else ""
        )
        exemplars = (
            f"\n\n{_VALID_CONSTRUCT_EXEMPLARS}"
            if selected_profile.prompt_exemplars_enabled
            else ""
        )
        repair_text = ""
        if repair is not None:
            repair_text = (
                "\n\nREPAIR THE SAME PROPOSAL REQUEST. The previous response failed local "
                "PlacementScript validation. Keep the same proposal mode and repair only "
                "what is needed while still returning the requested output format.\n"
                f"VALIDATION ERROR (verbatim): {repair.error}\n"
                "OFFENDING EXCERPT:\n"
                f"```text\n{repair.offending_excerpt}\n```"
            )
        ablation_text = f"\n\n{ABLATION_PROMPT_HINT}" if ablation_probe else ""
        return (
            f"PROPOSAL MODE: {mode.upper()}\n{format_instruction}{batch_contract}\n"
            f"{_TASK_BRIEF}\n"
            "Improve the parent as a cloud-edge placement scorer and bounded move proposer."
            f"{structural_instruction} "
            "Preserve every byte outside EVOLVE-BLOCK, including POLICY_NAME and "
            "POLICY_VERSION. The output language is PlacementScript, not Python: use only "
            "typed functions, local assignments, if/elif/else, bounded for/range loops, "
            "return/pass, homogeneous list displays/indexing, and list.append statements. "
            "Dictionaries, sets, tuples, None, while, break/continue, lambda, comprehensions, "
            "sorting, slicing, keyword arguments, imports, I/O, reflection, recursion, and "
            "undefined helpers are forbidden. Empty lists need a list[T] annotation; int and "
            "float never mix implicitly; min/max take exactly two same-typed numeric values. "
            f"{_STRUCTURAL_LIMITS}\n\n{_COMMON_REJECTIONS}\n\n"
            "Use only record fields and builtins already demonstrated by the parent unless "
            f"their PlacementScript types are certain.{ablation_text}\n\n"
            f"PARENT:\n{parent.code}{feedback_text}\n\n"
            f"{inspiration_text}{exemplars}{repair_text}"
        ).rstrip()

    def _select_mode(
        self,
        available_modes: tuple[str, ...],
        rng: random.Random,
    ) -> str:
        weights = self.profile.proposal_mode_weights
        if not self.profile.structural_rewrite_bias_enabled or weights is None:
            return rng.choice(available_modes)
        registered = dict(zip(PROPOSAL_MODES, weights, strict=True))
        return rng.choices(
            available_modes,
            weights=[registered[mode] for mode in available_modes],
            k=1,
        )[0]

    @staticmethod
    def _apply_completion(parent: str, content: str, mode: str) -> str:
        if mode == "diff":
            return apply_scoped_diff(parent, content)
        else:
            result = apply_full_patch(
                patch_str=content,
                original_str=parent,
                language="python",
                verbose=False,
            )
        source, applied, _output, error, _patch_text, _patch_path = result
        if error is not None or applied < 1:
            raise ValueError(error or f"{mode} completion changed no EVOLVE-BLOCK content")
        if source == parent:
            raise ValueError(f"{mode} completion produced an unchanged policy")
        validate_scoped_edit(parent, source)
        return source

    @staticmethod
    def _attach_batch_replay(offspring: list[ProposalOffspring]) -> None:
        """Store later initial outcomes once on the call owner for crash-safe replay."""
        if not offspring:
            return
        offspring[0].metadata["batch_replay"] = [
            {
                "source": item.source,
                "error": item.error,
                "metadata": item.metadata,
            }
            for item in offspring[1:]
        ]

    def propose_batch(
        self,
        parent: Program,
        inspirations: tuple[Program, ...],
        rng: random.Random,
        *,
        proposals_per_call: int,
        shared_call_id: str,
        candidate_attempt_seeds: tuple[int, ...],
    ) -> tuple[ProposalOffspring, ...]:
        """Dispatch one completion and validate every delimited proposal independently."""
        if not isinstance(rng, random.Random):
            raise TypeError("LLM proposer requires the runner-owned random.Random")
        if (
            not isinstance(proposals_per_call, int)
            or isinstance(proposals_per_call, bool)
            or not 2 <= proposals_per_call <= self.proposals_per_call
        ):
            raise ValueError("batch size must be between 2 and configured proposals_per_call")
        if (
            len(candidate_attempt_seeds) != proposals_per_call
            or len(set(candidate_attempt_seeds)) != proposals_per_call
            or any(
                not isinstance(seed, int)
                or isinstance(seed, bool)
                or not 0 <= seed < 2**63
                for seed in candidate_attempt_seeds
            )
        ):
            raise ValueError("batch candidates require distinct explicit attempt seeds")
        if not isinstance(shared_call_id, str) or not shared_call_id:
            raise ValueError("batched proposals require a non-empty shared call id")
        available_modes = tuple(
            mode for mode in self.proposal_modes if mode != "crossover" or inspirations
        )
        if not available_modes:
            return tuple(
                ProposalOffspring(
                    source=None,
                    error="no registered LLM proposal mode is available for this parent",
                    metadata={
                        "operator": self.name,
                        "api_costs_microusd": 0,
                        "shared_call_id": shared_call_id,
                        "batch_index": index,
                        "proposals_per_call": proposals_per_call,
                        "candidate_attempt_seed": seed,
                    },
                )
                for index, seed in enumerate(candidate_attempt_seeds, start=1)
            )
        selected_mode = self._select_mode(available_modes, rng)
        hint_metadata: dict[str, Any] = {}
        ablation_probe = False
        if self.ablation_prompt_hint_probability > 0.0:
            hint_seed = rng.randrange(0, 2**63)
            hint_draw = random.Random(hint_seed).random()
            ablation_probe = hint_draw < self.ablation_prompt_hint_probability
            hint_metadata = {
                "ablation_prompt_hint_probability": (
                    self.ablation_prompt_hint_probability
                ),
                "ablation_prompt_hint_seed": hint_seed,
                "ablation_prompt_hint_draw": hint_draw,
                "ablation_prompt_hint_applied": ablation_probe,
            }
        prompt = self._prompt(
            parent,
            inspirations,
            selected_mode,
            profile=self.profile,
            ablation_probe=ablation_probe,
            proposals_per_call=proposals_per_call,
        )
        call_seed = rng.randrange(0, 2**63)
        reservation = self.track.max_call_reservation_microusd

        def referenced_metadata(
            index: int,
            seed: int,
            *,
            owner: bool,
        ) -> dict[str, Any]:
            metadata: dict[str, Any] = {
                "operator": self.name,
                "call_seed": call_seed,
                "model_id": self.track.model_id,
                "revision": self.track.revision,
                "proposal_mode": selected_mode,
                "weight_set_sha256": self.track.weight_set_sha256,
                "shared_call_id": shared_call_id,
                "shared_call_owner": owner,
                "shared_call_owner_batch_index": 1,
                "batch_index": index,
                "proposals_per_call": proposals_per_call,
                "candidate_attempt_seed": seed,
                **hint_metadata,
            }
            if self.engine_descriptor is not None:
                metadata["engine_descriptor"] = self.engine_descriptor
            return metadata

        try:
            self.cost_ledger.reserve(reservation)
        except ValueError as exc:
            failures = []
            for index, seed in enumerate(candidate_attempt_seeds, start=1):
                owner = index == 1
                metadata = referenced_metadata(index, seed, owner=owner)
                metadata.update(
                    {
                        "api_costs_microusd": 0,
                        "budget_exhausted": True,
                        "cost_ledger": self.cost_ledger.to_dict(),
                    }
                )
                if owner:
                    metadata["prompt"] = prompt
                failures.append(
                    ProposalOffspring(
                        source=None,
                        error=f"proposal cost budget exhausted: {exc}",
                        metadata=metadata,
                    )
                )
            self._attach_batch_replay(failures)
            return tuple(failures)

        completion: LLMCompletion | None = None
        try:
            completion = self.client.complete(
                model=self.track.model_id,
                prompt=prompt,
                generation=self.track.generation,
                seed=call_seed,
            )
            if completion.cost_microusd is not None and completion.cost_microusd > reservation:
                raise ValueError("proposal host reported cost above its hard reservation")
            charged = reservation if completion.cost_microusd is None else completion.cost_microusd
            self.cost_ledger.settle(reservation, charged)
        except Exception as exc:
            self.cost_ledger.settle(reservation, reservation)
            client_metadata = getattr(exc, "archive_metadata", {})
            if not isinstance(client_metadata, dict):
                client_metadata = {}
            failures = []
            for index, seed in enumerate(candidate_attempt_seeds, start=1):
                owner = index == 1
                metadata = referenced_metadata(index, seed, owner=owner)
                metadata.update(
                    {
                        "api_costs_microusd": reservation if owner else 0,
                        "reported_cost_microusd": (
                            None if completion is None else completion.cost_microusd
                        ),
                        "cost_basis": (
                            "ambiguous-dispatch-reservation"
                            if owner
                            else "shared-call-reference"
                        ),
                    }
                )
                if owner:
                    metadata.update(
                        {
                            "prompt": prompt,
                            "raw_response": (
                                None if completion is None else completion.raw_response
                            ),
                            "cost_uncertain": reservation > 0,
                            "cost_ledger": self.cost_ledger.to_dict(),
                            **client_metadata,
                        }
                    )
                failures.append(
                    ProposalOffspring(
                        source=None,
                        error=f"proposal host failure: {type(exc).__name__}: {exc}",
                        metadata=metadata,
                    )
                )
            self._attach_batch_replay(failures)
            return tuple(failures)

        parsed = parse_batched_proposals(completion.content, proposals_per_call)
        offspring = []
        for item, seed in zip(parsed, candidate_attempt_seeds, strict=True):
            owner = item.batch_index == 1
            metadata = referenced_metadata(item.batch_index, seed, owner=owner)
            metadata.update(
                {
                    "prompt_tokens": completion.prompt_tokens if owner else 0,
                    "completion_tokens": completion.completion_tokens if owner else 0,
                    "reported_cost_microusd": completion.cost_microusd if owner else None,
                    "api_costs_microusd": charged if owner else 0,
                    "cost_basis": (
                        self._completion_cost_basis(completion)
                        if owner
                        else "shared-call-reference"
                    ),
                }
            )
            if owner:
                metadata.update(
                    {
                        "prompt": prompt,
                        "raw_response": completion.raw_response,
                        "response_content": completion.content,
                        "cost_ledger": self.cost_ledger.to_dict(),
                    }
                )
                if self._is_subscription_engine():
                    metadata["codex_usage_ledger"] = completion.raw_response.get(
                        "accounting"
                    )
            if not item.valid:
                metadata["batch_parse_failure"] = item.error
                offspring.append(
                    ProposalOffspring(source=None, error=item.error, metadata=metadata)
                )
                continue
            source: str | None = None
            try:
                source = self._apply_completion(
                    parent.code,
                    item.content or "",
                    selected_mode,
                )
                policy = ps.Policy(source)
            except (ValueError, ps.PlacementScriptError) as exc:
                error = str(exc)
                metadata["validation_failure"] = {
                    "error": error,
                    "offending_excerpt": _validation_excerpt(
                        source,
                        error,
                        item.content or "",
                    ),
                }
                offspring.append(
                    ProposalOffspring(source=None, error=error, metadata=metadata)
                )
                continue
            metadata.update({"policy_name": policy.name, "policy_version": policy.version})
            offspring.append(ProposalOffspring(source=source, metadata=metadata))
        valid_candidates = sum(item.valid for item in offspring)
        offspring[0].metadata["batch_summary"] = {
            "requested_proposals": proposals_per_call,
            "valid_candidates": valid_candidates,
            "invalid_candidates": proposals_per_call - valid_candidates,
        }
        self._attach_batch_replay(offspring)
        return tuple(offspring)

    def __call__(
        self,
        parent: Program,
        inspirations: tuple[Program, ...],
        rng: random.Random,
    ) -> ProposalOffspring:
        if not isinstance(rng, random.Random):
            raise TypeError("LLM proposer requires the runner-owned random.Random")
        return self.propose_with_repair(parent, inspirations, rng)

    def propose_with_repair(
        self,
        parent: Program,
        inspirations: tuple[Program, ...],
        rng: random.Random,
        *,
        mode: str | None = None,
        repair: ValidationRepair | None = None,
    ) -> ProposalOffspring:
        if not isinstance(rng, random.Random):
            raise TypeError("LLM proposer requires the runner-owned random.Random")
        if repair is not None and not self.profile.grammar_aware_retry_enabled:
            raise ValueError("grammar-aware repair is disabled for this evolution profile")
        available_modes = tuple(
            mode for mode in self.proposal_modes if mode != "crossover" or inspirations
        )
        if not available_modes:
            return ProposalOffspring(
                source=None,
                error="no registered LLM proposal mode is available for this parent",
                metadata={"operator": self.name, "api_costs_microusd": 0},
            )
        if mode is not None and mode not in available_modes:
            raise ValueError("repair proposal mode is unavailable for this parent")
        selected_mode = mode or self._select_mode(available_modes, rng)
        hint_metadata: dict[str, Any] = {}
        ablation_probe = False
        if self.ablation_prompt_hint_probability > 0.0:
            hint_seed = rng.randrange(0, 2**63)
            hint_draw = random.Random(hint_seed).random()
            ablation_probe = hint_draw < self.ablation_prompt_hint_probability
            hint_metadata = {
                "ablation_prompt_hint_probability": (
                    self.ablation_prompt_hint_probability
                ),
                "ablation_prompt_hint_seed": hint_seed,
                "ablation_prompt_hint_draw": hint_draw,
                "ablation_prompt_hint_applied": ablation_probe,
            }
        prompt = self._prompt(
            parent,
            inspirations,
            selected_mode,
            profile=self.profile,
            repair=repair,
            ablation_probe=ablation_probe,
        )
        call_seed = rng.randrange(0, 2**63)
        reservation = self.track.max_call_reservation_microusd
        try:
            self.cost_ledger.reserve(reservation)
        except ValueError as exc:
            return ProposalOffspring(
                source=None,
                error=f"proposal cost budget exhausted: {exc}",
                metadata={
                    "operator": self.name,
                    "prompt": prompt,
                    "call_seed": call_seed,
                    "model_id": self.track.model_id,
                    "revision": self.track.revision,
                    "proposal_mode": selected_mode,
                    "api_costs_microusd": 0,
                    "budget_exhausted": True,
                    "cost_ledger": self.cost_ledger.to_dict(),
                    **hint_metadata,
                },
            )
        completion: LLMCompletion | None = None
        try:
            completion = self.client.complete(
                model=self.track.model_id,
                prompt=prompt,
                generation=self.track.generation,
                seed=call_seed,
            )
            if completion.cost_microusd is not None and completion.cost_microusd > reservation:
                raise ValueError("proposal host reported cost above its hard reservation")
            charged = reservation if completion.cost_microusd is None else completion.cost_microusd
            self.cost_ledger.settle(reservation, charged)
        except Exception as exc:
            # Once dispatch is attempted, an exception leaves the external charge
            # ambiguous. Conservatively consume the whole reservation so retries and
            # resumes can never exceed the cap through lost responses, unless the client
            # durably settled a known charge (e.g. a billed response without content).
            settled_cost = getattr(exc, "cost_microusd", None)
            cost_known = (
                type(settled_cost) is int and 0 <= settled_cost <= reservation
            )
            charged = settled_cost if cost_known else reservation
            self.cost_ledger.settle(reservation, charged)
            client_metadata = getattr(exc, "archive_metadata", {})
            if not isinstance(client_metadata, dict):
                client_metadata = {}
            return ProposalOffspring(
                source=None,
                error=f"proposal host failure: {type(exc).__name__}: {exc}",
                metadata={
                    "operator": self.name,
                    "prompt": prompt,
                    "call_seed": call_seed,
                    "model_id": self.track.model_id,
                    "revision": self.track.revision,
                    "proposal_mode": selected_mode,
                    "api_costs_microusd": charged,
                    "cost_uncertain": not cost_known and reservation > 0,
                    "reported_cost_microusd": (
                        None if completion is None else completion.cost_microusd
                    ),
                    "raw_response": None if completion is None else completion.raw_response,
                    "cost_ledger": self.cost_ledger.to_dict(),
                    **hint_metadata,
                    **client_metadata,
                },
            )
        metadata = {
            "operator": self.name,
            "prompt": prompt,
            "raw_response": completion.raw_response,
            "call_seed": call_seed,
            "model_id": self.track.model_id,
            "revision": self.track.revision,
            "proposal_mode": selected_mode,
            "weight_set_sha256": self.track.weight_set_sha256,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "reported_cost_microusd": completion.cost_microusd,
            "api_costs_microusd": charged,
            "cost_basis": self._completion_cost_basis(completion),
            "cost_ledger": self.cost_ledger.to_dict(),
            **hint_metadata,
        }
        if self.engine_descriptor is not None:
            metadata["engine_descriptor"] = self.engine_descriptor
        if self._is_subscription_engine():
            metadata["codex_usage_ledger"] = completion.raw_response.get("accounting")
        source: str | None = None
        try:
            source = self._apply_completion(parent.code, completion.content, selected_mode)
            policy = ps.Policy(source)
        except (ValueError, ps.PlacementScriptError) as exc:
            error = str(exc)
            metadata["validation_failure"] = {
                "error": error,
                "offending_excerpt": _validation_excerpt(
                    source,
                    error,
                    completion.content,
                ),
            }
            return ProposalOffspring(source=None, error=str(exc), metadata=metadata)
        metadata.update({"policy_name": policy.name, "policy_version": policy.version})
        return ProposalOffspring(source=source, metadata=metadata)


__all__ = [
    "ABLATION_PROMPT_HINT",
    "CompletionClient",
    "LLMCompletion",
    "OpenWeightLLMOperator",
    "ParsedBatchProposal",
    "PROPOSAL_MODES",
    "ValidationRepair",
    "VLLMCompletionClient",
    "parse_batched_proposals",
]
