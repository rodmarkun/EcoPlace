# Durable transport for the new matched study

`DurableOpenRouterClient` retains the production OpenRouter parser, pinned token-price accounting
and HTTPS transport. It adds a configuration/source-bound hash-chain journal, exclusive process
locking, and flushed/fsynced request intent and successful completion records. API credentials
are not journaled. This is a new transport; historical clients and receipts remain unchanged.

Use `--proposal-journal PATH --proposal_engine openrouter` through the matched-control CLI.
The new config must specify `max_attempts: 1`. Automatic HTTP retries are disallowed because a
failed response does not prove that the provider failed to accept or bill the request. Exact
requests with saved responses replay without a new HTTP request or an extra transport charge.
Costs survive process restart; configuration, source or journal-integrity drift is rejected.
New campaigns identify this operator as `openrouter_llm`, leaving the historical misleading
`open_weight_qwen` records intact.

An intent without durable completion is **uncertain**, including an interruption before actual
dispatch. The conservative outcome is a process stop through `UnresolvedDispatch` (a `SystemExit`
subclass), bypassing proposal repair handlers that catch `Exception`. The reservation remains
in the journal. No automatic reconciliation tool or provider idempotency guarantee is claimed.
Reconcile provider outcome and cost with retained evidence before any explicit recovery; never
delete the intent to make a campaign resume. A response saved before the campaign's own candidate
record can be replayed into that pending candidate. The completed subprocess CLI pilot verifies
this interaction with the campaign ledger and paired initialization barrier; see
`../matched_cli_pilot_20260909/README.md`. The conservative operational policy is in
`RECONCILIATION.md`.

Nine tests cover exact response/charge replay, process interruption, response-persistence failure,
torn/modified journals, competing clients, changed pricing, retry refusal and budget restoration.
The HTTPS test starts a loopback-only TLS server with a temporary certificate and dummy API key,
then uses two separate client processes. Exactly one HTTP request is observed. It requires local
socket permission; the sandbox denied binding, and the permitted run outside it passed all nine
tests. It neither contacts OpenRouter nor exercises a real model. The remaining tests use fake
dispatch with the real response parser/accounting.

These are local process-crash durability checks, not a filesystem power-loss guarantee. Costs are
computed from pinned token prices using the existing transport, not independently verified bills.
No paid study has been launched through this transport.

The journal source binding covers this wrapper file. Underlying transport, parser, evaluator and
native dependencies must be bound by the full campaign source/environment snapshot. The journal
alone does not authenticate the full executable dependency closure.
