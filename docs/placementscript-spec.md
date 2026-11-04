# PlacementScript — language specification

**Version:** 1.0.
**Normative implementation:** `runtime/crates/placementscript-core` — the *single* interpreter
(C18). This document specifies the language; the conformance suite (`runtime/conformance`) is the
executable regression guard that the one implementation keeps matching this
document across its two build targets (native, `wasm32-wasip1`).

PlacementScript is a statically-typed, capability-free subset of Python surface syntax in
which all evolved placement policies are written. Design goals, in priority order:

1. **Determinism** — a program's result is a pure function of (source, arguments, limits).
   No clocks, no I/O, no OS capabilities, no unordered collections, no float
   nondeterminism (NaN cannot exist at runtime; see §6.3).
2. **Bounded execution** — every invocation runs under an explicit fuel budget and memory
   budget defined *here* (§8), not incidentally by the runtime.
3. **Auditability** — policies read as plain Python-like code (~30–150 lines); promotion
   diffs are code-review material.
4. **Evolvability** — small typed grammar suitable for LLM diff mutation, typed-subtree GP
   crossover (C19), and numeric-literal patching.

Where PlacementScript diverges from Python semantics, the divergence is listed in §10.

---

## 1. Lexical structure

- **Encoding:** UTF-8 source; only ASCII is significant outside comments and string literals.
- **Lines and blocks:** statements are newline-terminated; blocks are indentation-delimited
  (Python-style `INDENT`/`DEDENT`). Indentation is spaces only; a tab character anywhere in
  the source (outside string literals) is a parse error, as is a carriage return (LF line
  endings only). A block's statements must share one exact indentation column.
- **Comments:** `#` to end of line.
- **Blank lines** are ignored between statements.
- **Keywords:** `def return if elif else for in range and or not True False pass`.
  Type names `int float bool str list` and record names (§4) are reserved.
- **Identifiers:** `[A-Za-z_][A-Za-z0-9_]*`, not a keyword/reserved name.
- **Integer literals:** decimal digits with optional `_` separators (`1_000`). Must fit in
  `i64`; a literal exceeding `i64::MAX` when it appears without a leading `-` is a **parse
  error** (there are no negative literals; `-x` is unary negation, so `i64::MIN` is not
  writable as a literal — see §6.2).
- **Float literals:** decimal forms `1.5`, `.5`, `1.`, `1e9`, `2.5E-3` (optional `_`
  separators in digit runs). The literal is converted by standard (round-to-nearest) decimal
  → binary64 conversion; if the result is not **finite** (overflows to ±inf), it is a
  **parse error**. `nan`, `inf`, and `float("nan")` are not expressible: NaN is rejected at
  parse time by construction (C18).
- **String literals:** `"..."` or `'...'` on one line; escapes `\\ \" \' \n \t`. No other
  escapes; no f-strings; no concatenation of adjacent literals.
- **Continuation:** expressions may span lines *only* inside unclosed `(`, `[` brackets
  (implicit continuation, as in Python). No backslash continuation.

## 2. Grammar

EBNF over the token stream (NEWLINE/INDENT/DEDENT produced by the lexer):

```
program      = { NEWLINE | const_def | func_def } EOF ;
const_def    = NAME ":" type "=" literal NEWLINE ;            (* module level only *)
literal      = INT | FLOAT | STRING | "True" | "False" | "-" INT | "-" FLOAT ;
func_def     = "def" NAME "(" [ param { "," param } ] ")" "->" type ":" block ;
param        = NAME ":" type ;
type         = "int" | "float" | "bool" | "str"
             | "list" "[" type "]"
             | "Pod" | "NodeState" | "CallGraphCtx" | "ClusterCtx" | "ClusterState" | "Move" ;
block        = NEWLINE INDENT stmt { stmt } DEDENT ;
stmt         = simple_stmt NEWLINE | if_stmt | for_stmt ;
simple_stmt  = assign | aug_assign | return_stmt | "pass" | append_stmt ;
assign       = NAME [ ":" type ] "=" expr ;
aug_assign   = NAME aug_op expr ;
aug_op       = "+=" | "-=" | "*=" | "/=" | "//=" | "%=" ;
append_stmt  = NAME "." "append" "(" expr ")" ;
return_stmt  = "return" expr ;
if_stmt      = "if" expr ":" block { "elif" expr ":" block } [ "else" ":" block ] ;
for_stmt     = "for" NAME "in" iterable ":" block ;
iterable     = "range" "(" expr [ "," expr ] ")" | expr ;    (* expr must be list[T] *)
expr         = or_expr ;
or_expr      = and_expr { "or" and_expr } ;
and_expr     = not_expr { "and" not_expr } ;
not_expr     = "not" not_expr | comparison ;
comparison   = arith [ cmp_op arith ] ;                       (* NOT chained; see §10 *)
cmp_op       = "<" | "<=" | ">" | ">=" | "==" | "!=" ;
arith        = term { ("+" | "-") term } ;
term         = unary { ("*" | "/" | "//" | "%") unary } ;
unary        = "-" unary | postfix ;
postfix      = atom { "." NAME | "[" expr "]" } ;
atom         = INT | FLOAT | STRING | "True" | "False" | NAME
             | NAME "(" [ expr { "," expr } ] ")"             (* builtin / user function call *)
             | "(" expr ")"
             | "[" [ expr { "," expr } ] "]" ;                (* list display *)
```

Deliberately absent (grammatically impossible, per C2): `import`, `while`, `class`,
`lambda`, comprehensions, `with`, `try`, `raise`, `yield`, `global`/`nonlocal`, decorators,
keyword arguments, default parameters, tuple (un)packing, slicing, chained comparison,
chained assignment, string formatting, attribute *assignment*, `del`, dictionaries, sets.
There is no way to express I/O, reflection, recursion (§5.4), or unbounded loops (§5.3).

## 3. Types

Primitive: `int` (64-bit signed integer), `float` (IEEE-754 binary64, **always finite**,
§6.3), `bool`, `str` (immutable; equality only). Composite: `list[T]` for any type `T`
(homogeneous, mutable only via `append`), and the six **record types** of §4 (immutable;
field access only). There is no `None`, no optional types, no unions, no user-defined types.

- No implicit conversions anywhere. `int` and `float` never mix in an operation
  (§6.1); use `float(i)` / `int(x)` explicitly.
- `bool` is not an `int` (divergence from Python, §10).
- Every expression has exactly one static type, determined at type-check time.

## 4. Record types

Records are constructed by the host (simulator, scheduler plugin) and passed in as
arguments; policies read fields and construct only `Move` values (via the `move_pod` and
`wake` builtins, §7). Field access is `value.field`. Unknown fields are type errors.

Index conventions: `ClusterCtx.nodes` / `ClusterState.nodes` are indexed by node id
(`nodes[i].id == i`) and `.pods` by pod id (`pods[i].id == i`). Hosts MUST maintain these
invariants; the interpreter does not re-verify them. `NodeState.latency_to_ms` is indexed by
node id and has one entry per node.

| Record | Field | Type | Meaning |
|---|---|---|---|
| `Pod` | `id` | `int` | Pod id, dense `0..P` |
| | `service` | `int` | Service id the pod is a replica of |
| | `app` | `int` | Application id |
| | `cpu_request` | `int` | CPU request, millicores |
| | `mem_request` | `int` | Memory request, MiB |
| | `stateful` | `bool` | Pod holds local state (never proposed for moves, C8) |
| | `has_pdb` | `bool` | Covered by a PodDisruptionBudget at its limit (excluded from proposals, C8) |
| | `image_mb` | `int` | Container image size, MiB (pull-cost signal) |
| | `current_node` | `int` | Node id where the pod runs; `-1` if pending |
| `NodeState` | `id` | `int` | Node id, dense `0..N` |
| | `tier` | `int` | `0` cloud, `1` fog, `2` far edge |
| | `cpu_capacity` | `int` | Allocatable CPU, millicores |
| | `cpu_allocated` | `int` | Sum of scheduled pods' CPU requests, millicores |
| | `mem_capacity` | `int` | Allocatable memory, MiB |
| | `mem_allocated` | `int` | Sum of scheduled pods' memory requests, MiB |
| | `power_idle_w` | `float` | Idle wall power, watts (fitted, G1/C17) |
| | `power_max_w` | `float` | Full-utilization wall power, watts |
| | `suspended` | `bool` | Node is currently suspended/powered off |
| | `suspendable` | `bool` | Node class passed the G1 suspension gate |
| | `latency_to_ms` | `list[float]` | RTT to every node, ms, indexed by node id |
| `CallGraphCtx` | `src` | `list[int]` | Edge source **pod id** (columnar edge list) |
| | `dst` | `list[int]` | Edge destination pod id |
| | `rate` | `list[float]` | Edge call rate, calls/s |
| | `payload_kb` | `list[float]` | Mean payload per call, KB |
| `ClusterCtx` | `nodes` | `list[NodeState]` | All nodes, indexed by id |
| | `pods` | `list[Pod]` | All pods, indexed by id |
| | `graph` | `CallGraphCtx` | Cluster-wide call graph |
| | `now_ms` | `int` | Scenario-clock timestamp, ms (host-provided, deterministic) |
| `ClusterState` | `nodes` | `list[NodeState]` | All nodes, indexed by id |
| | `pods` | `list[Pod]` | All pods, indexed by id |
| `Move` | `kind` | `int` | `0` = move pod, `1` = wake node |
| | `pod` | `int` | Pod id to move (`-1` for `wake`) |
| | `dest` | `int` | Destination node id (move) / node id to wake |

The three parallel lists in `CallGraphCtx` (and `latency_to_ms`) keep the serialized state
columnar and cheap to build; hosts MUST keep the parallel lists equal-length.

### 4.1 Host input domain (validated at the boundary)

The index conventions above and the bounds below define the **host input domain**: what a
policy is guaranteed to see. They are *enforced* by shared validators in the core crate
(`bundle::validate_score_inputs` / `validate_propose_inputs`), invoked by every embedding
(PyO3 and the wasm ABI) before the interpreter runs — a violating input is a **host
error** (§9), never the policy's fault. The property-based risk gates
quantify scorer/proposer totality over exactly this domain.

The shape bounds are **jointly representable in the WASM sandbox**: latency vectors make
cluster state quadratic in the node count, so the domain maximum is sized to fit the
default 256 MiB linear-memory ceiling with the encoded request, the decoded wire tree,
and the runtime values resident at once. A domain-maximum request is executed end-to-end
against the wasm guest in the conformance suite; the host additionally rejects any
encoded request over its `max_request_bytes` ABI ceiling as a structured host error
before it touches guest memory. Experiment scales (testbed ~10 nodes, simulated
topologies of a few hundred) sit far below these caps.

| Quantity | Bound |
|---|---|
| `len(nodes)` (ctx/state, and candidates per score batch) | 1 … 512 (batch may be 0 … 512) |
| `len(pods)` | 0 … 16 384 |
| Call-graph edges | 0 … 65 536, equal-length columns, `src`/`dst` valid pod ids |
| `propose_moves` budget | 0 … 16 384 |
| `cpu_*`, `mem_*`, `image_mb` (all integer resources) | 0 … 2⁴⁰ (so any per-pod sum fits `i64` without overflow) |
| `tier` | 0 … 15 |
| `service`, `app` | 0 … 2³²−1 |
| `power_*_w` | 0 … 10⁶ (finite) |
| `latency_to_ms` entries | 0 … 10⁷ (finite), one entry per node |
| `rate` | 0 … 10⁹ (finite) |
| `payload_kb` | 0 … 10⁹ (finite) |
| `now_ms` | 0 … 2⁵³ |
| `nodes[i].id == i`, `pods[i].id == i` | dense, index-aligned |
| `pod.current_node` | −1 … `len(nodes)`−1 |
| Batch candidate `NodeState`s | `0 ≤ id < len(ctx.nodes)`; `latency_to_ms` sized to `len(ctx.nodes)` |

## 5. Programs, functions, scoping

### 5.1 Module structure

A program is a sequence of module-level constant definitions and function definitions.
Order matters (§5.4). Module-level constants must be annotated and initialized with a
literal (optionally negated); they are immutable and visible in every function.

### 5.2 Functions

Every function declares parameter types and a return type. **Every control path must end in
`return`** (checked statically; a function whose `if`/`else` arms don't all return and that
has no trailing `return` is a type error). Arguments are passed by reference for lists and
records, by value for primitives. Records are immutable, and lists admit only `append`,
with one further restriction: **host-provided state is read-only.** Every list that crosses
the host boundary (an argument, or anything reachable from one — e.g. `ctx.pods`) is
*frozen*; `append` to a frozen list (directly or through an alias) is a **trap**
(`trap: append to a host-provided list (read-only)`). This is what makes `score`
side-effect-free across the candidates of one scheduling cycle (C2): a policy cannot
mutate shared state that another candidate's evaluation would observe. Policy-created
lists are ordinary append-mutable lists; mutation through an alias of a policy-created
list is well-defined (same object).

### 5.3 Statements and definite assignment

- `x = e` — first assignment in a function *declares* `x` with the type of `e` (or the
  annotated type, which must match). Later assignments must keep the same type.
  `x: list[T] = []` — an empty list display requires an annotation (its element type is
  otherwise unknown); this is the only place an annotation is *required*.
- A variable may be read only where it is **definitely assigned**: assigned on every
  control path reaching the read (standard forward dataflow; `if` without `else` does not
  definitely assign). Violations are type errors.
- `for x in range(a)` / `range(a, b)` iterates integers `a <= x < b` (i.e. `0 <= x < a` in
  the 1-argument form) in ascending order. `for x in xs` iterates a `list[T]` in list
  order, front to back. The iterable expression / range bounds are evaluated **once**,
  before the first iteration; appending to the iterated list during iteration is a **trap**
  (`trap: list mutated during iteration`). The loop variable is scoped to the loop body and
  is not visible after the loop (divergence, §10); it may not shadow an existing variable.
- Loops are the only repetition construct; both forms are bounded by data (`len(xs)` or
  `b - a` iterations). With no `while` and no recursion, all loops terminate; fuel (§8) is
  the defense-in-depth backstop, not the termination argument.
- `xs.append(v)` appends in place; it is a statement, not an expression (`append` returns
  nothing and cannot be nested in an expression). Appending to a *frozen* (host-provided)
  list traps (§5.2).
- There is no `break`/`continue` (v1; a candidate must restructure with flags/conditionals).

### 5.4 Calls and the no-recursion rule

A function body may call **builtins (§7)** and **user functions defined earlier in the
file**. Calling a function at or below its own definition point (including itself) is a
type error (`recursion is not expressible`). This makes the call graph a DAG by
construction — mutual recursion is impossible, and the maximum call depth is the number of
functions in the file.

### 5.5 The PolicyBundle contract (C22)

A **PolicyBundle** is a single PlacementScript source file (one genome) defining:

1. `def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float` — per-pod scoring
   policy, the Scheduling Framework Score contract. Called once per (pod, feasible node);
   **feasibility is delegated to native Filter plugins** (C2) — `score` ranks, it never
   gates. Higher is better. The returned float is always finite (§6.3).
2. `def propose_moves(state: ClusterState, graph: CallGraphCtx, budget: int) -> list[Move]`
   — rebalancing proposal policy (C3), run outside the scheduler hot path. `budget` is the
   maximum number of **actions** (`move_pod` and `wake` combined) the actuation layer will
   execute this window; returning more than `budget` actions is **not** a language error
   (the property gates and the actuation layer enforce/verify it) but is a
   bundle-contract violation. Proposals naming
   a suspended node as destination are valid only if the same list contains a prior
   `wake(node)` action for it; `wake` is an explicit action that the actuation layer prices
   with measured wake energy/latency (C17/C22).
3. **Shared helpers** — any other functions, callable from both (subject to §5.4 ordering).
4. **Version metadata** — required module constants `POLICY_NAME: str` and
   `POLICY_VERSION: str`. Absence of either, or of either required function, or a signature
   mismatch, is a **bundle validation error** (reported in the `Type` error class, §9).

The host-side tie-breaking rule is part of the bundle contract: when the scheduler
(emulator or plugin host) compares candidate nodes and `score` ties exactly (bitwise-equal
`f64`, after `-0.0 → 0.0` normalization on the score boundary), **the lowest node id
wins**, regardless of candidate order. The rule is implemented once, in the core crate
(`bundle::select_node`), and every host selects through that one function (C18 parity);
the wasm ABI returns the selection as `best_node`, and the conformance suite checks it
across both build targets.

## 6. Operational semantics

Evaluation is strict, left-to-right, innermost-first, single-threaded. Argument lists,
list displays, and binary operands evaluate left to right. `and`/`or` short-circuit (and
are the only non-strict constructs). All effects are: local variable binding, list
`append`, fuel/memory consumption, and (on error) a trap that aborts the whole invocation
— there is no catch construct; any trap terminates the policy call with a structured error
(§9).

### 6.1 Typing of operators

| Operator | Operand types | Result | Notes |
|---|---|---|---|
| `+ - *` | `int,int` → `int`; `float,float` → `float` | | mixed int/float is a type error |
| `/` | `float,float` → `float` | | **not defined on int** (use `//` or `float()`) |
| `// %` | `int,int` → `int` | | floor division / Python-sign modulo (§6.2) |
| unary `-` | `int` → `int`; `float` → `float` | | |
| `< <= > >=` | `int,int`; `float,float` | `bool` | no ordering on str/bool/list/record |
| `== !=` | `int`, `float`, `bool`, `str` (same type both sides) | `bool` | not defined on lists/records |
| `and or not` | `bool` | `bool` | short-circuit |
| `xs[i]` | `list[T]`, `int` | `T` | trap if `i < 0` or `i >= len(xs)` (no negative indexing) |
| `x.field` | record | field type | statically checked |

### 6.2 Integer semantics (checked, overflow = trap)

`int` is `i64`. Every arithmetic operation is **checked**: `+ - * //  %` and unary `-`
trap on overflow (`trap: integer overflow`). The two extra trap cases: `x // 0` and
`x % 0` trap (`trap: division by zero`); `i64::MIN // -1`, `i64::MIN % -1`, `-i64::MIN`,
and `abs(i64::MIN)` trap as overflow. `//` is **floor** division and `%` takes the sign of
the divisor (Python semantics, e.g. `-7 // 2 == -4`, `-7 % 2 == 1`), so
`(a // b) * b + a % b == a` for all non-trapping inputs.

### 6.3 Float semantics (total order, no NaN, always finite)

`float` is IEEE-754 binary64 restricted to **finite values**. The invariant is maintained
end-to-end:

- **Parse:** literals that are not finite are parse errors (§1); NaN is not writable.
- **Runtime:** after every float-producing operation (`+ - * /`, `sqrt`, `float(int)`), the
  result is checked; a non-finite result **traps** (`trap: non-finite float result`,
  covering both overflow-to-infinity and NaN, e.g. `1e308 * 10.0`, `x / 0.0` for any `x`,
  `sqrt(-1.0)`). Note `x / 0.0` traps *as a non-finite result* — float division by zero
  never yields ±inf or NaN.
- **Host boundary:** hosts MUST reject non-finite floats in inputs (the FFI/ABI layers
  validate this; a non-finite input is an argument error, §9), and are guaranteed
  never to receive one as output.
- **Ordering:** since NaN cannot exist, IEEE comparison is a **total order** on
  PlacementScript floats. `-0.0` may exist and compares equal to `0.0` (IEEE). So that the
  host tie-breaking rule (§5.5, "bitwise-equal scores tie") agrees with IEEE equality,
  hosts normalize `-0.0` to `0.0` on the `score` output boundary; inside the language,
  `-0.0 == 0.0` is simply `True`.
- Subnormal values are ordinary finite values (no flush-to-zero; IEEE gradual underflow;
  an operation underflowing to a subnormal or to `0.0` does **not** trap).
- Float arithmetic is required to be bit-exact IEEE binary64 (round-to-nearest-even) on
  every build target; `sqrt` is the IEEE correctly-rounded square root. No FMA
  contraction, no fast-math, no reassociation. (Both native and wasm32 targets provide
  exactly this; the conformance suite checks witness values bit-exactly.)

### 6.4 Determinism

For fixed (source, arguments, limits), every build target of the interpreter produces the
identical result — same value bits, same error class, same fuel consumed, same memory
accounted. There is no observable clock, randomness, address, hash order, or environment.
This is the C18 property the conformance suite regression-guards.

## 7. Builtins

| Signature | Semantics | Fuel |
|---|---|---|
| `len(xs: list[T]) -> int` | list length | 1 |
| `abs(x: int) -> int` | trap on `i64::MIN` | 1 |
| `abs(x: float) -> float` | | 1 |
| `min(a, b)` / `max(a, b)` | both `int` or both `float`; **returns `a` on tie** (incl. `-0.0`/`0.0` ties) | 1 |
| `sqrt(x: float) -> float` | IEEE sqrt; trap if `x < 0` (result would be NaN) | 1 |
| `float(x: int) -> float` | round-to-nearest (always finite) | 1 |
| `int(x: float) -> int` | truncate toward zero; trap if the truncated value does not fit `i64` | 1 |
| `move_pod(pod_id: int, dest: int) -> Move` | constructs `Move{kind:0, pod:pod_id, dest:dest}` | 1 |
| `wake(node_id: int) -> Move` | constructs `Move{kind:1, pod:-1, dest:node_id}` | 1 |

`min`/`max` are 2-argument only (fold in a loop for lists). No other builtins exist; in
particular no `print`, no `sorted` (write a selection loop; determinism of user-written
selection is then explicit in the policy source), no `sum` (explicit loops keep fuel
accounting visible and give GP typed subtrees more material).

## 8. Fuel model and memory accounting

Both budgets are **per invocation** (one `score` or one `propose_moves` call), fixed
before the call by the host, and part of this specification: every host (simulator FFI,
wasmtime plugin host, conformance runner) enforces the same accounting rules below, because
they are implemented once, in the interpreter (C18). Exhaustion aborts the invocation with
`FuelExhausted` / `OutOfMemory` (§9) — never a partial result.

### 8.1 Fuel

Fuel is a count of abstract interpreter steps, charged **before** each step executes:

| Event | Cost |
|---|---|
| Statement dispatch (each `assign`/`aug_assign`/`return`/`pass`/`append`/`if`/`for` reached) | 1 |
| Expression node evaluation (literal, name, unary, binary, comparison, index, field, `and`/`or` node) | 1 |
| List display `[e1, .., en]` | 1 + n |
| Builtin call | 1 (after argument evaluation; each argument already paid its own costs) |
| User function call | 10 (call overhead) + callee's own costs |
| Loop iteration (each time the body is entered, plus the final bound-check that exits) | 1 |
| `range` header evaluation | 1 (plus bound expressions' own costs) |

**Load budget and structural limits.** Module load (compile) runs under its own `Limits`.
Load *fuel* is charged **one step per source byte plus one step per AST node plus one
step per module constant**. Source bytes are charged *before parsing begins*: a source
longer than the load-fuel budget is rejected as `FuelExhausted` before a single byte is
scanned (outcome-equivalent to the final accounting — an over-length source can never
load — and it makes lexer work and allocation O(budget) unconditionally, one long token
included), and exhaustion is additionally checked before every token. A pathologically
token-dense or oversized source therefore fails with `FuelExhausted` after O(budget)
work and allocation, never by exhausting compiler memory or time. The load fuel budget therefore bounds total program size,
and an oversized module fails at load with `FuelExhausted`, never by stressing the
interpreter later. Load *memory* is charged for interned string literals (§8.2). In
addition, five **structural limits** are enforced at load; together they statically
bound the height of the built AST and hence parser, type-checker, and interpreter
recursion depth on every build target (a legal program can never overflow the native or
wasm stack — a malformed one fails with a structured error instead, and an over-deep
tree is never even constructed):

| Limit | Value | Error class |
|---|---|---|
| Function definitions per module | 64 | `Type` ("too many functions") |
| Static call-chain depth (call graph is a DAG of earlier-defined functions, §5.4, so this is exact) | 16 | `Type` ("call chain too deep") |
| Expression nesting depth — the **exact height of every built expression tree** (`1 + max(child heights)`, computed compositionally as each node is constructed; an over-limit node is rejected before it is built, so the bound holds across precedence layers and parenthesization — `a + b + c` is a left-leaning tree of height 3, `x[0][0]` a spine of height 3, and nesting operator chains inside parentheses cannot stack heights past the limit). Additionally, each *open parser context* (parenthesized group, call argument, list element, index expression, `not`, unary `-`, comparison, and each additional operand of an operator/postfix chain) counts one level against the same limit while open, bounding the parser's own recursion (e.g. `((((x))))` collapses to height 1 but recurses per paren) | 32 | `Parse` ("expression nesting too deep") |
| Type nesting depth (each `list[...]` level) | 8 | `Parse` ("type nesting too deep") |
| Block nesting depth (each `if`/`elif`/`else`/`for` body level) | 8 | `Parse` ("block nesting too deep") |

Module-level constant initialization is charged to the *load* budget — not to
invocations.

**Default budgets** (hosts may lower, and the conformance/property suites may raise; the
defaults ship in `Limits::default_score()` / `Limits::default_propose()`):

| Invocation | Fuel | Rationale |
|---|---|---|
| `score` | 200,000 | ~10–50 µs interpreted; ×400 feasible nodes per cycle stays well under the ≤5 ms decision budget (hypothesis to be *measured*, E6/C9 — never claimed) |
| `propose_moves` | 20,000,000 | off-hot-path controller (C3); allows O(pods × nodes) scans on L1-scale clusters |
| load (module init) | 100,000 | metadata + constants only |

### 8.2 Memory accounting

Memory is accounted as **cumulative allocation** during the invocation (monotonic; no
credit on release — simple, deterministic, and identical across targets; the fuel budget
independently bounds how fast allocation can happen). Charged events:

| Event | Bytes charged |
|---|---|
| List creation (display or `[]`) | 32 + 16 × (number of elements) |
| `append` | 16 |
| `Move` construction (`move_pod` / `wake`) | 32 |
| String literal (interned at load, one interning per occurrence, charged to the **load** memory budget: 32 + byte length) | 0 per evaluation (evaluation clones the interned reference; no allocation) |
| Every value binding/copy (int/float/bool/record/list *reference*) | 0 (machine-word values; bounded by fuel) |

Host-constructed arguments (the cluster state) are **not** charged against the policy's
budget — the policy is charged for what it allocates, not for what it observes.

**Default budgets:** `score`: 1 MiB; `propose_moves`: 16 MiB; load: 64 KiB.

### 8.3 Decision-latency budget

The end-to-end scheduling-cycle budget is **≤ 5 ms** (plan §3.5). This spec deliberately
does not translate that into fuel (wall-time per fuel unit is hardware-dependent);
the property harness measures wall-clock per `score` batch and gates on the
measured value; E6 reports it. Fuel defaults above are sized so that fuel exhaustion, not
wall-clock, is the binding constraint for well-formed policies.

## 9. Errors (structured taxonomy)

Every failure is exactly one of five classes — the distinction is load-bearing for
evolution feedback (the ladder branches on it):

| Class | Meaning | Examples |
|---|---|---|
| `Parse` | source is not a well-formed program | syntax error, tab indentation, non-finite float literal, oversized int literal |
| `Type` | well-formed but ill-typed, **or bundle-contract violation** (§5.5.4) | mixed `int+float`, use of unassigned variable, missing `return` path, recursion attempt, missing `score`, wrong signature, missing `POLICY_NAME` |
| `Trap` | checked runtime fault | integer overflow, `//`/`%` by zero, non-finite float result, index out of bounds, list mutated during iteration, append to a frozen host-provided list |
| `FuelExhausted` | fuel budget exceeded | |
| `OutOfMemory` | memory budget exceeded | |

`Parse` and `Type` carry a 1-based line number and message. `Trap` carries a message
naming the fault. Host-boundary argument problems (wrong arity from FFI, non-finite float
input, malformed serialized state) are **host errors**, reported by the embedding layer
(PyO3 exception / ABI error code), not part of the language taxonomy — a policy is never
blamed for a malformed host input.

## 10. Divergences from Python (informative)

For readers/LLMs assuming Python semantics:

1. `int`/`float` never mix; `/` is float-only; `//`,`%` int-only. No `**`.
2. Integer overflow traps (Python has bignums).
3. No NaN/inf; float ops trap on non-finite results (incl. `x / 0.0`, which raises in
   Python only for `x == 0` … actually Python raises `ZeroDivisionError` for any `x / 0.0`
   — PlacementScript traps identically, but via the finiteness rule).
4. `bool` is not `int`; `True + 1` is a type error.
5. No chained comparison (`a < b < c` is a parse error).
6. Loop variables are scoped to the loop body; no leakage after the loop.
7. Negative list indices trap instead of indexing from the end; no slicing.
8. `min`/`max` take exactly two arguments and return the *first* on ties.
9. No `break`/`continue`, `while`, comprehensions, `None`, keyword args, defaults.
10. Variables are statically typed from first assignment; re-binding to another type is an
    error.
11. Empty list literals require a type annotation.
12. `append` is a statement; user functions cannot return "nothing" (every function returns
    a typed value).

## 11. Versioning

This document is versioned with the crate (`placementscript-core`). Grammar or semantics
changes bump the language minor/major version; the conformance suite pins the behavior of
each released version. `PolicyBundle` metadata (`POLICY_NAME`/`POLICY_VERSION`) versions
the *policy*; the language version of a bundle is recorded by the tracking store at
evaluation time.
