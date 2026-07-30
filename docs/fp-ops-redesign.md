# FP-Ops redesign: a typed unary async pipeline

- Status: implementation design
- Target: the next breaking FP-Ops release, delivered as a direct replacement
- Inputs: `architecture-audit.md` and `fp-concept-naming-audit.md`
- Final downstream migration and acceptance consumer: Silk

## Executive target architecture

FP-Ops will become a typed, immutable, asynchronous pipeline library centered on one unary abstraction:

```text
Operation[A, B]
  = A -> Environment -> Awaitable[Result[B, Exception]]
```

`Operation` is an input/output arrow, not a suspended nullary computation, a general DAG, or an argument-binding DSL. Its public interface is deliberately small: construct an operation, compose it with `.then`, and execute it with `.run`. Higher-order, recovery, collection, object, retry, and parallel behavior are named combinators built on that core.

The successful value is the only value passed from one stage to the next. `None` is ordinary data. A separate read-only `Environment` supplies capabilities. The runner never infers control behavior from a successful value's runtime type and never replaces the environment.

The default error behavior is a fail-fast Either-like channel using the existing `expression.Result[B, Exception]`. Lossy and all-settled behavior is available only through names and return types that disclose the policy.

The implementation uses an immutable linear AST with explicit nodes for the few behaviors that require interpreter support. It does not expose graph ports, edge types, scheduling metadata, or internal callable adapters.

This produces one deep module. Callers get leverage from one composition rule, one execution interface, one async normalizer, and one error boundary. Maintainers get locality because invocation, cancellation, environment lookup, and error conversion are implemented and tested once.

## Goals

- Make runtime behavior match the published type at every public seam.
- Make single-stage and multi-stage execution use identical input semantics.
- Preserve `None`, returned domain objects, and returned `Operation` objects as ordinary values unless an explicit combinator says otherwise.
- Normalize synchronous functions, async functions, and sync functions that return awaitables through one implementation.
- Preserve exception causes across pipelines, collection traversal, and object construction.
- Make a read-only capability environment explicit at `.run`.
- Give fail-fast, all-settled, recovery, and lossy policies different names.
- Support mypy and Pyright/Pylance as consumer-contract gates.
- Finish FP-Ops completely before using Silk as the final downstream migration
  and end-to-end acceptance test.
- Delete obsolete implementation and tests as soon as the replacement interface
  satisfies its target contracts.
- Retain ergonomic left-to-right pipeline syntax where it is truthful.

## Non-goals

- Implementing a general dataflow DAG, multi-tail graph, or port-routing engine.
- Claiming a Monad, Applicative, Validation, Arrow, or Alternative instance for the entire public surface.
- Encoding an error-type union as a third `Operation` type parameter.
- Providing arbitrary typed partial application for every Python signature.
- Treating environment replacement as hidden state.
- Making mutable browser handles transitively immutable.
- Automatically offloading every synchronous callable to a worker thread.
- Preserving legacy behavior, namespaces, call forms, or helper semantics.
- Shipping a compatibility adapter, dual implementation, or deprecation runway.
- Replacing `expression.Result` during this redesign.

## Current diagnosis

The present modules expose a graph while execution carries one linear `last_result`; graph metadata therefore adds interface and implementation complexity without graph leverage. Binding depends on graph size, so adding an identity stage can change argument precedence. Context is a second implicit data channel whose updates are selected by `isinstance` checks on successful outputs.

The error interface is equally fragmented. Ordinary execution is fail-fast, but helpers also swallow errors, convert them to `None`, remove items, retry, recover, or allow setup exceptions to escape. Callable representation changes both async and failure behavior.

`Operation[P, R]` describes a wrapped Python callable better than it describes a pipeline. It cannot preserve the relationship between the output of one stage and the unary input of the next. `Any`, broad overloads, and private graph inspection erase the relationships consumers need.

These are semantic defects, not annotation gaps. The new core must land before the old overloads, graph, or helper names are repaired in place.

## Chosen mathematical semantic model

For explanation only, the context-free kernel is a Kleisli arrow over an async Either-like effect:

```text
M X = Awaitable[Result[X, Exception]]
Operation[A, B] = A -> M B
```

With the environment included:

```text
Operation[A, B] = A -> Environment -> M B
```

Sequential composition is:

```text
(f.then(g))(a, env):
  Error(e) <- f(a, env)  => Error(e)
  Ok(b)    <- f(a, env)  => g(b, env)
```

The same environment value is presented to both stages. It is neither emitted as output nor changed by the interpreter. Effects performed through environment capabilities remain effects; the model does not claim referential transparency for I/O operations.

The following narrow laws are design targets:

- left and right identity for `identity` over unary operations;
- observational associativity of `.then` under the same input and environment;
- fail-fast short-circuiting for sequential composition;
- ordinary value preservation, including `None`, `Result`, `BaseContext`, and `Operation` instances returned by raw-value operations.

The laws compare observable `Ok`/`Error` outcomes and effect traces, not Python object equality. They do not apply to retry timing, lossy traversal, parallel completion order, mutable external capabilities, or user functions that depend on hidden global state.

`fanout_parallel` is a same-input concurrent product, not Validation or a full Applicative interface. `or_else` is recovery-by-alternative, not an asserted Alternative instance. `flat_map` is an explicit dynamic-effect operation, not evidence that every `Operation[A, B]` forms one monad.

## Proposed public `Operation` interface

The conceptual interface is:

```python
A_contra = TypeVar("A_contra", contravariant=True)
B_co = TypeVar("B_co", covariant=True)
C = TypeVar("C")

class Operation(Generic[A_contra, B_co]):
    async def run(
        self,
        value: A_contra,
        *,
        environment: Environment = EMPTY_ENV,
    ) -> Result[B_co, Exception]: ...

    def then(self, next: Operation[B_co, C]) -> Operation[A_contra, C]: ...
    def map(self, fn: ValueFn[B_co, C]) -> Operation[A_contra, C]: ...
    def flat_map(
        self,
        fn: Callable[[B_co], Operation[Unit, C]],
    ) -> Operation[A_contra, C]: ...
```

`Unit` is a singleton value named `UNIT`. It represents a deliberately nullary source without adding a second execution convention.

`.then` accepts an `Operation`, not an arbitrary callable. `.map` is the explicit callable-lifting shorthand. `.flat_map` invokes its callback only after success, runs the returned `Operation[Unit, C]` with `UNIT`, and reuses the same environment. A returned `Operation` from `.map` remains ordinary data.

The convenience operators are thin aliases:

- `a >> b` delegates to `a.then(b)`;
- `a | b` delegates to `or_else(a, b)`;
- `&` is not part of the new interface because its policy and tuple grouping are not self-describing.

Canonical constructors are:

```python
operation(fn)                     # callable returns a raw value
result_operation(fn)              # callable returns Result
environment_operation(key)(fn)    # raw value plus one capability
environment_result_operation(key)(fn)
threaded_operation(fn)            # explicit asyncio.to_thread adapter
identity()                        # Operation[T, T]
constant(value)                   # Operation[Any, T]
fail(error)                       # Operation[Any, Never]
```

The environment decorator factories bind only the capability type at the outer call. Their returned decorator object is generic in the decorated callable's input and output types, avoiding premature `Never` inference.

Canonical combinator names are:

- `pipe_all(operations)` for left-to-right composition;
- `recover(op, handler)` and `recover_with(op, handler)`;
- `default_on_error(op, value)`;
- `or_else(first, second)` and variadic `fallback(*operations)`;
- `tap(op, effect)` with propagated failure;
- `tap_best_effort(op, effect)` with deliberately ignored effect failure;
- `retry(op, *, attempts, backoff)` where `attempts` includes the first call;
- `fanout_parallel(*operations)` for fail-fast same-input concurrency;
- `fanout_all_settled(*operations)` for one `Result` per branch;
- `collect_named(**operations)` for fail-fast named same-input fanout.

Configuration is snapshotted into immutable tuples and mappings when a combinator is constructed. `Operation` is a value object with `name`, `description`, and optional `source_signature` metadata; it does not pretend to have the wrapped callable's invocation signature.

## Error and `Result` policy

The public error type is `Exception`. A third generic error parameter is rejected because Python callers primarily need a stable compositional error channel, while precise exception unions would spread overload complexity through every combinator.

The boundaries are:

1.  Invalid combinator configuration raises synchronously at construction.
2.  Once `.run` starts, planning, environment resolution, input adaptation, callable invocation, awaiting, and output adaptation are inside the `Result` boundary.
3.  Ordinary `Exception` becomes `Error(exception)` without rewriting the cause.
4.  `asyncio.CancelledError`, `KeyboardInterrupt`, `SystemExit`, and other `BaseException` subclasses propagate.
5.  A raw `operation` never dynamically flattens a returned `Result`.
6.  A `result_operation` flattens exactly the one `Result` its interface declares.
7.  Recovery handlers follow the same raw-versus-result distinction: `recover` returns a value; `recover_with` returns an operation.

Default sequential, traversal, build, and parallel helpers fail fast. A helper must not use `None` as an error marker.

All-settled helpers run all selected work and retain `Result` at each position or field. They do not claim error accumulation or require a semigroup.

Lossy helpers use policy-bearing names such as `traverse_lossy`, `filter_best_effort`, and `build_lossy`. They either drop failures or invoke an explicit `on_error` callback. No generic `errors="ignore"` flag is placed on the fail-fast helper because a flag obscures return-type and control-flow differences.

## Environment, not state

`Environment` is an immutable mapping from typed identity keys to capabilities:

```python
T = TypeVar("T")

class EnvKey(Generic[T]): ...

class Environment:
    def require(self, key: EnvKey[T]) -> T: ...
    def with_value(self, key: EnvKey[T], value: T) -> Environment: ...
```

`with_value` creates a new environment before execution; stages cannot install it into the running pipeline. Missing capabilities and incompatible runtime values produce `Error(EnvironmentError(...))` from `.run`.

The mapping is read-only, not transitively pure. A database pool, logger, browser driver, or browser session can expose side effects. Ownership and concurrency rules belong to that capability's interface.

Returning `BaseContext`, `Environment`, or any capability is ordinary success data. FP-Ops will have no output-sensitive state transition. If a future product needs a state transformer, it must be a separately named type with an explicit `(value, state)` outcome and separate laws.

Environment requirements are stored as immutable AST metadata and composition
unions the required keys. `Environment.with_value` validates a capability
against its key when the environment is assembled; composition does not register
or replace capability values. `Operation[A, B]` intentionally does not add an
environment type parameter: typed `EnvKey[T]` protects capability access, while
completeness is validated at `.run`.

## Async normalization

Every callable adapter uses one internal `invoke` implementation:

1.  call the function;
2.  while the returned object is awaitable, await it;
3.  return the first non-awaitable object;
4.  let the surrounding runner convert ordinary exceptions to `Error`.

Normalization depends on the returned object, not `inspect.iscoroutinefunction`. The same implementation is used by stages, callbacks, predicates, recovery handlers, builders, and traversal helpers.

Synchronous callables run inline by default. This preserves `contextvars`, thread affinity, and existing low-cost behavior. Blocking work must use `threaded_operation`, whose adapter uses `asyncio.to_thread`.

Fail-fast parallel nodes cancel unfinished siblings when the first observed error is selected, await their cancellation, and return that error. All-settled nodes await every branch. Output order is declaration order; completion order is unspecified. A concurrency limit is mandatory for collection-parallel helpers and optional for fixed fanout.

## Binding

`Operation.__call__`, `Operation.__await__`, runtime placeholder rendering, and
representation-dependent deferred arguments are not part of the new core.
`.run` always consumes exactly one value.

Multi-argument callables live at an adapter seam, outside composition:

- `Args[P]` can lift a genuinely multi-input source into one typed record;
- `UNIT` represents a nullary source;
- construction-only operation templates use one direct `_` to select the unary
  input while snapshotting every other argument;
- ordinary Python factory functions capture configuration and return a unary `Operation`;
- `functools.partial` remains available inside a factory when its honest Python typing is sufficient.

For example:

```python
@environment_result_operation_template(BROWSER_SESSION)
async def get_attribute(
    selector: str,
    element: Element,
    attribute: str,
    session: BrowserSession,
) -> Result[str, Exception]:
    ...

read_href = get_attribute("a", _, "href")
```

This makes `selector` and `attribute` construction-time configuration and
`element` pipeline data. Duplicate, missing, and excess configuration arguments
are rejected against the original Python signature.

This is not the old placeholder engine: a template call must contain exactly one
direct placeholder among at most six positional parameters and immediately
returns a unary `Operation`. Nested placeholders, multiple holes, execution-time
argument appending, and arbitrary `ParamSpec` subtraction are not supported.
The runtime construction check remains authoritative when `Any` weakens static
checking. Variadic configuration is supported when the hole is a fixed
positional parameter, with limited checker support for forwarding configured
variadic arguments. Keyword holes work at runtime but are intentionally outside
the mypy/Pyright overload surface.

`threaded_operation_template` shares the same construction typing and moves the
configured blocking callable to `asyncio.to_thread`. Environment templates
identify the injected capability as the final positional parameter, allowing
typed keyword-only configuration after it.

## Collection and object operations

The collection module separates pure/async callbacks from nested operations:

- `map_each(fn)` maps a value function and fails on its first exception;
- `traverse(op)` runs an operation sequentially and fails fast;
- `traverse_parallel(op, *, limit)` runs concurrently and fails fast;
- `traverse_all_settled(op, *, limit)` returns per-item `Result` values;
- `traverse_lossy(op, *, limit)` returns only successes;
- `filter_each(predicate)` fails if the predicate raises;
- `filter_operation(predicate)` is sequential and fail-fast;
- `filter_best_effort(predicate)` treats predicate errors as rejection;
- `fold(fn, initial)` and `fold_operation(op, initial)` distinguish effects;
- `fanout_each(*operations)` replaces the current non-positional `zip`.

List order and dictionary insertion order are preserved. Concurrent traversal returns results in input order. Empty inputs do not invoke the callback.

`CollectionOperation` carries the input/output shape through `.run` and
composition, so iterable inputs produce lists, mapping inputs retain keys, and
a list-producing traversal can feed a list-consuming operation template without
exposing a list-or-dict output union.

The object module provides:

- `get_path(path, default=MISSING)`, which distinguishes missing from `None`;
- `build(schema, model=None)`, which fails fast with `BuildFieldError(field, cause)`;
- `build_all_settled(schema)`, which returns field-level `Result` values;
- `build_lossy(schema)`, which omits failed fields rather than inserting ambiguous `None`;
- `assign`/`merge_shallow` and `assign_fields` for explicit shallow updates.

`build` evaluates fields in schema insertion order. A separate `build_parallel` may be added only with a concurrency limit and documented capability-safety requirements. Schema mappings are snapshotted at construction.

For operation- and callable-only schemas, `build` infers one shared input type
across fields. Typed constants use `constant(value)`; explicitly dynamic
`Mapping[str, Any]` schemas opt out of that inference.

## Internal modules and seams

The target package layout is:

```text
fp_ops/
  core.py             Operation, Unit, public constructors
  templates.py        construction-only typed placeholder adapters
  environment.py      EnvKey and immutable Environment
  result.py           public error wrappers and Result helpers
  composition.py      pipe, recovery, retry, fanout
  collections.py      traversal, filtering, folding
  objects.py          paths, build, assign
  _ast.py             immutable internal node definitions
  _invoke.py          callable normalization and adapters
  _runner.py          the sole AST interpreter
```

`core.Operation` is the external seam. Its interface contains input/output types, immutability, environment requirements, error behavior, async behavior, and ordering—not merely method signatures.

`_invoke.Invocation` is an internal seam justified by multiple adapters: raw-value, returned-`Result`, environment-aware, and thread-offloaded. Adapters share the normalizer and differ only in explicit output/environment adaptation.

`_runner` is one implementation, not a port with a hypothetical alternate adapter. It pattern-matches the closed internal AST. Tests normally cross the `Operation` interface; direct AST tests cover only invariants that cannot be observed economically through that interface.

`Environment` is a concrete immutable module. `MappingEnvironment`, `SilkEnvironment`, and similar wrapper names should not exist unless they actually provide different adapters at a real seam.

The current `graph.py`, edge/port types, renderer compilation, runtime
placeholder implementation, and executor plan are deleted in Phase 1 as soon as
the unary core can execute and compose. The construction-only template adapter
does not reuse that implementation. Keeping the legacy machinery until the
replacement surface is complete would reduce locality and fail the deletion
test. Git history is the archive; the package does not carry a second
implementation.

## mypy and Pylance typing design

Inline typing supports Python 3.10 syntax and uses contravariant input and covariant output type variables. `ParamSpec` appears only in callable/factory adapters, never as the pipeline's internal type.

Async overloads precede broad synchronous overloads so `Callable[[A], Awaitable[B]]` produces `Operation[A, B]`, not `Operation[A, Awaitable[B]]`. Result-returning constructors have their own overloads and never compete with raw-value constructors.

Environment decorator factories return a non-parameterized-in-output generic decorator protocol. The outer call specializes only `EnvKey[T]`; the protocol's `__call__` specializes `A` and `B` from the decorated function.

`.then` requires `Operation[B, C]`. Plain callables use `.map`, so mypy and Pylance do not need an overload that conflates an operation with a value function. `run` always requires its unary value, including `UNIT`.

Contract fixtures live outside package internals:

```text
tests/typecheck/mypy/good.py
tests/typecheck/mypy/bad.py
tests/typecheck/pyright/good.py
tests/typecheck/pyright/bad.py
```

Good fixtures use `assert_type`/`reveal_type`. Bad fixtures assert stable, specific diagnostics for incompatible stages, missing input, nested `Result`, and wrong capability values. CI checks the Python floor and newest supported version, builds a wheel, installs it into an empty environment, and runs both type checkers against that wheel.

Strictness is enabled first for the new public modules. The gate includes `disallow_any_generics`, `warn_unused_ignores`, and no public `Any` except explicit dynamic data helpers such as `get_path`.

## Silk migration

Silk should migrate as a downstream consumer, not as a special case inside the core.

`ActionContext` is split by role:

- stable driver/session/configuration capabilities live under typed environment keys;
- the current `Page`, `BrowserContext`, `ElementHandle`, or other changing focus is ordinary pipeline input/output;
- creating or selecting a page returns that page explicitly and never replaces hidden runner context.

Silk action names become normal factories. For example, `navigate(url)` returns `Operation[Page, Page]`, `query_all(selector)` returns `Operation[Page | ElementHandle, list[ElementHandle]]`, and `get_text(selector)` consumes a page or element scope. Configuration is captured by the factory signature.

The flagship pipeline becomes conceptually:

```python
pipeline = (
    navigate(url)
    .then(query_all("article.product_pod"))
    .then(traverse(extract_book_details))
)
result = await pipeline.run(page, environment=environment)
```

`build` is fail-fast, so a browser failure is reported as `BuildFieldError("title", browser_error)` with the original exception as cause. Silk opts into `build_lossy` only where partial records are a product feature.

The 42 context-decorator ignores disappear as actions move to typed environment constructors. Broad action-level `try/except Exception` blocks are removed when they only duplicate `result_operation`; third-party errors that need domain translation remain explicit.

Silk's capitalized re-export wrappers and its fp-ops compatibility layer are
deleted during this final phase. Actions and examples move directly to the new
interface. Silk pins the completed breaking FP-Ops release and adds the real
books pipeline to runtime and type-check CI.

## Breaking cutover and legacy removal

This is an aggressive replacement, not a compatibility program.

The new implementation replaces the existing top-level `fp_ops` exports in one
cutover. There is no intermediate `fp_ops.pipeline` namespace, `fp_ops.legacy`
package, compatibility adapter, dual runner, forwarding method, or warning
schedule. `.execute`, callable/awaitable `Operation`, runtime partial binding,
the legacy placeholder renderer, `bind`, `apply_cont`, old `sequence`, and
binary `&` are deleted in Phase 1 when the new core becomes the package default.
Construction-only templates may create unary operations but never change
execution semantics. The refactor branch may temporarily lack higher-level
helpers while they are rebuilt; it must never contain two `Operation`
implementations.

Old tests are not retained merely to keep obsolete behavior alive. Tests that
describe target-independent facts are rewritten through the new interface;
tests that assert excess-argument dropping, hidden context replacement,
errors-as-`None`, graph representation, or dynamic flattening are deleted.
Replace-don't-layer keeps the `Operation` interface as the test surface and
prevents two implementations from competing for locality.

`>>` may remain because it is implemented only as syntax for the new `.then`
semantics. `|` may remain only as syntax for the new `.or_else`. They do not
delegate to old code. Every other old-to-new correspondence belongs in a
breaking-change migration guide, not in runtime adapters:

| Removed interface | Replacement |
|----|----|
| `.execute(value)` | `.run(value)` |
| `await bound_operation` | `.run(UNIT)` on an explicitly nullary operation |
| `operation(args...)` binding | construction-only template or ordinary typed factory |
| `a & b` | `fanout_parallel(a, b)` |
| `.bind(fn)` | `.flat_map(fn)` with `Operation[Unit, C]` |
| `.apply_cont(fn)` | explicit `.map`, `.flat_map`, or recovery |
| `Operation.sequence` | `pipe_all` |
| `sequences.map(Operation)` | `traverse` |
| `sequences.filter(Operation)` | `filter_operation` or `filter_best_effort` |
| `sequences.zip` | `fanout_each` |
| lossy `build`/`merge` | explicit fail-fast, all-settled, or lossy helper |
| returned-`BaseContext` state update | explicit pipeline output |

## Dependency-ordered implementation phases

### Phase 0 — Freeze only the target contract

**Deliverables:** Public-interface contract tests, law probes, mypy and
Pyright/Pylance good/bad fixtures, and packaged-wheel import tests. Existing
failures remain documented in the audits; they are not converted into permanent
compatibility tests.

**Acceptance:** The target fixtures fail against the current implementation for
the expected reasons and describe only the replacement behavior. No new test asserts
graph shape, call binding, hidden context state, errors-as-`None`, or another
contract scheduled for deletion.

### Phase 1 — Replace legacy with the unary core

**Deliverables:** `Unit`, the new top-level `Operation[A, B]`, immutable
sequential AST, `.then`, `.run`, raw-value and returned-`Result` constructors,
one invocation normalizer and exception boundary, generic
`identity`/`constant`, configuration snapshots, and metadata. In the same phase,
delete the old `Operation`, graph, executor, runtime templates/placeholder
renderer, binding methods, compatibility-only exports, and tests whose contracts
have been rejected.

**Acceptance:** Identity, associativity, `None`, exception closure, and
sync/async tests pass through the public interface; incompatible `.then` calls
fail both type checkers. The worktree contains one `Operation` implementation
and no graph, edge, port, renderer, execution plan, runtime placeholder engine,
legacy namespace, or old runner.

### Phase 2 — Add environment and higher-order behavior

**Deliverables:** Typed `EnvKey`, immutable `Environment`, environment-aware constructors, `.map`, `.flat_map`, recovery, alternatives, tap, validated retry, and the thread-offload adapter.

**Acceptance:** A run cannot replace its environment; raw stages preserve `BaseContext`, `Result`, and `Operation` values; missing capabilities return errors; decorator fixtures infer concrete types without ignores.

### Phase 3 — Add parallel, collection, and object modules

**Deliverables:** Fail-fast and all-settled fanout, bounded sequential/concurrent traversal, explicit lossy variants, path/build/assignment helpers, and field/item error wrappers with preserved causes.

**Acceptance:** Default helpers never convert failure to `None` or omission;
fail-fast concurrency cancels and joins siblings; all-settled output order is
stable; a field's original exception remains inspectable through
`BuildFieldError`.

### Phase 4 — Complete FP-Ops and cut the breaking release

**Deliverables:** Complete top-level exports, generated interface reference,
executable examples, the breaking-change migration guide, packaged-wheel
validation, and deletion of any stale docs or tests that survived the earlier
replacement.

**Acceptance:** Source, examples, and the built wheel pass runtime, mypy, and
Pyright/Pylance gates. Documentation contains only installed exports. The
package has no `.execute`, call binding, hidden environment update, implicit
error loss, compatibility namespace, or private graph machinery. The breaking
release is published or otherwise pinned as an immutable build before Silk
changes begin.

### Phase 5 — Migrate Silk

This is deliberately the final phase. FP-Ops is complete and has no Silk-aware
compatibility code before downstream work begins.

**Deliverables:** A hard dependency pin to the breaking FP-Ops release, typed
browser capability keys, explicit page/context/element flow, action factories,
deletion of capitalized re-export wrappers and fp-ops compatibility helpers,
fail-fast build/traversal with deliberate lossy opt-ins, and rewritten examples.

**Acceptance:** No fp-ops decorator or pipeline ignore, old wrapper, or legacy
call form remains. The books example and Silk's full relevant suite pass at
runtime and under both type checkers; browser causes survive object
construction; parallel tests characterize shared-capability safety. No FP-Ops
change is required to make Silk pass.

## Law and contract test matrix

| Area | Required assertion | Scope or caveat |
|----|----|----|
| Identity | `identity().then(f)` and `f.then(identity())` match `f` | Observable result/effect trace |
| Associativity | regrouping `.then` matches | Same input and environment |
| Short circuit | later stages do not run after `Error` | Sequential nodes only |
| `None` | `Ok(None)` reaches the next stage | Never absence/error |
| Raw `Result` | remains successful data | Raw constructor only |
| Result constructor | flattens exactly once | No dynamic recursion |
| Returned operation | remains data under `.map` | `.flat_map` executes it |
| Environment | same object supplied to every sequential stage | No replacement semantics |
| Async | sync, async, and sync-returning-awaitable agree | Threading only when explicit |
| Cancellation | `CancelledError` propagates | Never converted to `Error` |
| Fanout | declaration-order output | Completion order unspecified |
| Fail-fast fanout | outstanding siblings cancelled and joined | First observed error |
| All-settled | every branch represented by `Result` | No Validation claim |
| Traversal | input order preserved | Concurrent completion may differ |
| Build | field and original cause retained | Fail-fast insertion order |
| Binding | templates/factories reject invalid configuration normally | No graph-size dependency |
| Immutability | caller config mutation cannot change an operation | Captured domain values documented |
| Typing | good fixtures reveal exact types; bad fixtures fail | mypy and Pyright/Pylance |

Property tests use pure total functions for identity and associativity. Example tests cover I/O ordering and errors. Tests do not assert laws for retry timing, parallel effect order, lossy helpers, or mutable capability implementations.

## Risks and mitigations

- **Migration size:** Silk relies on argument binding and implicit context.
  Replace one-hole configuration with construction-only templates and rewrite
  hidden context state after the FP-Ops cutover; do not reduce short-term diff
  size by carrying two execution interfaces.
- **Environment type erasure:** `Operation[A, B]` cannot statically prove all keys are installed. Mitigate with typed key access, AST requirements, and `.run` preflight validation.
- **Parallel cancellation races:** an external effect may complete after another branch fails. Document best-effort cancellation and test cooperative tasks.
- **Mutable capabilities:** a read-only mapping can still contain unsafe shared objects. Require capability-level concurrency tests; do not claim isolation.
- **Typing divergence:** mypy and Pyright resolve overloads differently. Keep constructors distinct and gate both against installed wheels.
- **Hard-cutover blast radius:** removing the old interface exposes every stale
  caller immediately. Complete FP-Ops contract, wheel, example, and type gates
  before starting Silk; do not mitigate this by restoring legacy code.
- **Error-wrapper noise:** field/item wrappers can obscure causes. Use exception chaining and expose `.cause`, path, and index directly.
- **Surface growth:** policy variants can become shallow duplication. Add a public variant only for a demonstrated use; keep shared implementation local.

## Rejected alternatives

- **Repair the current DAG executor:** no current consumer requires general multi-source/multi-tail dataflow, so the added interface would not earn depth.
- **Keep the graph as an internal optimization:** the linear AST is smaller and more truthful; unused graph semantics harm locality.
- **Patch `Operation[P, R]` overloads:** `ParamSpec` cannot describe both source arguments and unary stage composition.
- **Use `Operation[A, B, E]`:** typed error unions add pervasive complexity without improving Silk's exception-handling decisions.
- **Make all-settled or Validation the default:** existing pipelines principally need fail-fast behavior; accumulation requires a different return shape.
- **Dynamically flatten `Result` and `Operation`:** runtime guessing makes representation change semantics and defeats static typing.
- **Keep context as implicit State:** output-sensitive replacement breaks identity and prevents context values from being ordinary data.
- **Clone context for parallel branches:** shallow copies do not isolate browser handles and create a false safety guarantee.
- **Automatically thread all sync callables:** it changes thread affinity and adds overhead; offload must be explicit.
- **Keep callable/awaitable `Operation`:** construction, binding, and execution need separate interfaces.
- **Expose a runner port:** there is one interpreter adapter; a second interface would be hypothetical indirection.
- **Ship a compatibility namespace or deprecation layer:** it would preserve
  precisely the graph, binding, context, and error contracts the redesign exists
  to remove, while doubling the test surface.
- **Adopt dense Haskell naming:** Python-first names make data shape and failure policy visible without claiming unsupported algebraic instances.

## Genuinely unresolved product decisions

Only two product choices remain; neither changes the core semantic model:

1.  For each Silk selector action, decide whether “not found” is successful `None` or a domain `NotFound` error. Fallback activates only for the latter; an explicit `require_some` adapter can convert between them.
2.  Identify which current Silk workflows intentionally need partial records or
    dropped collection items. Those call sites determine whether
    `build_lossy`/`traverse_lossy` belong in the public interface or should be
    implemented locally in Silk.

The release number and cutover date are scheduling decisions, not architecture
decisions. There is intentionally no warning-window decision.
