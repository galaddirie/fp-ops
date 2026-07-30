# FP-Ops architecture, typing, and error-semantics audit

- Audit date: 2026-07-30
- Repository state: `07358e2` (`v0.2.11`)
- Primary consumer reviewed: `../silk`

## Executive summary

FP-Ops currently has four competing identities:

1. a statically typed function-composition library;
2. an asynchronous effect runner that converts exceptions to `Result`;
3. a lazy DAG/execution engine;
4. an ergonomic partial-application and placeholder DSL.

Those identities impose different contracts, and the current implementation does
not choose between them. The public interface says "typed DAG," while the runtime
is a linear, single-value interpreter with implicit context state and
representation-dependent argument binding. The result is not merely imperfect
typing. Seemingly neutral refactors—adding `identity`, decorating a callable,
or grouping `&` differently—can change runtime behavior. Returning `Result`
instead of raising usually produces the same runtime result through dynamic
flattening, but changes the advertised static shape.

The most consequential conclusions are:

- The graph abstraction is not implemented as a graph at execution time.
- Calling, binding, and executing an operation have incompatible semantics.
- `Operation[P, R]` cannot express the library's actual input, error, context, or
  result-flattening behavior.
- Errors cross at least a dozen different boundaries, with policies ranging from
  fail-fast to silent `None`, item removal, fallback, retry, and uncaught escape.
- Context is an implicit second data channel selected and updated by runtime type
  checks.
- CI reports mypy success because it checks only package internals, where
  extensive `Any`, casts, and erased relationships remain; consumer-facing
  examples and integrations are outside the gate.
- Silk confirms the problems in real use: 42 necessary decorator suppressions,
  a failing flagship pipeline type check, loss of browser root causes inside
  `build`, and ambiguous `Ok(None)` versus `Error` fallback semantics.

This is a design problem before it is an annotation problem. Patching overloads
without first choosing one execution and error model will move the failures
around.

## Scope and method

This audit covered:

- all modules under `src/fp_ops`;
- the complete test suite and type-hint examples;
- README, changelog, packaging metadata, and CI workflows;
- concrete fp-ops use in `../silk`;
- focused runtime probes for binding, returned `Result`, context creation,
  placeholders, higher-order operations, async recovery, and swallowed errors;
- mypy behavior at both the package and consumer interfaces.

Verification results:

| Check | Result |
|---|---|
| `pytest -q` | 508 passed, 106 warnings |
| Coverage run | 508 passed, 93% line coverage, 102 warnings |
| `mypy -p fp_ops` | success across 14 source files |
| `mypy examples` | 35 errors across all 3 example files |
| `pytest examples` | collection error from nonexistent imports |
| Silk focused unit tests | 133 passed |
| Silk source plus real example under mypy | 2 errors in the real pipeline; decorator suppressions remain necessary |

High coverage is a strength, but it is also misleading here: several tests
explicitly lock in the surprising semantics identified below.

## Risk summary

| ID | Severity | Finding |
|---|---|---|
| C1 | Critical | The advertised DAG is executed as one linear running value |
| C2 | Critical | Binding/calling/execution are representation-dependent and violate identity |
| C3 | Critical | The type model does not describe pipeline composition or partial application |
| H1 | High | Static and runtime `Result` shapes disagree |
| H2 | High | `None` is both valid data and "no value" |
| H3 | High | Error policy changes across combinators and callable representation |
| H4 | High | Exceptions escape outside the advertised `Result` boundary |
| H5 | High | Context is an implicit, weakly typed state channel |
| H6 | High | Higher-order composition has three incompatible meanings |
| H7 | High | Sync/async normalization leaks coroutines and can block the event loop |
| M1 | Medium | Documentation and examples teach interfaces that do not exist or do not work |
| M2 | Medium | Quality gates validate implementation internals, not the consumer contract |
| M3 | Medium | Frozen specifications retain mutable captured values |
| M4 | Medium | Duplicate composition interfaces have subtly different shapes and vocabulary |
| M5 | Medium | Defensive runtime behavior contradicts declared types |
| M6 | Medium | Operations do not expose normal callable metadata and signatures |
| M7 | Medium | `BaseContext.merge` permits incompatible types, then fails during mutation |
| M8 | Medium | Retry/wait/parallel semantics are underspecified |

## Detailed findings

### C1 — The DAG interface is not implemented by the executor

`OpGraph` describes a directed acyclic multigraph with result, error, and context
paths (`src/fp_ops/graph.py:9-13`). Edges contain typed ports, optional/default
metadata, edge types, and transforms (`src/fp_ops/primitives.py:195-217`).
`ExecutionPlan` records successor relationships
(`src/fp_ops/execution.py:23-29,99-103`).

The executor does not use those semantics:

- it maintains one `last_result` (`execution.py:214-218`);
- every node after the first receives that one value (`execution.py:302-304`);
- `id2value` is populated but never read (`execution.py:214,368`);
- successors, edge source/target ports, edge types, defaults, and transforms do
  not affect evaluation.

A focused branch probe with `source -> plus_one` and
`source -> times_ten` returned `Ok(40)` for input `3`: the sibling operations
were run as a sequence. Sibling insertion order therefore changes meaning.

This graph module fails the deletion test for module depth: deleting it and
representing the real behavior as a linear pipe removes complexity rather than
forcing that complexity into callers.

Recommendation: choose one of two honest designs.

- Recommended now: replace the graph with a small immutable linear pipe AST.
  Silk and the current public operators principally need pipelines, not general
  dataflow.
- If a real DAG is a product requirement: evaluate nodes from their incoming
  edge source values, define ready-node scheduling and multi-tail outputs, and
  implement port, transform, error, and context edges as actual semantics.

Do not retain graph vocabulary as an aspirational internal abstraction. It makes
validation, visualization, and future optimization less trustworthy.

### C2 — Binding, calling, and execution are not substitutive

The meaning of `__call__` depends on whether the operation has one node:

- a single-step call appends arguments into the node template
  (`src/fp_ops/operator.py:203-243`);
- a pipeline call stores a separate deferred argument set
  (`operator.py:245-254`);
- a pre-bound head in a multi-step pipeline discards all call-time arguments
  (`src/fp_ops/execution.py:281-288`);
- extra runtime positionals are silently ignored
  (`execution.py:171-172`).

The test suite and changelog make much of this intentional:

- `tests/test_execution.py:135-145` requires excess arguments to disappear;
- `tests/test_binding.py:368-392` requires a pre-bound pipeline to ignore later
  overrides;
- `CHANGELOG.md:18-22` advertises silent dropping instead of `TypeError`.

Observed behavior:

| Expression | Result |
|---|---|
| `f.execute(1, 2, 3)` | `Ok(1)` |
| `f(1, 2, 3).execute()` | `Error(TypeError(...))` |
| `add(1, 2).execute(a=10, b=6)` | `Ok(16)` |
| `(add(1, 2) >> identity).execute(a=10, b=6)` | `Ok(3)` |

Adding `identity` changes argument precedence. Right identity is therefore not
observationally valid, despite the package's functional framing. Callers must
know graph size, template state, and the path by which arguments were bound.

The class docstring also says an unbound operation call executes immediately
(`operator.py:88-100`), while `__call__` always constructs another `Operation`.
Only subsequently awaiting that object executes it.

Recommendation:

- make `.run(...)` (or `.execute(...)`) the one execution interface;
- make `.bind_args(...)`/`.partial(...)` the explicit construction interface;
- use one `inspect.Signature.bind`-based argument algorithm;
- reject excess and duplicate arguments consistently;
- do not make an `Operation` itself awaitable unless that behavior adds a
  contract that `.run` cannot express.

### C3 — The type model does not describe the composition model

`Operation[P, R]` uses `ParamSpec P` for the externally invoked source
parameters and `R` for a successful value. That works for a wrapped function in
isolation, but it cannot simultaneously model:

- the unary value passed between pipeline stages;
- a multi-argument source function;
- placeholder or partial application;
- a hidden context argument;
- an error type;
- a stage that returns `Result`;
- a stage that returns another `Operation`.

The implementation works around that mismatch by erasing exactly the
relationships the type checker should enforce:

- `__rshift__` accepts `Operation[..., S]` and explicitly ignores the downstream
  input type (`src/fp_ops/operator.py:275-303`);
- runtime `__rshift__` accepts a plain callable, but its annotation accepts only
  `Operation`, so valid `op >> str`/lambda composition is rejected by mypy;
- `__call__` returns the original `Operation[P, R]` after any amount of binding
  (`operator.py:193-205`);
- the placeholder overload includes `Any`, collapsing the accepted argument
  type to `Any` (`operator.py:198-201`);
- `execute(self)` is a valid overload for every operation, so missing required
  parameters cannot be rejected (`operator.py:403-409`);
- `identity` is `Any -> Any`, breaking generic inference
  (`operator.py:881-891`);
- `pipe` and `compose` return `Operation[..., Any]`
  (`src/fp_ops/composition.py:30-77`);
- free return-only `P` variables cause `parallel(...)` and callable-condition
  `branch(...)` to infer `Operation[Never, ...]`.

A focused mypy probe revealed:

```text
add                     Operation[[left: int, right: int], int]
add(1)                  Operation[[left: int, right: int], int]
add(1, 2)               Operation[[left: int, right: int], int]
add >> stringify        Operation[[left: int, right: int], str]
contextual operation    Operation[Never, Never]
```

The last result is the central VS Code/mypy issue. Common
`@operation(context=True, context_type=...)` use fails inference from the
decorator-factory overloads (`operator.py:783-802`). Four of the five decorators
in `examples/test_typehints_simple.py` fail without ignores. Silk consequently
has 42 necessary `# type: ignore[arg-type]` annotations across its context-aware
actions.

The decorator factory binds `P` and `R` at the outer `operation(...)` call,
before mypy sees the decorated function. Its return should instead be a
non-parameterized decorator protocol whose `__call__` is itself generic. That
defers specialization until the actual function is supplied.

Other focused probes found:

- `parallel(op1, op2)` inferred `Operation[Never, tuple[Any, ...]]`;
- a simple callable-condition `branch` inferred `Operation[Never, int]`;
- `sync_inc("wrong")` was accepted and retained
  `Operation[[x: int], int]`;
- `sync_inc.execute()` was accepted despite the missing required argument;
- a generic identity could select the wrong async overload, while the exported
  identity normally erases output to `Any`.

Recommendation: use a unary compositional core, conceptually:

```text
Operation[Input, Output, Error]
Operation[A, B, E1].then(Operation[B, C, E2])
    -> Operation[A, C, E1 | E2]
```

Keep `ParamSpec` at a lifting/source adapter, not as the internal pipe type.
Multi-argument sources can accept a typed input dataclass/tuple or use a
separate `lift_callable[P, R]` adapter. Arbitrary partial application is not
fully expressible in Python's current type system; an explicit restricted
binding interface is better than claiming the original `P` still applies.

Test the public type surface with both mypy and Pyright/Pylance reveal/assert
fixtures. Package-internal mypy success is insufficient.

### H1 — Static and runtime `Result` shapes disagree

Execution declares `Result[R, Exception]` (`operator.py:403-409`). If a wrapped
function is annotated as returning `Result[T, E]`, the decorator therefore
produces a static `Operation[P, Result[T, E]]`, and execution appears to return:

```text
Result[Result[T, E], Exception]
```

At runtime the executor automatically flattens any returned `Result`
(`src/fp_ops/execution.py:345-355`). Silk actions use this pattern extensively.
A value statically described as a nested `Result` is actually `Ok(T)`.

The inverse mismatch is also possible: the executor accepts `Error("text")`,
even though the public error type is `Exception`. A runtime probe returned an
error payload of type `str`.

Internally, combinators repeatedly convert the result channel back into
exception control flow:

1. call nested `.execute()`;
2. inspect `is_error()`;
3. `raise result.error`;
4. let the outer executor catch the exception and recreate `Error`.

Examples include `retry`, `&`, `|`, `bind`, and `apply_cont`
(`operator.py:542-765`), plus collection and flow helpers.

Recommendation: make one policy statically explicit.

- Raw-operation constructor: the function returns `T`; raised `Exception` is
  captured as `Result[T, Exception]`.
- Result-operation constructor: the function returns `Result[T, E]`; the
  constructor is typed to create `Operation[..., T, E]`.

These can be separate decorators or carefully ordered overloads, but the public
type and runtime shape must match. Do not accept arbitrary error payloads if the
contract promises exceptions.

### H2 — `None` is conflated with absence

The unary and variadic argument renderers omit the downstream argument when its
value is `None` (`src/fp_ops/execution.py:68-79`). `None` is nevertheless a
legitimate value and is intentionally returned by conversions such as
`to_int`/`to_float` (`src/fp_ops/utils.py:280-308`).

Consequences:

- `(constant(None) >> identity).execute()` calls `identity` with no argument;
- optional-returning operations are not closed under composition;
- `None` can mean valid data, conversion failure, swallowed error, missing path,
  absent pipeline value, or failed collection item.

Recommendation: use a private `MISSING` sentinel for the interpreter's absence
state and always forward a real `None`.

### H3 — Error policy changes across combinators and representations

The current error algebra is:

| Location | Current policy |
|---|---|
| Ordinary stage raises | convert to `Error(exception)` |
| Stage returns `Result` | flatten it dynamically |
| `tap` callback raises | silently ignore |
| `filter(Operation)` fails | silently remove item |
| `filter(callable)` raises | fail outer operation |
| `map(Operation)` fails | fail fast |
| `map(callable)` returns awaitable | retain coroutine as value |
| `zip` item operation/callable fails | insert `None` |
| `build` nested operation/callable fails | insert `None` |
| `merge` operation fails | skip source |
| `merge` callable raises | fail outer operation |
| `catch` with sync handler | recover |
| `catch` with async handler | return unawaited coroutine as success |
| `fallback` sees `Error` | try next operation |
| `fallback` sees `Ok(None)` | stop successfully |

Evidence:

- `src/fp_ops/operator.py:475-526,579-620`;
- `src/fp_ops/sequences.py:46-74,112-136,233-255`;
- `src/fp_ops/objects.py:79-97,195-211,290-309`.

This violates representation independence: wrapping the same predicate in
`@operation` can change whether an exception fails the outer computation or
silently filters an item. Valid `None` and lost error become indistinguishable.
Retry and recovery cannot act on an error after a container helper has erased it.

Silk demonstrates the impact. Its flagship model-building example passes browser
operations into `build`. A nested `RuntimeError("root cause")` becomes
`Ok({"field": None})`; when a Pydantic model is requested, the caller sees a
generic validation failure and the browser root cause is no longer the primary
error.

Recommendation:

- normalize every accepted callable through one `as_operation` seam;
- make fail-fast the default traversal/build policy;
- expose explicit named alternatives: `filter_map`, `collect_results`,
  `validate_all`, `recover`, `ignore_tap_error`, and similar;
- never encode an unselected error policy as implicit `None`.

### H4 — Exceptions escape the advertised `Result` boundary

Only the actual stage function call is inside the executor's `try/except`
(`src/fp_ops/execution.py:345-349`). These can escape as exceptions:

- context factory creation/awaiting (`operator.py:442-444`);
- graph compilation and renderer creation (`operator.py:448`);
- renderer and argument merge failures (`execution.py:218-304`);
- insufficient placeholder values, observed as an escaping `RuntimeError`.

The `execute` docstring says context creation and pipeline execution exceptions
will be wrapped (`operator.py:409-434`), but a failing context factory raises
directly.

This split makes callers write both:

```text
try:
    result = await op.execute(...)
except Exception:
    ...

if result.is_error():
    ...
```

That defeats the purpose of a `Result` interface.

Recommendation: define the boundary precisely. If `run` promises `Result`, put
planning, context construction, rendering, validation, and stage invocation
inside it. Continue to let cancellation and other `BaseException` subclasses
propagate.

### H5 — Context is an implicit, weakly typed state channel

Context behavior is distributed across graph metadata, a reserved keyword, and
runtime inspection of successful values:

- composition picks the first non-null context factory/type with `or`, silently
  discarding later metadata (`src/fp_ops/operator.py:298-307`);
- explicit context remains in kwargs and can be sent to a plain head that does
  not accept it (`operator.py:440-453`);
- a successful value that happens to be a `BaseContext` updates hidden state and
  preserves the prior data stream (`execution.py:357-365`);
- `parallel` passes the same mutable context object to every branch
  (`src/fp_ops/composition.py:89-96`);
- `map` and `build` inspect private `_graph._nodes` to rediscover requirements
  (`sequences.py:99-110`, `objects.py:176-190`);
- `BaseContext` declares an inner class named `ConfigDict`, but this is not
  Pydantic v2 configuration (`src/fp_ops/context.py:4-9`). At runtime,
  `BaseContext.model_config == {}`.

A context-aware stage cannot return `BaseContext` as ordinary user data: when
`spec.require_ctx` is true, that output type activates the hidden state
transition. Stage ordering is also observable: if a later stage requires
context, supplying it to a pipeline whose head is plain can make that head fail
with an unexpected `context` keyword before the context-aware stage runs.
Parallel mutation can race, and Silk incorrectly documents that `&` provides
isolated browser contexts even though both branches receive the same object.

Recommendation:

- pass environment/context to the runner as its own channel, never as ordinary
  user kwargs;
- represent context updates explicitly rather than by inspecting output type;
- validate context compatibility at composition;
- choose a documented single-context, product-context, or capability model;
- define whether parallel branches share context or receive branch-local
  context from a factory/clone adapter. Freezing the Pydantic container alone
  does not isolate mutable browser/page handles stored inside it;
- expose context requirements through a stable interface instead of private
  graph inspection.

### H6 — Higher-order composition has incompatible meanings

There are three competing operations:

- `>>` wraps a callable and treats its returned value literally;
- `bind` invokes an operation-producing callback and executes the result;
- `apply_cont` accepts raw, awaitable, `Result`, or `Operation` output and
  dynamically flattens it (`src/fp_ops/operator.py:671-765`).

The README claims these are equivalent:

```python
double >> (lambda x: add(x, 10))
double >> add(_, 10)
```

They are not (`README.md:78-80`). The first pipeline returns an `Operation`
object as successful data; the second produces the numeric result.

`flow.when` has the opposite construction mistake: the configuration function is
itself decorated (`src/fp_ops/flow.py:115-128`). The documented
`when(condition, transform)(value)` path appends a third argument and returns
`Error(TypeError(...))`.

Nested combinators such as retry, fallback, catch, and parallel also create
opaque executors inside function closures. The graph cannot validate, visualize,
or schedule their real structure.

Recommendation: define distinct, lawful operations:

- `map: (A -> B) -> Operation[A, B]`;
- `flat_map: (A -> Operation[B]) -> Operation[A, B]`;
- `recover: (E -> B or Result[B, E2]) -> ...`.

Make choice, retry, and parallelism explicit AST nodes if inspection and
planning are product requirements.

### H7 — Sync/async normalization is incomplete

`_ensure_async` runs a synchronous callable directly on the event loop
(`src/fp_ops/operator.py:60-75`).

Two failures follow:

- blocking synchronous work blocks the event loop, including inside
  `parallel`;
- a synchronous function that returns an awaitable produces
  `Ok(coroutine)`—the awaitable is not awaited.

The main test run emitted an unawaited-coroutine warning from precisely this
case (`tests/test_basic_operations.py:205`). Plain async callbacks passed into
`filter`, `map`, or `zip` are likewise not awaited
(`src/fp_ops/sequences.py:74,124-136,247-251`).

The annotations disagree with the runtime here too:

- `attempt` accepts only `Callable[P, R]`, so an async function is inferred as
  `Operation[P, Coroutine[..., R]]` even though runtime execution awaits it
  (`src/fp_ops/flow.py:50-55`);
- `loop_until` types only a synchronous predicate while its implementation
  explicitly awaits awaitable predicate results (`flow.py:184-208`).

Recommendation:

- normalize callables once and await any awaitable result;
- add distinct sync/async overloads to all lifting and higher-order helpers;
- either reject blocking sync callables in async operations, document that they
  run inline, or expose an explicit thread-offload adapter;
- name sequential and concurrent traversals separately and specify concurrency
  limits/cancellation.

### M1 — Documentation and examples do not describe the installed interface

Current failures include:

- README imports nonexistent `sequence`, `map`, and `transform` from
  `fp_ops.composition` (`README.md:132`);
- README imports a nonexistent `fp_ops.data` package
  (`README.md:188-194`);
- README lists nonexistent `Operation.transform` and `.filter`
  (`README.md:284-292`);
- README's getting-started call binds an operation but never awaits or executes
  it (`README.md:47-50`);
- README passes a plain lambda where `loop_until` requires an `Operation`
  (`README.md:175-180`);
- `examples/test_typehints_complex.py:6-9` imports ten nonexistent composition
  names;
- `pytest examples` cannot collect because `sequence` does not exist;
- `mypy examples` reports 35 errors.

The examples are named `test_*.py` but `pytest.ini` restricts discovery to
`tests`, and CI checks only `mypy -p fp_ops`
(`.github/workflows/mypy.yml:23-24`). The most consumer-relevant artifacts are
therefore excluded from both gates.

Recommendation: first make the documentation truthful, then enforce it:

- runnable README snippets or doctests;
- `pytest examples`;
- `mypy examples tests/typecheck`;
- Pyright/Pylance fixtures for published-editor behavior;
- import/export smoke tests built against the wheel, not only the source tree.

### M2 — Existing quality gates do not protect the consumer contract

The test suite is broad and has high line coverage, but it largely tests the
current implementation's paths rather than functional laws and public
substitutability.

Examples:

- a type-incompatible `int -> str-method` pipeline is permitted to construct and
  fails only at runtime (`tests/test_advanced_operations.py:288-300`);
- `Operation.validate()` only checks non-emptiness and compiles renderers, so it
  would not detect that stage mismatch (`src/fp_ops/operator.py:325-343`);
- excess arguments disappearing is explicitly tested
  (`tests/test_execution.py:135-145`);
- pre-bound pipeline overrides being ignored are explicitly tested
  (`tests/test_binding.py:368-392`);
- build/zip error-to-`None` behavior is explicitly tested;
- there are no consumer type-contract tests in CI;
- Silk has no `>>` pipeline runtime coverage beyond its failing example.

The test run also reports:

- deprecated pytest-asyncio loop behavior and custom loop fixtures;
- unawaited coroutines;
- Pydantic v1-style deprecations;
- invalid escape warnings in text docstrings;
- deprecated positional `re.sub` arguments;
- collection warnings for context model classes whose names begin with `Test`.

The configured mypy check is also substantially weaker than the published
surface suggests. An independent `--strict` run found 75 errors across nine
files, dominated by raw `Operation`, `Callable`, regex, collection, graph, and
execution generics. The config does not enable `disallow_any_generics`,
`warn_unused_ignores`, or strict mode (`pyproject.toml:47-56`). It checks as
Python 3.11 while the package claims 3.10–3.13 and CI itself runs on 3.13. No
Pyright/Pylance project configuration exists.

Recommendation: add contract tests before refactoring:

- right and left identity where claimed;
- associativity where claimed;
- representation independence for callable versus wrapped operation;
- `None` preservation;
- one error-boundary matrix;
- context compatibility and parallel isolation policy;
- compile-time type assertions for good and bad pipelines;
- Silk's flagship pipeline as a downstream contract fixture.

Increase strictness incrementally, starting with public modules, and type-check
against the supported Python floor plus the CI version.

### M3 — Immutability is shallow

`Template` and `OpSpec` are frozen dataclasses, which is useful, but they retain
arbitrary mutable sequences, mappings, and values
(`src/fp_ops/primitives.py:61-171`). `constant` returns the same captured object
on every execution (`src/fp_ops/operator.py:862-879`). `build` and `update`
capture caller-owned dictionaries (`src/fp_ops/objects.py:127-236,316-331`).

Mutation after construction can change later results for the same input, and a
caller mutating a returned constant affects future callers.

Recommendation: snapshot configuration at construction with immutable
containers. Decide and document whether mutable constants are copied on return.

### M4 — Duplicate composition interfaces dilute the module seam

The same concepts appear through multiple interfaces:

- `op1 & op2` and `parallel(op1, op2, ...)`;
- `op1 | op2` and `fallback(...)`;
- `Operation.retry(attempts=...)` and `flow.retry(max_retries=...)`;
- `Operation.sequence(...)`, `pipe(...)`, and `compose(...)`;
- `bind(...)` and `apply_cont(...)`;
- `__call__`, `partial(...)`, `execute(...)`, and `__await__`.

They are not mere aliases. Binary `&` creates nested tuples according to
grouping, while variadic `parallel` returns a flat tuple.
`(a & b) & c != a & (b & c)` by output shape.

Silk adds a shallow capitalized re-export layer on top, increasing the number of
names without adding a stable seam.

Recommendation: choose one canonical interface per concept. Keep compatibility
aliases thin, documented, and behaviorally identical during migration.

### M5 — Defensive runtime behavior contradicts declared types

Several helpers are typed narrowly but silently accept and return values outside
their annotations:

- text functions typed `str -> str` return the original non-string input;
- `keys`/`values`/`items` typed for dictionaries return empty lists for other
  inputs;
- `length` claims `Sized` but retains a no-length fallback;
- conversions use `Any` and erase the reason for failure.

This style makes runtime calls appear forgiving while undermining mypy's
narrowing and caller trust. The choice should be explicit:

- strict typed operation that rejects invalid input into `Error`; or
- widened input and output types that honestly describe totalized behavior.

### M6 — Operations do not expose wrapped callable introspection

The decorator copies only the wrapped function's docstring
(`src/fp_ops/operator.py:843-851`). The original signature is retained
internally on `OpSpec.signature` (`operator.py:175-181`), but an `Operation`
exposes no `__name__`, `__wrapped__`, or original `inspect.signature`;
inspection reports the generic
`Operation.__call__(*args: Any, **kwargs: Any)` interface. `_ensure_async` also
does not apply `functools.wraps`, so sync stages lose the original function name
inside the execution spec.

This affects help, tracebacks, debugger displays, framework inspection, and
editor parameter hints even when static generic inference succeeds.

Recommendation: expose stable metadata and the original source signature, or
make it explicit that an Operation is a value object rather than pretending it
is a normal decorated callable.

### M7 — `BaseContext.merge` is typed too broadly

`BaseContext.merge` accepts any `BaseContext` and annotates only `BaseContext`
(`src/fp_ops/context.py:11-46`). It deep-copies `self`, then attempts to assign
every field from `other`. Merging unrelated subclasses passes the initial check
and then raises a Pydantic `ValueError` while assigning an unknown field.

`examples/test_typehints_complex.py:150-154` attempts exactly such a merge
between user and database contexts.

Recommendation: return `Self` and accept `Self`, or define an explicit product
context/composition function. Do not imply arbitrary context subclasses are
merge-compatible.

### M8 — Retry, wait, and parallel behavior needs a sharper contract

Examples:

- `flow.retry(max_retries=3)` passes `3` as total `attempts`, not three retries
  after an initial attempt (`src/fp_ops/flow.py:91-103`);
- `retry(attempts=0)` eventually relies on an assertion rather than validating
  configuration (`operator.py:542-560`);
- `wait(timeout=...)` does not bound the duration of a single operation call
  (`flow.py:242-251`);
- `parallel` waits for all operations to return `Result` before inspecting any
  error, and the same context object is shared;
- sync operations in `parallel` are not concurrent.

These are reasonable choices only if named and documented precisely.

## Functional-law audit

| Principle/law | Status | Evidence |
|---|---|---|
| Deferred construction | Mostly sound | Stages run only on await/execute |
| Input immutability | Mostly sound | Helpers generally allocate fresh outputs |
| Right identity | Broken | Adding `identity` changes argument binding |
| Composition associativity | Plausible for linear execution but unverified | No law test; literal `==` is unsupported because `Operation` has no equality |
| Parallel associativity | Broken by value shape | Binary `&` nests tuples by grouping |
| Referential transparency | Conditional | Captured mutable templates/constants/config can change results |
| Error closure | Broken | Some errors become `None`, some escape, some remain `Error` |
| Optional-value closure | Broken | `None` disappears between stages |
| Representation independence | Broken | callable versus `Operation` changes async/error behavior |
| Typed composition | Broken | downstream input is erased by `Operation[..., S]` |
| Explicit effects | Mixed | `tap` is explicit; context mutation and error erasure are hidden |

The presence of local mutation inside a function is not itself a functional
violation. The relevant problems are observable mutation, hidden effects, and
failure of substitution/equational reasoning.

## Silk case study and defect ownership

Silk is valuable because it distinguishes fp-ops defects from consumer mistakes.

| Observation | Owner | Assessment |
|---|---|---|
| 42 context decorator `type: ignore[arg-type]` suppressions | fp-ops | Decorator factory infers `Never` |
| Real `books_to_scrape` pipeline fails mypy | Shared | fp-ops map/pipe inference fails; one Silk external-input annotation is also conceptually wrong |
| `build` loses nested browser root cause | fp-ops | Error is converted to `None` |
| `GetText(primary) \| GetText(fallback)` does not fall back | Silk | `GetText` returns `Ok(None)`; fp-ops fallback correctly responds only to `Error` |
| Claimed isolated `&` browser contexts | Silk docs plus fp-ops limitation | fp-ops shares the same context object |
| `fp-ops >=0.1.3,<1.0.0` | Silk | v0.1.3 lacks required exports/modules; actual lower bound is approximately v0.2.11 |
| 40 broad `except Exception` blocks in Silk actions | Shared friction | Mostly redundant because fp-ops already captures stage exceptions |
| `cast(ElementHandle, None)` and similar casts | Shared/tooling | Expression's `default_value` narrowing is awkward; an unwrap/match helper should avoid lying casts |

Silk's direct actions are well covered, but its central `>>` usage is not. A
downstream contract suite must exercise composition, not only isolated actions.

## What is sound and worth preserving

The redesign should retain these strengths:

- the built distribution includes `fp_ops/py.typed`; PEP 561 packaging is
  correctly configured;
- bare `@operation` preserves concrete sync and async signatures in focused
  mypy probes;
- `constant` preserves its output type and `objects.build` provides useful
  model-return overloads;
- construction is genuinely deferred;
- operation composition usually creates new graphs/specs instead of mutating
  the source operation;
- `Template` and `OpSpec` provide a useful immutable structural starting point;
- most collection/object helpers do not mutate caller inputs;
- `BaseContext.merge` deep-copies before updating same-shaped contexts;
- cancellation is not swallowed by the ordinary `Exception` handler;
- `tap`, retry, wait, fallback, and parallelism are legitimate explicit effect
  combinators—their existence is not a purity defect;
- cached compilation and UUID node IDs are not observable result
  nondeterminism under the current private-graph assumptions.

## Recommended target identity

The recommended product identity is:

> A typed, immutable, asynchronous linear pipeline with an explicit result
> channel and explicit effect combinators.

That identity matches the real consumer. It provides a deep module: callers learn
one run interface, one composition rule, and one error rule while the module
owns async normalization and execution.

It does not preclude a DAG later. Parallel, choice, traversal, retry, and recovery
can be AST nodes in a pipeline without exposing a general-purpose graph interface.

### Target contract

1. `Operation[A, B, E]` composes only with `Operation[B, C, E2]`.
2. `run(input, runtime=...)` is the only execution interface and always returns
   the advertised `Result`.
3. Multi-argument Python callables are lifted at the edge; they do not define the
   internal pipeline parameter model.
4. Binding is explicit and cannot silently discard call-time input.
5. A raw operation returns a raw value and raises captured exceptions.
6. A result operation returns `Result` and is typed/flattened exactly once.
7. `None` is ordinary data.
8. Context/environment is a separate runner channel, not a magic kwarg or output
   type.
9. Fail-fast is the default; every lossy or accumulating policy has a distinct
   name.
10. A returned `Operation` is ordinary data unless `flat_map` is used.

## Migration plan

### Phase 0 — Make v0.2.x truthful

Before redesigning internals:

1. Fix or remove every broken README/example import.
2. Add awaits/execution to examples that currently only bind.
3. Put `examples` under pytest and mypy.
4. Add Pyright/Pylance type-contract fixtures.
5. Reproduce Silk's flagship pipeline in downstream CI.
6. Fix the context-decorator overload that infers `Never`.
7. Make returned-`Result` static types match runtime flattening.
8. Put context factory/render/placeholder failures inside the documented result
   boundary, or explicitly document them as throws.
9. Emit deprecation warnings for silent extra-argument dropping and implicit
   error-to-`None`.
10. Raise Silk's fp-ops lower bound and correct Silk's fallback/context-isolation
    documentation.

### Phase 1 — Freeze behavioral contracts

Add law and boundary tests before structural changes:

- `None` preservation;
- consistent single-step and multi-step invocation;
- identity and associativity for the exact claimed domain;
- callable/Operation representation parity;
- one table-driven error-policy suite;
- context compatibility and parallel policy;
- compile-time good/bad pipeline tests;
- sync, async, and sync-returning-awaitable normalization.

Mark which current behaviors are compatibility constraints and which are bugs.

### Phase 2 — Introduce the new core

Build the unary linear core next to a compatibility adapter:

- immutable pipe AST;
- explicit `run`;
- explicit raw/result lifting;
- typed `then`, `map`, `flat_map`, `recover`, `or_else`, `parallel`, and
  traversal nodes;
- one callable normalizer;
- explicit runtime/context object;
- fail-fast default.

Implement old `Operation` interfaces through the new core where semantics are
unambiguous. Do not reimplement a general DAG unless a concrete multi-source,
multi-tail use case exists.

### Phase 3 — Remove the ambiguous surface

In the next breaking release:

- remove awaitable `Operation` values if `.run` replaces them;
- remove representation-dependent `__call__`;
- remove silent excess-argument dropping;
- remove implicit error-to-`None`;
- remove or rename the unused graph vocabulary;
- collapse duplicate composition helpers;
- require explicit `flat_map` for operation-producing callbacks.

## Suggested issue sequence

The following sequence minimizes churn:

1. Consumer type-test harness plus Silk fixture.
2. Context decorator overload fix.
3. Result-flattening type fix.
4. README/examples correction.
5. `MISSING` sentinel for `None`.
6. Unified exception boundary.
7. Unified callable normalization.
8. Explicit error policies for build/map/filter/zip/merge/tap.
9. Explicit binding/run split.
10. Linear AST replacing decorative DAG.
11. Explicit context runtime/state semantics.
12. Deprecation and compatibility release.

The first four items improve trust without deciding every redesign detail. Items
five through eleven should be developed as one coherent interface change, not
independent annotation patches.
