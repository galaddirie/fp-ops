# FP concept and naming audit

- Audit date: 2026-07-30
- Repository state: `07358e2` (`v0.2.11`)
- Evidence base: `src/fp_ops/`, focused executable probes, then README comparison
- Perspective: Haskell, Scala Cats, fp-ts, and production Python

## Verdict

FP-Ops has a sound, useful kernel hiding inside a larger abstraction that makes
stronger claims than it can support.

The context-free, unbound, unary subset is an asynchronous exception-aware
Kleisli arrow. Its sequential composition and fail-fast behavior are familiar
and mostly unsurprising. The full `Operation` type is not isomorphic to a
standard monad transformer, however. Argument templates change execution
semantics, context values are reinterpreted as hidden state updates, parallel
composition loses those updates, and several combinators silently replace
errors with `None` or omission.

`Result` is an `Either`, not a `Validation`. The core executor short-circuits on
the first `Error`. `&` starts both branches but still selects one error rather
than accumulating them. `build`, `zip`, `merge`, and effectful `filter` do
something else again: they locally reinterpret failure as missing data.

The identity crisis is therefore real, but it has two layers:

1. semantic drift: different APIs assume Kleisli arrows, Reader, State,
   validation, optional values, or best-effort batch processing;
2. vocabulary drift: Haskell operators, a Haskell `bind`, continuation jargon,
   Rx-style recovery, and lodash-like object helpers coexist without one stable
   convention.

An FP practitioner will wince most at `>>`, `sequence`, the comment equating
`constant` with `identity`, and the comment treating JavaScript binding and
Haskell binding as versions of the same operation. Those are terms of art, not
generic synonyms. More seriously, context and argument binding break identity,
so the full public abstraction is not merely unconventionally named; it is not
lawful under the concepts its surface evokes.

## What `Operation` actually is

### The lawful-looking kernel

Ignore context, partial application, placeholders, and multi-argument sources.
For a unary operation:

```text
M X = Awaitable[Either[Exception, X]]

Operation₀[A, B]
  ≅ Kleisli[M, A, B]
  ≅ A -> Awaitable[Either[Exception, B]]
  ≅ A -> ExceptT[Exception, Async, B]
```

Python's `expression.Result` supplies the `Either`; `Executor.run` catches
ordinary exceptions and returns `Error`, accepts an already returned `Result`,
and short-circuits on `Error` (`src/fp_ops/execution.py:345-353`). `>>` passes
the successful output to the next arrow (`src/fp_ops/operator.py:275-307`).

For a multi-argument source, `A` above is really the argument record described
by `ParamSpec P`:

```text
Args[P] -> Awaitable[Either[Exception, R]]
```

That is an effectful arrow. It is not, by itself, the Haskell type constructor
`m a`, and `>>` is not monadic sequencing. The corresponding Haskell operation
is Kleisli composition, `(>=>)`.

If the ordinary input `A` is fixed and treated as an environment, then
`Operation₀[A, B]` can also be viewed as `ReaderT[A, M, B]`, which is a monad in
`B`. Its `pure` ignores `A`, and its `bind` must give the same `A` to the
operation selected by the continuation. `constant` can play that `pure` role in
restricted probes. The current `bind` does not reuse `A`, which is why an
input-requiring returned operation fails.

### Context starts as Reader and turns into ad hoc State

A context that is only injected and read would give:

```text
OperationReader[C, A, B]
  ≅ A -> ReaderT[C, ExceptT[Exception, Async], B]
  ≅ A -> C -> Awaitable[Either[Exception, B]]
```

That is a reasonable Reader interpretation. The implementation goes further.
If a context-aware step returns any `BaseContext`, the executor treats the value
as a new hidden context and preserves the prior pipeline value
(`src/fp_ops/execution.py:357-365`). That resembles an intended State stack:

```text
A -> C -> Awaitable[Either[Exception, (B, C)]]
```

But it is not that stack:

- the state transition is inferred from `isinstance(value, BaseContext)`;
- a normal domain value cannot safely be a `BaseContext`;
- state and output are not returned as a pair;
- a terminal context-returning step exposes the context as the result, while
  the same step followed by another operation hides it and restores its input;
- `&` runs branches with the same context and does not combine returned context
  updates (`src/fp_ops/operator.py:579-595`);
- mutable context objects are shared concurrently between `&` branches.

The closest description of a context-aware step is consequently:

```text
(A, C) -> Awaitable[Either[Exception, B | C]]
```

with an interpreter-only rule saying that the `C` case sometimes means
"replace hidden state and keep `A`." There is no standard transformer
isomorphic to the complete behavior.

The binding/template interpreter adds another independent layer. A one-node
operation and a pipeline do not consume bound and runtime arguments the same
way (`src/fp_ops/operator.py:203-254`,
`src/fp_ops/execution.py:273-303`). No monad transformer accounts for
representation-dependent argument precedence.

### `Result` is `Either`, not `Validation`

The core behavior is:

```text
Ok(a)    >>= f  = f(a)
Error(e) >>= f  = Error(e)
```

That is `Either`/`ExceptT`: sequential and fail-fast. There is no semigroup for
combining errors and no error accumulation.

`&` does run both effects using `asyncio.gather`, but after both finish it checks
the left result first and returns that error, otherwise the right
(`src/fp_ops/operator.py:579-589`). Two invalid branches therefore execute
twice and report once. That is a parallel product with positional error
priority, not Validation.

`build`, `zip`, effectful `filter`, and `merge` are not Validation either. They
discard errors as `None`, removal, or an empty dictionary. Validation preserves
and accumulates typed errors; it does not erase them.

## Law checks

These results came from executable probes against the checked-out source.
Equality below means normalized `Ok`/`Error` observations unless explicitly
called literal equality.

| Structure under test | Result | Evidence |
|---|---|---|
| Functor identity, plain unary subset | Pass | `add_one >> identity` and `add_one` both returned `Ok(4)` |
| Functor composition, plain unary subset | Pass | two pure transforms composed both ways returned `Ok("v=8")` |
| Identity after argument binding | **Fail** | adding `>> identity` changed how runtime keyword arguments were handled |
| Identity after a context update | **Fail** | context update alone returned the new context; with `>> identity` it returned the original data value |
| `>>` associativity, pure three-stage probe | Observational pass | both groupings returned `Ok("v=8")` |
| `>>` associativity, failing middle stage | Observational pass | both ran the first two effects and returned the same middle error |
| README's literal `(a >> b) >> c == a >> (b >> c)` | **Fail** | `Operation` has no structural/observational `__eq__`; the expression evaluates `False` |
| Monad laws using `constant` and nullary returned operations | Restricted pass | left identity, right identity, and associativity passed three focused examples |
| `bind` with an input-requiring returned operation | **Fail** | returned operation is executed without the original input |
| `&` same-input product | Pass | `add_one & times_two` returned `Ok((4, 6))` for `3` |
| `&` associativity | Only up to tuple reassociation | results were `((4, 6), "v=3")` and `(4, (6, "v=3"))` |
| `&` error accumulation | **Fail** | both failing effects ran; only the left error survived |
| `&` context/state propagation | **Fail** | `bump >> read` returned `1`; `(bump & bump) >> read` returned `0` |
| `|` associativity, three pure alternatives | Pass in the focused case | both tried `a`, then `b`, then returned the success |
| Alternative identity | Not implementable from current surface | there is no `empty`; an always-failing right branch overwrites the original error |
| Canonical `sequence` behavior | **Fail** | current method returned final pipeline value `8`; canonical sequencing would collect `[4, 6]` |

The positive associativity probes are evidence for linear graph concatenation,
not a proof over the full template/context surface. The README should not use
Python `==` for an unimplemented equality. A defensible claim today is:
"linear, unbound, stateless pipelines are observationally associative under
successful values and fail-fast errors." Property tests should guard that
narrow statement.

Minimal identity failure through context:

```python
class Ctx(BaseContext):
    counter: int = 0

@operation(context=True, context_type=Ctx)
async def bump(value: int, *, context: Ctx) -> Ctx:
    return context.model_copy(update={"counter": context.counter + 1})

await bump.execute(7, context=Ctx())               # Ok(Ctx(counter=1))
await (bump >> identity).execute(7, context=Ctx()) # Ok(7)
```

Minimal `bind` boundary failure:

```python
@operation
def source(x: int) -> int:
    return x * 2

@operation
def needs_input(x: int) -> int:
    return x + 10

await source.bind(lambda _value: needs_input).execute(3)
# Error(TypeError: needs_input() missing 1 required positional argument: 'x')
```

`bind` executes the returned operation with only "rest" arguments after the
upstream result (`src/fp_ops/operator.py:692-716`). In an ordinary unary
pipeline there are no rest arguments. The signature nevertheless advertises
an arbitrary `Operation[Q, S]`.

## Severity-ranked findings

### C1 — Wrong concept: the full `Operation` is not one lawful effect type

**Source:** `src/fp_ops/execution.py:305-365`,
`src/fp_ops/operator.py:203-307`

The public class combines an exception-aware Kleisli arrow, hidden Reader
injection, output-sensitive State updates, and a partial-application DSL. These
layers are not orthogonal. Adding identity changes bound execution, and adding a
successor changes whether a returned context is data or control.

This is the root issue. No naming pass can make the complete type a lawful
Functor, Monad, Arrow, or State transformer while those semantics remain.

**Fix:** choose an explicit core:

```text
Operation[A, B] = A -> Env -> Awaitable[Result[B, Exception]]
```

Keep `Env` read-only. If state transitions are required, represent them
uniformly as `(B, State)` internally or create a separate stateful operation
type. Never infer control flow from the runtime class of a successful value.
Move partial application to an explicit source adapter rather than changing
the execution semantics of `Operation`.

### C2 — Wrong concept: identity is observably false at two public boundaries

**Source:** `src/fp_ops/operator.py:203-254`,
`src/fp_ops/execution.py:273-303,357-371`

For a pre-bound head, a single node accepts runtime overrides while a pipeline
drops call-time input. A context-returning terminal step exposes a different
value than that same step followed by identity.

Without identity, `Operation` is not even a category over its whole advertised
surface. This matters more than whether `>>` is aesthetically Haskell-like.

**Fix:** one binding algorithm for one node and many nodes; one uniform outcome
shape for context state; property tests for left/right identity and
associativity over bound, unbound, erroring, and context-aware operations.

### C3 — Wrong concept: `constant` is not `identity`

**Source:** `src/fp_ops/operator.py:858-882`

The source says `# todo: change to be alias for identity`. Those functions are
categorically distinct:

```text
identity(x) = x
constant(a)(_) = a
```

`identity` is the identity arrow. `constant(a)` is the lifted constant function
and is the closest available candidate for `pure(a)` when the input environment
is ignored.

**Fix:** delete the TODO. Make `identity` generic rather than `Any -> Any`, and
document `constant` as a constant arrow. Add a deliberately named `pure` only
after deciding which monad instance it belongs to.

### H1 — Wrong concept: `&` is neither `&`, Validation, nor a complete Applicative

**Source:** `src/fp_ops/operator.py:564-596`,
`src/fp_ops/composition.py:80-115`

The operation duplicates the same input, starts all branches concurrently,
retains tuple position, and chooses the first positional error after all
branches finish. This is parallel fanout:

```text
f &&& g = \x -> (f x, g x)
```

with effects. It is not Haskell `(&)`, which is reverse function application.
It is not `(***)`, which applies functions to different sides of an input pair.
It is not `<*>`, because no function-valued effect is applied. It can be viewed
as a specialized parallel `liftA2(tuple)`, but the API supplies no Applicative
interface or laws.

Context makes it worse. Returned immutable context updates disappear inside the
tuple, while in-place mutations race on the same object. Completion order is
scheduler-dependent. Output order and error selection are left-to-right, but
effect completion order is not specified.

**Fix:** rename the primitive `fanout_parallel` or `parallel_on`, define whether
context is copied, shared read-only, or mergeable, and choose one error policy:
fail-fast with cancellation, all-settled results, or accumulated validation
errors. Do not call it Applicative or Validation until that policy has laws.

### H2 — Wrong concept: one `Result` boundary has at least five error algebras

**Source:** `src/fp_ops/execution.py:345-353`,
`src/fp_ops/operator.py:463-526,579-620`,
`src/fp_ops/sequences.py:46-74,112-136,233-255`,
`src/fp_ops/objects.py:79-97,192-232,284-309`

The policies are:

- `>>`, effectful `map`, `reduce`, and `combine`: fail fast;
- `|`, `catch`, and `default_value`: recover;
- `&`: run all, keep one positional error;
- effectful `filter`: treat an error as predicate `False`;
- `zip`: replace each error with `None`;
- `build`: replace operation/callable failure with `None`;
- `merge`: skip a failed source or insert `None`;
- `tap`: swallow every ordinary exception.

These are all potentially useful policies, but they cannot hide behind the same
undifferentiated `Result` story. In particular, `None` is valid Python data, so
lossy combinators make failure indistinguishable from success containing
`None`.

**Fix:** make error policy explicit in names and return types:
`traverse` should fail fast, `validate_all` should accumulate typed errors,
`all_settled` should return per-item `Result`, and names containing
`best_effort`/`drop_errors`/`errors_as_none` should be reserved for intentional
loss. `tap` should propagate by default; offer `tap_best_effort` separately.

### H3 — Broken effect normalization: returned awaitables are not flattened

**Source:** `src/fp_ops/operator.py:60-75`,
`src/fp_ops/execution.py:345-351`

`_ensure_async` awaits a function only when
`inspect.iscoroutinefunction(fn)` is true. A synchronous function returning an
awaitable is wrapped in an async function that returns the awaitable unchanged:

```python
def delayed(x: int):
    return asyncio.sleep(0, result=x + 1)

result = await operation(delayed).execute(1)
# Ok(<coroutine object ...>), not Ok(2)
```

This is a missing `join`: the library creates a nested async effect instead of
normalizing it. The test suite currently emits a "coroutine was never awaited"
warning.

**Fix:** call the function, inspect the returned value, and await it when it is
awaitable, regardless of how the function was declared. Decide separately
whether returned `Result` and returned `Operation` are flattened; encode those
choices in types and names.

### H4 — Broken higher-order composition: a mid-pipeline operation is data

**Source:** `src/fp_ops/operator.py:289-303`,
`src/fp_ops/composition.py:47-58`,
`src/fp_ops/operator.py:671-765`

`pipe` converts a lambda to an ordinary operation. If that lambda returns an
`Operation`, the next stage receives the `Operation` object; it is not flattened.
The README explicitly claims otherwise at `README.md:78-81`:

```python
double >> (lambda x: add(x, 10))  # returns Ok(<Operation>)
double >> add(_, 10)              # executes add
```

The first expression is `map` producing a nested effect. Flattening it would be
`flat_map`. `bind` and `apply_cont` contain two different attempts to provide
that behavior.

`when` is broken for the documented calling form too. The decorator wraps the
outer builder itself (`src/fp_ops/flow.py:115-128`), so
`when(pred, transform)(value)` adds a third argument to the outer function and
returns `Error(TypeError)`.

**Fix:** provide separate, typed `map` and `flat_map`; never dynamically guess
between them. Remove or repair `when` as an ordinary builder returning a
decorated inner operation.

### H5 — Wrong concept: `bind` only implements a restricted flat-map

**Source:** `src/fp_ops/operator.py:671-723`

The comment says it should become "a real bind method, like js bind or haskell's
bind." JavaScript `Function.prototype.bind` is partial application and receiver
binding. Haskell `(>>=)` is monadic substitution. They are unrelated.

The current method is monadic only when the builder returns an operation that
can run without an ordinary input, such as a bound operation or `constant`.
Accepting a plain callable broadens it further: the callable itself becomes the
next operation and is then invoked with the remaining arguments, not with the
upstream value that selected it.

`apply_cont` accepts raw values, awaitables, `Result`, and `Operation`
(`src/fp_ops/operator.py:725-765`). That is a more general dynamic flattening
operation hidden under continuation jargon, and it does not propagate context
metadata.

**Fix:** reserve `flat_map` for exactly
`Callable[[B], Operation[Unit, C]]` if operations are computations, or redesign
Reader/Kleisli bind to pass the same environment. Use `map` for raw/awaitable
values and `recover` for errors. Remove `apply_cont` after those cases have
explicit homes. Use `partial`/`bind_args` only for argument binding.

### H6 — Documentation claims an API and laws that the source does not contain

**Source:** `src/fp_ops/composition.py:30-150`,
`src/fp_ops/__init__.py:1-18`; claims at `README.md:14-20,127-155,183-220,284-292`

Executed discrepancies:

- `from fp_ops.composition import sequence, map, transform` raises
  `ImportError`; the module exports only `pipe`, `compose`, `parallel`, and
  `fallback`.
- `fp_ops.data` does not exist.
- `Operation.transform` and `Operation.filter` do not exist.
- README `sequence(op1, op2, op3)` does not exist; the only implementation is
  `Operation.sequence([op1, op2, op3])`.
- the README says sequence collects all results, while the implementation pipes
  outputs and returns only the last.
- the README's `build` field composes backwards:
  `sum_by("amount") >> get("user.orders")`. With an equivalent local
  `sum_amounts`, the documented order raised `TypeError`; the correct
  left-to-right order returned `300`:
  `get("user.orders") >> sum_amounts`.
- `pluck`, `sum_by`, and many other advertised data operations have no source
  implementation.
- the literal associativity equation uses an equality the class does not
  implement.

The README is not merely stale; it teaches users the wrong higher-order and
composition model.

**Fix:** generate the API reference from exports, execute every README snippet
in CI, and state laws only at the scope where property tests establish them.

### M1 — Wrong name for the right concept: `>>` is not monadic `>>`

**Source:** `src/fp_ops/operator.py:275-307`

In Haskell:

```text
(>>)  :: m a -> m b -> m b       -- discard the left value
(>=>) :: (a -> m b) -> (b -> m c) -> (a -> m c)
(>>>) :: Arrow a => a b c -> a c d -> a b d
```

FP-Ops passes the left successful value into the right operation. That is
Kleisli left-to-right composition (`>=>`) or arrow composition (`>>>`), not
monadic `then` (`>>`).

Python users may reasonably read `>>` as a pipeline symbol, so this is not a
runtime bug. It becomes a problem when the library also uses words like
`bind`, `pure`-adjacent `constant`, and `Alternative`, inviting Haskell laws.

**Fix:** make `.then(...)` and `pipe(...)` the documented primary API. Keep `>>`
as optional syntax described only as a pipeline alias. Do not explain it using
Haskell `(>>)`.

### M2 — Wrong name for the right concept: `sequence` is composition

**Source:** `src/fp_ops/operator.py:629-647`

Canonical `sequence` has the shape:

```text
sequence :: [m a] -> m [a]
```

FP-Ops folds `>>` over operations:

```text
[A -> M B, B -> M C, C -> M D] -> A -> M D
```

That is `pipe_all`, `flow`, or composition of Kleisli arrows. It neither accepts
homogeneous effects in the canonical sense nor collects their results.

**Fix:** rename it `pipe_all`/`flow`. If a real `sequence` is added, define
whether it is sequential or parallel and return the container of values.

### M3 — Wrong names and overloaded concepts in collection operations

**Source:** `src/fp_ops/sequences.py:25-257`

| Current surface | Actual concept | Problem | Recommended name/policy |
|---|---|---|---|
| `map(callable)` | ordinary collection map | Sound | `map_each` in a pipeline API |
| `map(Operation)` | sequential fail-fast traversal | This is `traverse`, not pure map | `traverse` or `map_each_operation` |
| `filter(callable)` | ordinary collection filter | Sound | `filter_each` |
| `filter(Operation)` | effectful filter with errors treated as `False` | Not `filterM`; errors vanish | fail-fast `filter_m`, or explicit `filter_best_effort` |
| `reduce(callable)` | left fold | Sound | `reduce`/`fold` |
| `reduce(Operation)` | sequential effectful left fold | This is `foldM` | `fold_m` or `reduce_operation` |
| `zip(*operations)` | apply every operation to every item, errors as `None` | Not zip and not parallel | `fanout_each(..., errors="none")` |

Canonical `zip` combines multiple containers positionally. This implementation
is `map(fanout(...))`, executed sequentially. Its docstring calls it "parallel
map" and immediately says operations run sequentially
(`src/fp_ops/sequences.py:190-198`).

The library also has three distinct `map`-like ideas: successful-output mapping
(claimed as nonexistent `transform`), pure collection mapping, and effectful
collection traversal. They need three explicit names or one principled
overloaded protocol with sound types.

`Operation.filter` exists only as a README claim (`README.md:286-288`), so its
intended behavior cannot be determined from source. If it is meant to retain or
reject one successful value, then it needs an explicit empty/failure meaning:
`mfilter`/`guard` for a `MonadPlus`, or more plainly `ensure(predicate, error)`
for this Either-like effect. It is not the same operation as collection filter.

### M4 — Recovery vocabulary does not state whether handlers are pure or effectful

**Source:** `src/fp_ops/operator.py:492-526`,
`src/fp_ops/flow.py:50-89`

- `catch` accepts an exception-to-raw-value function. In Cats this is
  `handleError`; in RxJS, `catchError` returns another observable. `recover` is
  the least surprising cross-language name.
- `default_value` constructs a new recovering operation. On
  `expression.Result`, `default_value` extracts a value immediately. The same
  name has two different abstraction levels.
- `attempt` is just `operation(...)`; all operations already capture exceptions.
  In Cats, `attempt` exposes an error as `Either` inside the effect. Here it adds
  no semantic layer.

**Fix:** use `recover(handler)`, `recover_with(handler_returning_operation)`,
and `default_on_error(value)`. Remove `attempt`, or redefine it only if there is
an unsafe operation whose exception channel it explicitly reifies.

### M5 — Context injection is unnamed Reader; context replacement is unnamed State

**Source:** `src/fp_ops/operator.py:409-453`,
`src/fp_ops/execution.py:305-365`

Automatic injection of a read-only environment is Reader. Returning a new
environment is State. Calling both simply "context awareness" hides the
important difference between dependency access and state transition.

**Fix:** call the read-only input `environment` and expose `ask`/`asks` only if
FP vocabulary is desired. Otherwise use ordinary dependency-injection wording.
If state must change, make that a separate explicit `state` channel with
`get_state`, `set_state`, and a uniform transition result.

### M6 — Lodash-flavored object names only partly match lodash

**Source:** `src/fp_ops/objects.py:30-76,127-331`,
`src/fp_ops/utils.py:74-85`

- `get` is a recognizable path getter, but an existing `None` is treated as
  missing and replaced by the default. Lodash's default applies to
  `undefined`/missing, not a present `null`.
- `merge` is shallow; lodash `merge` is recursive. The source uses
  `dict.update` (`src/fp_ops/objects.py:293-309`).
- `update` performs immutable shallow assignment from a mapping. Lodash
  `update` means path plus updater; Python `dict.update` mutates.
- `default` is `None` coalescing, closer to lodash `defaultTo`.

These functions do not imply lenses. `get` is only a getter, and `update` has
no path/focus or setter relationship with it. Lens laws are therefore not
applicable; the problem is ordinary naming, not a broken lens.

**Fix:** use `get_path` with an explicit missing sentinel, `merge_shallow` or
`assign`, `assign_fields`, and `default_to`. Add lens terminology only if
`view`, `set`, and `over` share a lawful optic.

### M7 — Right concepts under unrecognizable names: `apply_cont` and `combine`

**Source:** `src/fp_ops/operator.py:649-765`

`apply_cont` dynamically maps or flattens several possible return forms. Outside
continuation-passing-style internals, users will not recognize it as the
operation they wanted. `combine(**kw_ops)` applies named arrows to one value and
builds a record; in fp-ts this resembles `sequenceS`, and in applicative builder
APIs it is often `struct`.

**Fix:** split `apply_cont` into `map`, `flat_map`, and recovery operations.
For the target audience, name `combine` `collect_named` or `fanout_named`.
`struct`/`sequenceS` would be precise but less recognizable to lodash/RxJS
users.

### N1 — Nits and smaller naming inconsistencies

- `fail` is a recognizable `raiseError`/`throwError` constructor, but its return
  is typed `Operation[Any, None]` even though the wrapped function is
  `NoReturn` (`src/fp_ops/flow.py:27-48`).
- `fallback` accurately describes variadic first-success behavior
  (`src/fp_ops/composition.py:118-150`). Prefer the same word or `or_else`
  everywhere rather than mixing it with unexplained `|`.
- `pipe` and `compose` have internally correct left-to-right and right-to-left
  directions (`src/fp_ops/composition.py:30-77`). `pipe` is a reasonable choice
  for the intended audience.
- `parallel` is reasonably named, but it should say "same-input fanout" and
  document output order, completion order, cancellation, and error selection.
- `branch`, `retry`, and `loop_until` are recognizable operational names. They
  need stronger type/error contracts, not FP renaming.

## Recommended naming tradition

Choose a **Python-first explicit pipeline vocabulary**. Do not make Haskell
symbols the conceptual API, and do not copy RxJS stream names where a
single-shot async operation has different semantics. The intended lodash and
RxJS audience already understands `pipe`, `map`, `flat_map`, `tap`, `retry`,
and explicit English error handling. Python users benefit from names that state
data shape and policy.

Recommended primary surface:

| Current | Recommended primary name |
|---|---|
| `a >> b` | `a.then(b)`; retain `>>` as syntax sugar |
| pure successful-output transform, currently missing | `a.map(fn)` |
| lawful operation-returning continuation | `a.flat_map(fn)` |
| `a & b` | `a.fanout_parallel(b)` |
| `a | b` | `a.or_else(b)` |
| `Operation.sequence(ops)` | `pipe_all(ops)` |
| `Operation.combine(**ops)` | `collect_named(**ops)` |
| `catch` | `recover` |
| `default_value` on `Operation` | `default_on_error` |
| `tap` swallowing errors | `tap_best_effort`; make `tap` propagate |
| `sequences.map` | `map_each`; use `traverse` only for the fail-fast effectful form |
| `sequences.filter` | `filter_each`; expose error policy explicitly |
| `sequences.zip` | `fanout_each` |
| `objects.merge` | `merge_shallow` or `assign` |
| `objects.update` | `assign_fields` |
| `utils.default` | `default_to` |

This tradition is intentionally less academically compact. It makes the
important distinctions visible:

- `map` versus `flat_map`;
- sequential pipe versus same-input parallel fanout;
- fallback versus an `Alternative` instance;
- fail-fast traversal versus best-effort item processing;
- read-only environment versus mutable state.

Symbols can remain ergonomic aliases after the named operations are the
documented source of truth and share their tests.

## What is already right

- The context-free unary execution path is a recognizable and useful
  `ExceptT`-over-async Kleisli arrow.
- Sequential errors short-circuit cleanly at the executor boundary.
- Returning an `expression.Result` from a step is flattened one level rather
  than wrapped as `Ok(Result(...))` (`src/fp_ops/execution.py:351-353`).
- `pipe` and `compose` agree with their documented directions.
- `fallback` is a clear name for variadic first-success behavior.
- Plain `map`, `filter`, and `reduce` branches implement familiar collection
  operations; their issue is overloading them with effectful policies.
- `get` is a normal path getter, not a fake lens.
- `&` preserves positional output order even though completion order is
  concurrent.
- The current regression suite is broad: `pytest -q` completed with 508 passing
  tests. The problem is that tests largely lock in implementation behavior
  rather than assert the advertised algebraic laws.

## Decisions the source does not reveal

The following cannot be settled by renaming:

1. Is context an immutable environment, mutable service container, state value,
   or all three?
2. Should a `BaseContext` ever be valid pipeline data?
3. Is `tap` intentionally best-effort, or should observability failures fail an
   operation?
4. Are `build`, `merge`, `zip`, and effectful `filter` intended for lossy ETL?
   If so, is `None` the desired failure representation?
5. Should parallel failures cancel siblings, await every sibling, accumulate
   errors, or return per-branch `Result` values?
6. Is an `Operation` principally an input/output arrow or a suspended
   computation? The correct `flat_map` signature depends on the answer.

Those are product decisions. Until they are explicit, adding more FP names will
increase the identity crisis.

## Suggested repair order

1. Freeze new combinator names.
2. Choose the unary `Operation[A, B]` semantic core and make context
   read/update behavior explicit.
3. Make binding independent of pipeline representation; restore identity.
4. Define one default error algebra and explicit names for all lossy policies.
5. Add law/property tests for the restricted structures actually claimed.
6. Introduce the Python-first named API; keep operators as compatibility aliases.
7. Split collection `map`/`filter`/`reduce` from their effectful variants.
8. Regenerate and execute the README against real exports.

Renaming before steps 2–4 would give broken concepts cleaner labels. The
semantic split has to come first.

## Verification performed

```text
poetry run pytest -q
508 passed, 102 warnings

Focused law and semantics probes:
- functor identity/composition
- `>>` literal and observational associativity
- restricted `constant`/`bind` monad laws
- `&` fanout, grouping, effect scheduling, dual failure, and context
- `|` associativity, dual failure, and absent identity
- Either versus Validation behavior
- `sequence`, effectful `map`, and `zip`
- higher-order lambda, `when`, and `tap`
- `get`, `build`, `merge`, and `update`
- README build direction and import availability
```
