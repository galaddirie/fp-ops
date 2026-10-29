# Migrating to FP-Ops 0.5

0.5 is additive for almost all code. Check these points:

- **Template protocol variance.** `OperationTemplate2`–`OperationTemplate6`
  type parameters are now invariant. Code that only *calls* templates is
  unaffected; code that annotates variables as `OperationTemplateN[...]` may
  need exact parameter types.
- **Undeclared requirements of `flat_map` results.** A missing key needed by an
  operation returned from `flat_map` / `recover_with` now raises
  `UndeclaredRequirementError`, a subclass of `EnvironmentError`, so existing
  `except EnvironmentError` handlers still match. Declare the key with
  `requires=[KEY]` to have it checked before the run starts.

## Replacing common workarounds

| Before | 0.5 |
|---|---|
| `flat_map` recursion to repeat a step | `loop(step)` with `Continue` / `Stop` |
| `fanout_all_settled(use)` + manual release | `bracket(acquire, use, release)` |
| Closures over a value read earlier (`flat_map` + `constant`) | `provide(KEY, read, body)` + `from_environment(KEY)` |
| Rebuilding an operation to pass a runtime URL | `template(_, derive(read_url))` |
| `recover` with a handler that re-raises other errors | `recover(op, handler, on=SomeError)` |
| Splitting `list[Result]` by hand | `.map(partition)` |
| `tap` with an operation inside a callable | `passthrough(effect_operation)` |
| `constant(x)` wrappers in `build` schemas | plain scalars, or `build(Model, field=...)` |
