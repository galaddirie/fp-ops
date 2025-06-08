# Changelog

## [0.2.11] - 2025-06-09

### Added
- Added pydantic model and dataclass support to the `build` operation, making it easier to create type-safe objects from data

### Changed
- **Build Operation API**: Updated `build` operation to support model instantiation:
  - Added optional `model` parameter for type-safe object construction
  - Improved error messages for validation failures
  - Better handling of optional fields and defaults
  - More robust type conversion and validation

## [0.2.10] - 2025-06-08

### Changed
- **Argument Precedence**: Updated how arguments are merged in operations:
  - Runtime keyword args (explicit overrides) now have highest priority
  - Pre-bound constants (template kwargs/positionals) come next
  - Runtime positionals fill remaining empty slots
  - Extra positional args are now silently ignored instead of raising TypeError
- **Placeholder Handling**: Improved how placeholders are processed:
  - Better handling of direct positional placeholders
  - More robust nested placeholder resolution
  - Enhanced multi-step pipeline handling
- **Context Awareness**: Added automatic context detection and forwarding:
  - `build` operation now properly detects and forwards context requirements
  - `map` operation automatically inherits context requirements from mapped operation
  - Added context type inference for better type safety

### Fixed
- Fixed binding behavior to respect pre-bound operations
- Improved test coverage for argument precedence and binding
- Fixed context propagation in complex operation pipelines

## [0.2.9] - 2025-06-07

### Fixed
- fixed type errors


## [0.2.8] - 2025-05-29

### Added
- Added a new helper function to handle chained and nested operations, making it easier to work with complex operation pipelines
- Made the build operation more powerful:
  - It now properly handles nested operations and forwards context through the entire pipeline
  - Better distinction between callable functions and built-in types
  - More graceful handling of operation failures
- Updated the map operation to correctly pass context to each item being processed
- Added a bunch of new tests covering:
  - How operations work inside object schemas
  - Context forwarding through different operation types
  - Complex nested operations in schemas
  - Various error scenarios and how they're handled

### Changed
- Made operation execution more reliable, especially when operations are chained together
- Improved how context flows through nested operations and object schemas, making it more predictable


## [0.2.7] - 2025-05-28

### Fixed
-  Fixed type errors with @operation() decorator and its overloads
-  Fixed type signature of attempt() function to properly handle return types


## [0.2.6] - 2025-05-27

### Added
- **Data Operations Module**: Added comprehensive data manipulation operations:
  - Path-based access: `get` for dot-notation access to nested data
  - Object construction: `build` for schema-based object creation, `merge` for dictionary merging, `update` for dictionary updates
  - Collection operations: `filter`, `map`, `reduce`, `zip`, `contains`, `not_contains`
  - List operations: `flatten`, `flatten_deep`, `unique`, `reverse`
  - Dictionary operations: `keys`, `values`, `items`
  - Utility operations: `length`
- **Flow Operations**: Added `when` operation for conditional transformations
- **Operator Updates**: Simplified `constant` and `identity` operations for better ergonomics
- **Test Coverage**: Added comprehensive test suite for data operations

### Changed
- **Operator Behavior**: Made `constant` and `identity` operations more ergonomic by simplifying their implementation
- **Documentation**: Updated README with data operations examples and usage patterns

### Fixed
- improved type hints


## [0.2.5] - 2025-05-26

### Fixed
- fixed map bugs
- Added test coverage for the `map` function, including scenarios with placeholders, context integration, and concurrent execution.


## [0.2.4] - 2025-05-26

### Fixed
- fix type errors.


## [0.2.3] - 2025-05-26

### Added
- **Iterable Map Function**: Added `fp_ops.composition.map` for applying an operation to each item in an iterable, with support for concurrency control.

### Changed
- **Renamed `map` to `transform`**: The method `Operation.map` and the function `fp_ops.composition.map` (for transforming a single operation's output) have been renamed to `Operation.transform` and `fp_ops.composition.transform` respectively, to avoid confusion with the new iterable map.

## [0.2.2] - 2025-05-30

### Added
- **Append Semantics**: Operations now merge arguments when called multiple times instead of replacing them
- **Partial Method**: Added `partial()` method to Operation class for creating partially applied operations
- **Deep Placeholder Preservation**: Placeholders inside nested structures are properly maintained when appending arguments
- **Context Factory Support**: Context factories are now respected after appending new arguments

### Fixed
- **Better Error Messages**: Improved error messages for operations with too many positional arguments
- **Context Handling**: Fixed context injection when appending arguments to context-aware operations

### Changed
- **Argument Handling**: Changed how single-step operations handle multiple calls - positional args are appended, kwargs are merged with newer values taking precedence

## [0.2.1] - 2025-05-22

### Added
- improved test coverage
- chores


## [0.2.0] - 2025-05-22

### Added
- **Graph-based Execution Model**: Completely new architecture using directed acyclic graphs (DAG) for operations
- **New Primitives Package**: Added primitives.py with core types like Template, OpSpec, Port, and Edge
- **Execution Engine**: New execution.py with ExecutionPlan and Executor classes for more efficient operation execution
- **Lazy Execution**: Operations are now only executed when results are needed
- **Better Type Safety**: Comprehensive type hints with ParamSpec and TypeVar throughout the codebase
- **Graph Visualization**: Added dot_notation() method for visualizing operation pipelines
- **Associativity Guarantee**: Operation composition is now associative: `(a >> b) >> c == a >> (b >> c)`
- **Example Script**: Added example.py with comprehensive usage examples

### Changed
- **Complete Type System Overhaul**: 
  - Operation is now generic over input parameters and return type: `Operation[P, R]`
  - Improved type annotations throughout for better IDE support
- **Simplified API**: 
  - Cleaner operation signatures and interfaces
  - More consistent behavior across composition methods
- **Placeholder Handling**: Improved how placeholders are processed and substituted
- **Context Management**: Better handling of context objects throughout the pipeline
- **Flow Operations**: Redesigned branch, attempt, retry, and other flow operations
- **Composition Functions**: Updated sequence, pipe, parallel, and other composition helpers
- **Error Handling**: More consistent error propagation and recovery
- **Documentation**: Updated README and docstrings to reflect new architecture

### Fixed
- **Runtime Type Checking**: Better validation of types at runtime
- **Context Propagation**: Fixed issues with context objects not being properly passed through pipelines
- **Placeholder Substitution**: More reliable handling of nested placeholders
- **Binding Behavior**: Fixed issues with argument binding and forwarding
- **Operation Chaining**: More predictable behavior when chaining multiple operations

### Technical Details
- **Architecture**: Moved from function-based to graph-based execution model
- **Execution**: Split operation definition from execution for better performance
- **Templates**: Added Template class for handling placeholders and arguments
- **Nodes and Edges**: Operations are now composed as a graph with nodes and edges
- **Development**: Added Graphviz dependency for visualization capabilities

## [0.1.4] - 2025-04-13

### Added
- addded deep chaining context tests

### Fixed
- Always use bound_args for bound operations, regardless of passed args

### Changed


## [0.1.3] - 2025-04-13

### Added

### Fixed
more fixes for function signature, docstring and other metadata

### Changed


## [0.1.2] - 2025-04-13

### Added

### Fixed
preserve function signature, docstring and other metadata

### Changed


## [0.1.1] - 2025-04-12

### Added

### Fixed
package name changed

### Changed



## [0.1.0] - 2025-04-12

### Added
Initial release of fp-ops

### Fixed


### Changed


