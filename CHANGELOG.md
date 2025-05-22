# Changelog

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
