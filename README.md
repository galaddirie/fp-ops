# FP-Ops: Functional Programming Operations for Python

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.4-blue.svg)](https://pypi.org/project/fp-ops/)
[![Python versions](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/fp-ops/)
[![codecov](https://codecov.io/gh/galaddirie/fp-ops/graph/badge.svg?token=8MHGFYBD8V)](https://codecov.io/gh/galaddirie/fp-ops)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type check: mypy](https://img.shields.io/badge/type%20check-mypy-blue)](https://github.com/python/mypy)

FP-Ops is a functional programming library for Python that lets you convert you functions into composable operations.

## Features

- **Composition as a First-class Citizen**: Build complex pipelines using simple operators like `>>`, `&`, and `|`
- **Context Awareness**: Pass context through operation chains with automatic validation
- **Async-First**: Designed for asynchronous operations from the ground up
- **Type Safety**: Comprehensive type hints for better IDE support and code safety
- **Functional Patterns**: Implements common functional programming patterns like map, filter, and reduce

## Installation

```bash
pip install fp-ops
```

## Getting Started

Here's a simple example to get you started:

```python
from fp_ops.operator import operation
import asyncio

# Define some operations
@operation
async def get_user(user_id: int) -> dict:
    # Simulate API call
    return {"id": user_id, "name": "John Doe", "age": 30}

@operation
async def format_user(user: dict) -> str:
    return f"User {user['name']} is {user['age']} years old"

# Compose operations
get_and_format = get_user >> format_user

get_and_format(1)
```

## Key Concepts

### Operations

The core concept in FP-Ops is the `Operation` class. An operation wraps an async function and provides methods for composition using operators:

- `>>` (pipeline): Passes the result of one operation to the next
- `&` (parallel): Executes operations in parallel and returns all results 
- `|` (alternative): Tries the first operation and falls back to the second if it fails

### Placeholders

You can use the placeholder `_` to specify where the result of a previous operation should be inserted:

```python
from fp_ops.placeholder import _

# Define operations
@operation
async def double(x: int) -> int:
    return x * 2

@operation
async def add(x: int, y: int) -> int:
    return x + y

# These are equivalent:
pipeline1 = double >> (lambda x: add(x, 10))
pipeline2 = double >> add(_, 10)
```

### Context Awareness

Operations can be context-aware, allowing you to pass contextual information through the pipeline:

```python
from fp_ops.operator import operation
from fp_ops.context import BaseContext
from pydantic import BaseModel

class UserContext(BaseContext):
    auth_token: str
    user_id: int

@operation(context=True, context_type=UserContext)
async def get_user_data(context: UserContext) -> dict:
    return {"id": context.user_id, "name": "Jane Doe"}

# Initialize context
context = UserContext(auth_token="abc123", user_id=42)

# Execute with context
result = await get_user_data(context=context)
```

## Advanced Usage

### Error Handling

FP-Ops uses the `Result` type for robust error handling:

```python
@operation
async def divide(a: int, b: int) -> int:
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

# Handle errors with default values
safe_divide = divide.default_value(0)

# Or with custom error handling
safe_divide = divide.catch(lambda e: 0 if isinstance(e, ValueError) else -1)
```

### Composition Functions

Besides operators, FP-Ops provides various composition functions:

```python
from fp_ops.composition import sequence, pipe, parallel, fallback

# Run operations in sequence and collect all results
results = await sequence(op1, op2, op3)

# Complex pipelines with conditional logic
pipeline = pipe(
    op1,
    lambda x: op2 if x > 10 else op3,
    op4
)

# Run operations in parallel
combined = await parallel(op1, op2, op3)

# Try operations until one succeeds
result = await fallback(op1, op2, op3)
```

### Higher-Order Flow Operations

FP-Ops provides utilities for creating higher-order operations:

```python
from fp_ops.flow import branch, attempt, retry, wait, loop_until

# Conditional branching
conditional = branch(
    lambda x: x > 0,
    positive_op,
    negative_op
)

# Retry an operation
resilient_op = retry(flaky_operation, max_retries=3, delay=0.5)

# Loop until a condition is met
counter = loop_until(
    lambda x: x >= 10,
    lambda x: x + 1,
    max_iterations=20
)
```

## API Reference

### Core Classes

- `Operation`: The main class representing a composable asynchronous operation
- `BaseContext`: Base class for all operation contexts
- `Placeholder`: Used to represent where a previous result should be inserted

### Decorators

- `@operation`: Convert a function to an Operation
- `@operation(context=True, context_type=MyContext)`: Create a context-aware operation

### Operators

- `op1 >> op2`: Pipeline composition
- `op1 & op2`: Parallel execution
- `op1 | op2`: Alternative execution

### Methods

- `operation.map(func)`: Apply a transformation to the output
- `operation.filter(predicate)`: Filter the result using a predicate
- `operation.bind(binder)`: Bind to another operation
- `operation.catch(handler)`: Add error handling
- `operation.default_value(default)`: Provide a default value for errors
- `operation.retry(attempts, delay)`: Retry the operation
- `operation.tap(side_effect)`: Apply a side effect without changing the value

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.