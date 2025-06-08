# FP-Ops: Functional Programming Operations for Python

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.7-blue.svg)](https://pypi.org/project/fp-ops/)
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
- **Lazy Execution**: Only execute operations when the result is needed
- **Composition is associative**: `(a >> b) >> c == a >> (b >> c)`

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
from fp_ops.composition import sequence, pipe, parallel, fallback, map, transform

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

# Apply an operation to each item in an iterable
# (e.g., transforming [1, 2, 3] to [2, 3, 4] if item_op increments by 1)
mapped_results = await map(item_op, max_concurrency=5)([item1, item2, item3]) 

# Transform the output of a single operation
transformed_result = await transform(op1, lambda x: x * 2)
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

### Data Operations

FP-Ops provides a rich set of data manipulation operations that make it easy to work with nested data structures, collections, and transformations:

```python
from fp_ops.data import (
    get, pick, pluck, build, merge, update,
    filter_by, group_by, sort_by, unique_by,
    map_values, map_keys, rename, omit,
    count_by, sum_by,
    to_lower, to_upper, strip, split, join
)

# Path-based access
user_data = {
    "user": {
        "profile": {"name": "John", "email": "john@example.com"},
        "orders": [{"id": 1, "amount": 100}, {"id": 2, "amount": 200}]
    }
}

# Get nested values
name = await get("user.profile.name").execute(user_data)
email = await get("user.profile.email", "no-email@example.com").execute(user_data)

# Pick specific fields
user_info = await pick("user.profile.name", "user.profile.email").execute(user_data)

# Extract values from a list
order_amounts = await pluck("amount").execute(user_data["user"]["orders"])

# Build new objects
user_summary = await build({
    "name": get("user.profile.name"),
    "email": get("user.profile.email"),
    "order_count": lambda d: len(d["user"]["orders"]),
    "total_spent": sum_by("amount") >> get("user.orders")
}).execute(user_data)

# Collection operations
users = [
    {"name": "Alice", "role": "admin", "score": 90},
    {"name": "Bob", "role": "user", "score": 85},
    {"name": "Charlie", "role": "user", "score": 75}
]

# Filter and group
active_users = await filter_by({"role": "user"}).execute(users)
by_role = await group_by("role").execute(users)

# Sort and get unique values
sorted_users = await sort_by("score", reverse=True).execute(users)
unique_names = await unique_by("name").execute(users)

# Transform dictionaries
data = {"user_name": "John Doe", "user_email": "john@example.com"}
cleaned = await (
    map_keys(lambda k: k.replace("user_", "")) >>
    map_values(strip) >>
    rename({"name": "full_name"})
).execute(data)

# Aggregate data
status_counts = await count_by("status").execute(orders)
total_amount = await sum_by("amount").execute(orders)
```

These operations can be composed together to create powerful data transformation pipelines:

```python
# Complex pipeline example
pipeline = (
    get("user.orders") >>
    filter_by({"status": "completed"}) >>
    sort_by("amount", reverse=True) >>
    pluck("amount") >>
    sum_by(lambda x: x)
)

total_completed = await pipeline.execute(user_data)
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

- `operation.transform(func)`: Apply a transformation to the output
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