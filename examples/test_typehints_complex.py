from typing import Any, List, Dict, Optional, TypeVar, Generic, Type
from pydantic import BaseModel, Field
from fp_ops.operator import operation, Operation, P, R
from fp_ops.context import BaseContext
from fp_ops.flow import attempt, retry, branch, loop_until, wait
from fp_ops.composition import (
    sequence, pipe, compose, parallel, fallback,
    transform, map, filter, reduce, zip, flat_map,
    group_by, partition, first, last
)
from fp_ops.placeholder import _

# Example context types
class UserContext(BaseContext):
    user_id: str
    role: str = "user"
    permissions: List[str] = Field(default_factory=list)

class DatabaseContext(BaseContext):
    connection_string: str
    timeout: float = 5.0
    max_retries: int = 3

# Example operations with contexts
@operation(context=True, context_type=UserContext)
async def get_user_profile(user_id: str, *, context: UserContext) -> Dict[str, Any]:
    """Get user profile with context-aware operation."""
    return {
        "id": user_id,
        "role": context.role,
        "permissions": context.permissions
    }

@operation(context=True, context_type=DatabaseContext)
async def query_database(query: str, *, context: DatabaseContext) -> List[Dict[str, Any]]:
    """Database query with context-aware operation."""
    # Simulate database query
    return [{"result": "data"}]

# Higher-order operations with type checking
@operation
async def validate_user(user: Dict[str, Any]) -> bool:
    """Validate user data."""
    return bool(user.get("id") and user.get("role"))

@operation
async def enrich_user_data(user: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich user data with additional fields."""
    return {**user, "enriched": True}

# Example of using flow functions with type checking
async def demonstrate_flow_functions():
    # Attempt with type checking
    safe_query = attempt(query_database, context=True, context_type=DatabaseContext)
    
    # Retry with type checking
    retry_query = retry(safe_query, max_retries=3, delay=0.1)
    
    # Branch with type checking
    user_pipeline = branch(
        validate_user,
        enrich_user_data,  # true branch
        lambda x: {"error": "Invalid user"}  # false branch
    )
    
    # Loop until with type checking
    async def is_data_valid(data: Dict[str, Any]) -> bool:
        return data.get("enriched", False)
    
    enrichment_loop = loop_until(
        is_data_valid,
        enrich_user_data,
        max_iterations=3,
        context=True,
        context_type=UserContext
    )

# Example of using composition functions with type checking
async def demonstrate_composition_functions():
    # Create contexts
    user_ctx = UserContext(user_id="123", permissions=["read", "write"])
    db_ctx = DatabaseContext(connection_string="postgresql://localhost/db")
    
    # Sequence with type checking
    user_sequence = sequence(
        get_user_profile,
        enrich_user_data,
        validate_user
    )
    
    # Pipe with type checking
    user_pipe = pipe(
        get_user_profile,
        lambda user: enrich_user_data if user.get("role") == "admin" else validate_user,
        lambda result: {"status": "success" if result else "failure"}
    )
    
    # Parallel with type checking
    parallel_ops = parallel(
        get_user_profile,
        query_database
    )
    
    # Map with type checking
    @operation
    async def process_user(user: Dict[str, Any]) -> Dict[str, Any]:
        return {**user, "processed": True}
    
    user_list = [{"id": "1"}, {"id": "2"}]
    map_operation = map(process_user, max_concurrency=2)
    
    # Filter and reduce with type checking
    filter_operation = filter(validate_user, lambda x: x.get("role") == "admin")
    reduce_operation = reduce(
        map_operation,
        lambda acc, curr: {**acc, "count": acc.get("count", 0) + 1}
    )
    
    # Group by and partition with type checking
    group_by_role = group_by(
        map_operation,
        lambda user: user.get("role", "unknown")
    )
    
    partition_admins = partition(
        map_operation,
        lambda user: user.get("role") == "admin"
    )

# Example of complex pipeline with type checking
async def demonstrate_complex_pipeline():
    # Create a complex pipeline with multiple operations
    pipeline = (
        get_user_profile
        >> attempt(enrich_user_data, context=True, context_type=UserContext)
        >> retry(validate_user, max_retries=3)
        >> branch(
            lambda x: x.get("role") == "admin",
            query_database,  # true branch
            lambda x: [{"result": "unauthorized"}]  # false branch
        )
        >> map(transform(lambda x: x.get("result", "")))
        >> reduce(lambda acc, curr: acc + [curr], [])
    )
    
    # Execute with contexts
    user_ctx = UserContext(user_id="123", role="admin", permissions=["read", "write"])
    db_ctx = DatabaseContext(connection_string="postgresql://localhost/db")
    
    # Merge contexts
    merged_ctx = user_ctx.merge(db_ctx)
    
    # Execute pipeline
    result = await pipeline.execute("123", context=merged_ctx)
    print(f"Pipeline result: {result}")

if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("Demonstrating flow functions...")
        await demonstrate_flow_functions()
        
        print("\nDemonstrating composition functions...")
        await demonstrate_composition_functions()
        
        print("\nDemonstrating complex pipeline...")
        await demonstrate_complex_pipeline()
    
    asyncio.run(main()) 