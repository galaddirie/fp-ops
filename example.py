from typing import Any

if __name__ == "__main__":
    from fp_ops.operator import operation
    from fp_ops.placeholder import _
    import asyncio

    @operation
    async def add(a: int, b: int) -> int:
        return a + b
    
    @operation
    async def add_one(a: int) -> int:
        return a + 1

    @operation
    async def mul(x: int, y: int) -> int:
        return x * y

    @operation
    async def identity(value: Any) -> Any:
        return value
    
    @operation
    def to_string(value: Any) -> str:
        return str(value)

    # Simple chaining
    pipeline = add >> add_one >> to_string
    pipeline2 = add >> add_one(2)
    pipeline3 = add >> add_one(_)

    pipeline4 = add >> mul(_, 3)
    pipeline5 = add >> mul(3, _)
    pipeline6 = add >> mul(1, 2)
    pipeline7 = add >> mul(_, 4)

    pipeline7 = add >> identity(_)

    pipeline8 = add(1,2) >> add_one
    pipeline9 = add(1,2) >> add_one(1)

    # placeholders down the chain, they should be the result of the previous operation add_one
    pipeline10 = add(1,2) >> add_one >> identity(_)
    pipeline11 = add(1,2) >> add_one(1) >> identity(_)

    # Validation
    pipeline.validate()
    # validate all pipelines
    for i, p in enumerate([pipeline, pipeline2, pipeline3, pipeline4, pipeline5, pipeline6, pipeline7]):
        p.validate()
        print(f"Pipeline {i} built and validated: {p}")

    async def main():
        # execute pipelines
        result1 = await pipeline.execute(a=1, b=2)
        res = result1.default_value(None)
        print(result1)
        assert res == 4, f"expect (1 + 2) + 1 = 4, got {res}"
        
        result2 = await pipeline2.execute(a=1, b=2)
        res = result2.default_value(None)
        print(result2)
        assert res == 3, f"expect (1 + 2) = 3, then add_one(2) is ignored because it's bound, got {res}"
        
        result3 = await pipeline3.execute(a=1, b=2)
        res = result3.default_value(None)
        print(result3)
        assert res == 4, f"expect (1 + 2) + 1 = 4, got {res}"
        
        result4 = await pipeline4.execute(a=1, b=2)
        res = result4.default_value(None)
        print(result4)
        assert res == 9, f"expect (1 + 2) * 3 = 9, got {res}"
        
        result5 = await pipeline5.execute(a=1, b=2)
        res = result5.default_value(None)
        print(result5)
        assert res == 9, f"expect 3 * (1 + 2) = 9, got {res}"
        
        result6 = await pipeline6.execute(a=1, b=2)
        res = result6.default_value(None)
        print(result6)
        assert res == 2, f"expect 1 * 2 = 2, ignoring result of add, got {res}"
        
        result7 = await pipeline7.execute(a=1, b=2)
        print(result7)
        res = result7.default_value(None)
        assert res == 3, f"expect identity(1 + 2) = 3, got {res}"

        result8 = await pipeline8.execute()
        res = result8.default_value(None)
        assert res == 4, f"expect add(1,2) + 1 = 4, got {res}"

        result9 = await pipeline9.execute()
        res = result9.default_value(None)
        assert res == 2, f"expect 1 + 1, ignoring add(1,2), got {res}"

        result10 = await pipeline10.execute()
        res = result10.default_value(None)
        assert res == 4, f"expect add(1,2) + 1 = 4, got {res}"

        result11 = await pipeline11.execute()
        res = result11.default_value(None)
        assert res == 2, f"expect 1 + 1 = 2, ignoring add(1,2), got {res}"
        
    print("--------------------------------")
    print("testing execution, binding, and placeholders")
    print("--------------------------------")
    asyncio.run(main())


    async def test_execution():
        @operation
        async def add(a, b):
            return a + b
            
        @operation
        async def add_one(a):
            return a + 1

        pipeline = add >> add_one
        pipeline.validate()
        print(pipeline.dot_notation())
        # test the various ways to execute the pipeline
        result = await pipeline.execute(a=1, b=2)
        print(result)
        assert result.is_ok()
        assert result.default_value(None) == 4

        result2 = await pipeline(1, 2).execute()
        print(result2)
        assert result2.is_ok()
        assert result2.default_value(None) == 4


        result3 = await pipeline(1, 2).execute()
        print(result3)
        assert result3.is_ok()
        assert result3.default_value(None) == 4

        result4 = await pipeline(1, b=2).execute()
        print(result4)
        assert result4.is_ok()
        assert result4.default_value(None) == 4

        result5 = await pipeline.execute(1, b=2)
        print(result5)
        assert result5.is_ok()
        assert result5.default_value(None) == 4


    print("--------------------------------")
    print("testing execution and calling")
    print("--------------------------------")
    asyncio.run(test_execution())
