in readme show how we can use the placeholder to create a pipeline 
and the alternative with bind + lambda

example:

```python
add(1, 2) >> multiply(_, 2) # 6

# is equivalent to

add(1, 2).bind(lambda x: multiply(x, 2)) # 6
```



