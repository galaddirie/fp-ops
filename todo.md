in readme show how we can use the placeholder to create a pipeline 
and the alternative with bind + lambda

example:

```python
from silk.placeholder import _
add(1, 2) >> multiply(_, 2)

# is equivalent to
add(1, 2) >> (lambda x: multiply(x, 2))

add(1, 2).bind(lambda x: multiply(x, 2))
```



