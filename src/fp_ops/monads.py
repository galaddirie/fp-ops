import asyncio
from abc import abstractmethod
from typing import Generic, Protocol, runtime_checkable, TypeVar, Callable

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")

@runtime_checkable
class Monad(Protocol):
    @staticmethod
    @abstractmethod
    def rtn(value: T) -> 'Monad[T]':
        raise NotImplementedError

    @abstractmethod
    def then(self, fn: Callable[[T], 'Monad[S]']) -> 'Monad[S]':
        raise NotImplementedError



class Cont(Generic[T, R]):
    def __init__(self, cps: Callable[[Callable[[T], R]], R]):
        self.__cps = cps # fn: ('a -> 'r) -> 'r

    @staticmethod
    def rtn(value: T) -> 'Cont[T, R]':
        """Return a value in the Cont monad context (unit/return)."""
        return Cont(lambda cont: cont(value))

    def run(self, cont: Callable[[T], R]) -> R:
        self.__cps(cont)

    def then(self, fn: Callable[[T], 'Cont[S, R]']) -> 'Cont[S, R]':
        """Bind operation apply a function that returns a Cont."""
        # Cont <| fun c -> run cont (fun a -> run (fn a) c )
        return Cont(lambda c: self.run(lambda a: fn(a).run(c)))
    
    def map(self, fn: Callable[[T], S]) -> 'Cont[S, R]':
        """Map operation: apply a function to the value."""
        return self.then(lambda a: Cont.rtn(fn(a)))
    
    def __await__(self):
        """Make Cont awaitable in async functions."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        def done(value):
            future.set_result(value)
        self.run(done)
        return iter(future)


class Maybe(Generic[T]):
    def __init__(self, value: T | None):
        self.__value = value

    @staticmethod
    def rtn(value: T) -> 'Maybe[T]':
        return Maybe(value)

    def then(self, fn: Callable[[T], 'Maybe[S]']) -> 'Maybe[S]':
        if self.__value is None:
            return Maybe(None)
        return fn(self.__value)
