
import heapq
from typing import Generic, TypeVar, Optional

T = TypeVar('T')


class MaxHeapElem(Generic[T]):
    __slots__ = ('value',)

    def __init__(self, value: T):
        self.value = value

    def __lt__(self, other: 'MaxHeapElem[T]') -> bool:
        return self.value > other.value  # type: ignore

    def __eq__(self, other: 'MaxHeapElem[T]') -> bool:
        return self.value == other.value

    def __repr__(self):
        return f'MaxHeapElem({self.value})'


class MaxHeap(Generic[T]):
    def __init__(self):
        self._elements = []

    def push(self, elem: tuple[int, tuple[bytes, bytes]]):
        heapq.heappush(self._elements, MaxHeapElem(elem))

    def pop(self) -> tuple[int, tuple[bytes, bytes]]:
        if self.is_empty():
            raise IndexError("pop from an empty MaxHeap")
        return heapq.heappop(self._elements).value

    def is_empty(self):
        return not self._elements
