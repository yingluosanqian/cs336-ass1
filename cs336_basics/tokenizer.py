
import os
import timeit
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator

import regex as re

from .load_data import load_text_from_file
from .data_structure import MaxHeap


def pre_tokenize(
    chunks: Iterable[str],
    special_tokens: list[str],
    token_pattern=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
) -> Counter:
    special_pattern = re.compile('|'.join(map(re.escape, special_tokens)))
    token_pattern = re.compile(token_pattern, re.UNICODE)
    return Counter(
        match.group().encode("utf-8")
        for chunk in chunks
        for text in special_pattern.split(chunk) if text
        for match in token_pattern.finditer(text) if match.group()
    )


class IndexManager:
    def __init__(self):
        self.total = 0
        self.indices: set[tuple[int, int]] = set()

    def add(self, p_id, b_id, num):
        self.total += num
        self.indices.add((p_id, b_id))
        return self.total

    def remove(self, p_id, b_id, num):
        self.total -= num
        self.indices.remove((p_id, b_id))
        return self.total


class PairManager():
    def __init__(self, token_counter: Counter):
        self._pair_counter: dict[tuple[bytes, bytes],
                                 IndexManager] = defaultdict(IndexManager)
        self._pairs: tuple[tuple[tuple[bytes, ...], int]] = ()
        self._next: list[list[int]] = []
        self._prev: list[list[int]] = []
        self._max_heap: MaxHeap = MaxHeap()
        self._initialize(token_counter)

    def _initialize(self, token_counter: Counter):
        self._pairs = tuple(
            (tuple(bytes([byte]) for byte in token), count)
            for token, count in token_counter.items()
        )
        self._next = [list(range(1, len(pair) + 1)) for pair, _ in self._pairs]
        self._prev = [list(range(-1, len(pair) - 1))
                      for pair, _ in self._pairs]
        self._initialize_heap()

    def _initialize_heap(self):
        for p_id, (tokens, _) in enumerate(self._pairs):
            for b_id, (byte1, byte2) in enumerate(zip(tokens[:-1], tokens[1:])):
                self._add((byte1, byte2), p_id, b_id)

    def _add(self, bytes_pair: tuple[bytes, bytes], p_id: int, b_id: int):
        total = self._pair_counter[bytes_pair].add(
            p_id, b_id, self._pairs[p_id][1])
        self._max_heap.push((total, bytes_pair))

    def _remove(self, bytes_pair: tuple[bytes, bytes], p_id: int, b_id: int):
        total = self._pair_counter[bytes_pair].remove(
            p_id, b_id, self._pairs[p_id][1])
        self._max_heap.push((total, bytes_pair))

    def update_index(self, bytes_pair: tuple[bytes, bytes]):
        cur_bytes_0, cur_bytes_1 = bytes_pair
        indices = self._pair_counter[bytes_pair].indices.copy()
        merged_bytes = cur_bytes_0 + cur_bytes_1
        # print("merged_bytes:", merged_bytes)
        last_p_id, last_b_id = -1, -1
        for p_id, b_id in sorted(indices):
            # Avoid overlap
            if p_id == last_p_id and self._next[p_id][last_b_id] > b_id:
                continue
            last_p_id, last_b_id = p_id, b_id
            line = self._pairs[p_id][0]
            # Example: a, (b, c), d
            # Remove: (a, b), (b, c), (c, d); Add: (a, bc), (bc, d)
            # 1. Remove (b, c)
            self._remove((cur_bytes_0, cur_bytes_1), p_id, b_id)
            # 2. Remove (c, d), Add(bc, d)
            next_1 = self._next[p_id][b_id]
            next_2 = self._next[p_id][next_1]
            if next_2 < len(self._next[p_id]):
                next_3 = self._next[p_id][next_2]
                next_bytes = b"".join(line[next_2: next_3])
                self._remove(
                    (cur_bytes_1, next_bytes), p_id, next_1)
                self._add(
                    (merged_bytes, next_bytes), p_id, b_id)
                self._prev[p_id][next_2] = b_id
            self._next[p_id][b_id] = next_2
            # 3. Remove (a, b), Add (a, bc)
            prev_1 = self._prev[p_id][b_id]
            if prev_1 >= 0:
                prev_bytes = b"".join(line[prev_1: b_id])
                self._remove(
                    (prev_bytes, cur_bytes_0), p_id, prev_1)
                self._add(
                    (prev_bytes, merged_bytes), p_id, prev_1)

    def get_max(self) -> tuple[bytes, bytes]:
        while not self._max_heap.is_empty():
            record_total, bytes_pair = self._max_heap.pop()
            if record_total == self._pair_counter[bytes_pair].total:
                # print("record_total:", record_total, "bytes_pair:", bytes_pair)
                return bytes_pair
        raise ValueError("No valid item found")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {
        **{i: token.encode("utf-8") for i, token in enumerate(special_tokens)},
        **{i + len(special_tokens): bytes([i]) for i in range(256)}
    }
    merges: list[tuple[bytes, bytes], int] = []
    chunks = load_text_from_file(input_path)
    token_counter = pre_tokenize(chunks, special_tokens)
    pair_manager = PairManager(token_counter)

    while (len(vocab) < vocab_size):
        # Select the most frequent pairs with keyword: 1) frequency; 2) lexicographical greater.
        bytes_pair = pair_manager.get_max()
        # Merge and update indices
        pair_manager.update_index(bytes_pair)
        # Update Vocab
        next_id = len(vocab)
        vocab[next_id] = bytes_pair[0] + bytes_pair[1]
        merges.append(bytes_pair)
    return vocab, merges
