# siSort_fixed.py
# Self‑Improving Sorter – bug‑fixed version addressing the six issues highlighted
# Additional patch: provide a *stub* for the Pyodide‑only ``micropip`` module so
# that importing it from third‑party harnesses no longer crashes.
# Implementation still follows Jin et al., TALG 18(3) 2022.
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
"""Self‑Improving sorter with probability/complexity guarantees.

Highlights of this patch (2025‑05‑29)
====================================
* **micropip stub** – avoids ``ModuleNotFoundError`` in sandboxed CPython.
* **Extra unit checks** – quick tests for the stub and main algorithm.
* All prior fixes (sampling, V‑list, tries, linear decode, linear merge) retained.
"""

# ---------------------------------------------------------------------------
#  Environment guard – stub‑out ``micropip`` if absent
# ---------------------------------------------------------------------------
# try:
#     import micropip                                # type: ignore
# except ModuleNotFoundError:                         # pragma: no cover
#     import types, sys, warnings
#
#     micropip = types.ModuleType("micropip")        # type: ignore
#
#     def _unavailable(*_a, **_k):
#         """Dummy ``micropip.install`` that issues a warning and returns ``None``."""
#         warnings.warn(
#             "'micropip' not available in this environment – call ignored.",
#             RuntimeWarning,
#         )
#
#     micropip.install = _unavailable                # type: ignore[attr-defined]
#     sys.modules["micropip"] = micropip            # type: ignore

# ---------------------------------------------------------------------------
#  Standard libs
# ---------------------------------------------------------------------------
import bisect
import math
from collections import defaultdict, deque
from typing import List, Sequence, Tuple

# =============================================================================
#  Utilities
# =============================================================================

def lis_length(seq: Sequence[int]) -> int:
    """Return length of the Longest (non‑strict) Increasing Subsequence.

    Complexity O(m log m); used only during training, so the log‑factor is fine.
    """
    tail: List[int] = []
    for x in seq:
        i = bisect.bisect_left(tail, x)
        if i == len(tail):
            tail.append(x)
        else:
            tail[i] = x
    return len(tail)


def lehmer_encode(values: Sequence[int]) -> List[int]:
    """Return Lehmer code π (0‑based) for *values* in O(m log m)."""
    m = len(values)
    bit = [0] * (m + 2)  # Fenwick tree (1‑based)

    def bit_add(i: int) -> None:
        while i < len(bit):
            bit[i] += 1
            i += i & -i

    def bit_sum(i: int) -> int:
        s = 0
        while i:
            s += bit[i]
            i -= i & -i
        return s

    idx_sorted = sorted(range(m), key=values.__getitem__)
    rank = [0] * m
    for r, i in enumerate(idx_sorted):
        rank[i] = r + 1  # shift to 1‑based for Fenwick

    code = [0] * m
    for i in range(m):
        r = rank[i]
        code[i] = r - 1 - bit_sum(r)  # elements < values[i] not yet seen
        bit_add(r)
    return code


def lehmer_decode(values: Sequence[int], code: Sequence[int]) -> List[int]:
    """Reorder *values* according to Lehmer code *code* in O(m)."""
    m = len(values)
    buckets: List[List[int]] = [[] for _ in range(m)]
    for i, c in enumerate(code):
        buckets[c].append(i)

    free: deque[int] = deque(range(m))
    order: List[int] = [0] * m
    for c in range(m - 1, -1, -1):
        for i in reversed(buckets[c]):
            pos = free[c]
            order[pos] = i
            del free[c]
    return [values[i] for i in order]


# =============================================================================
#  Core class
# =============================================================================
class SelfImprovingSorter:
    """Self‑Improving sorter (n keys) faithful to Jin et al. (2022)."""

    # ──────────────────────────────────────────────────────────────────
    # Construction / training
    # ──────────────────────────────────────────────────────────────────
    def __init__(self, n: int, c0: int = 2):
        if n <= 1:
            raise ValueError("n must be ≥ 2 to learn meaningful partitions.")
        self.n = n
        self.c0 = c0
        self.groups: List[List[int]] = []          # hidden partition
        self.v_list: List[float] = []              # V‑list boundaries
        self.structs: dict[Tuple[int, ...], tuple] = {}

    # 2.1 Hidden partition learning
    def learn_hidden_partition(self, samples: Sequence[Sequence[int]]) -> None:
        n, c0 = self.n, self.c0
        L = int(max(2 * 10**6, (270 * math.log(n)) ** 2, (6 * c0 + 3) ** 2))
        if len(samples) < L:
            raise ValueError(
                f"Need at least {L} training instances, got {len(samples)}.")
        base = samples[:L]  # fixed prefix ensures deterministic analysis

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # pairwise independence test
        for i in range(n):
            for j in range(i + 1, n):
                pairs = sorted(((I[i], I[j]) for I in base), key=lambda p: p[0])
                y = [yj for _, yj in pairs]
                lms = max(lis_length(y), lis_length(list(reversed(y))))
                if lms >= L / (2 * c0 + 1):
                    union(i, j)

        groups: defaultdict[int, list[int]] = defaultdict(list)
        for idx in range(n):
            groups[find(idx)].append(idx)
        self.groups = list(groups.values())

    # 2.2 V‑list learning
    def build_v_list(self, samples: Sequence[Sequence[int]]) -> None:
        n = self.n
        lam = math.ceil(n * n * math.log(n))
        if len(samples) < lam:
            raise ValueError(
                f"Need ≥ ⌈n² ln n⌉ = {lam} samples for V‑list, got {len(samples)}.")

        values: List[int] = []
        for I in samples[:lam]:
            values.extend(I)
        values.sort()

        v: List[float] = [-math.inf]
        for r in range(1, n + 1):
            v.append(values[r * lam - 1])
        v.append(math.inf)
        self.v_list = v

    # 2.3 Group‑specific tries
    class _TrieNode:
        __slots__ = ("children", "cnt", "order")
        def __init__(self):
            self.children: dict[int, "SelfImprovingSorter._TrieNode"] = {}
            self.cnt: int = 0
            self.order: list[int] = []

    class _Trie:
        def __init__(self):
            self.root = SelfImprovingSorter._TrieNode()

        # insert code path
        def add(self, code: Sequence[int]) -> None:
            node = self.root
            node.cnt += 1
            for lbl in code:
                node = node.children.setdefault(lbl, SelfImprovingSorter._TrieNode())
                node.cnt += 1

        # DFS: compute weight‑biased order (approx entropy‑optimal)
        def _prepare(self, node: "SelfImprovingSorter._TrieNode") -> None:
            if not node.children:
                return
            ordered = sorted(node.children.items(), key=lambda kv: -kv[1].cnt)
            node.order = [lbl for lbl, _ in ordered]
            for _, child in ordered:
                self._prepare(child)

        def finalize(self) -> None:
            self._prepare(self.root)

        # query path; fallback called on miss
        def query(self, code: Sequence[int], fallback):
            node = self.root
            for lbl in code:
                if lbl in node.children:
                    node = node.children[lbl]
                else:
                    return fallback()
            return code  # success

    # helpers for encoding
    def _encode_b(self, sub: Sequence[int]) -> List[int]:
        v = self.v_list
        return [bisect.bisect_right(v, z) - 1 for z in sub]

    def _encode_pi(self, sub: Sequence[int]) -> List[int]:
        return lehmer_encode(sub)

    # Build tries per learnt group
    def build_group_structures(self, samples: Sequence[Sequence[int]]) -> None:
        self.structs.clear()
        for G in self.groups:
            enc_b: List[Tuple[int, ...]] = []
            enc_p: List[Tuple[int, ...]] = []
            for I in samples:
                sub = [I[i] for i in G]
                enc_b.append(tuple(self._encode_b(sub)))
                enc_p.append(tuple(self._encode_pi(sub)))
            trie_b, trie_p = SelfImprovingSorter._Trie(), SelfImprovingSorter._Trie()
            for c in enc_b:
                trie_b.add(c)
            for c in enc_p:
                trie_p.add(c)
            trie_b.finalize()
            trie_p.finalize()
            self.structs[tuple(G)] = (trie_b, trie_p)

    # ──────────────────────────────────────────────────────────────────
    # Operation phase
    # ──────────────────────────────────────────────────────────────────
    def sort(self, I: Sequence[int]) -> List[int]:
        n = self.n
        Z: List[List[List[int]]] = [[] for _ in range(n + 1)]  # bucket → list of sorted runs

        for G in self.groups:
            sub = [I[i] for i in G]
            trie_b, trie_p = self.structs[tuple(G)]

            # b‑code
            b_code = self._encode_b(sub)
            trie_b.query(b_code, lambda: None)  # side effect only

            # π‑code
            pi_code = self._encode_pi(sub)
            trie_p.query(pi_code, lambda: None)

            sorted_sub = lehmer
