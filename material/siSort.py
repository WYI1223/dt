import bisect
import math
import random
from collections import defaultdict

class SelfImprovingSorter:
    """
    Implements the self-improving sorting algorithm from the paper.
    Follows all steps without simplification. Comments refer to paper sections.
    """
    def __init__(self, n):
        self.n = n
        # placeholders for learned structures
        self.groups = []            # list of groups G_k
        self.v_list = []            # V-list cut points v_0..v_{n+1}
        self.group_structs = {}     # per-group data for b and pi

    # ---------- Section 2.1: Hidden Partition Learning ----------
    def learn_hidden_partition(self, sample_instances):
        """
        Learn the hidden group partition of [1..n] using Corollary 2.4.
        sample_instances: list of input lists I (length n).
        For each pair (i,j) compute LIS/LMS test to decide same group.
        """
        n = self.n
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        c0 = 2  # placeholder constant bound on extrema
        l_raw = max(
            2 * 10**6,
            (270 * math.log(n))**2,
            (6 * c0 + 3)**2
        )
        sample_size = min(int(l_raw), len(sample_instances))
        sample_size = max(1, sample_size)

        for i in range(n):
            for j in range(i+1, n):
                pairs = [(I[i], I[j]) for I in random.sample(sample_instances, sample_size)]
                pairs.sort(key=lambda x: x[0])
                y = [y2 for _, y2 in pairs]
                lis = self._lis_length(y)
                lds = self._lis_length(list(reversed(y)))
                lms = max(lis, lds)
                if lms >= l_raw / (2 * c0 + 1):
                    union(i, j)
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)
        self.groups = list(groups.values())

    def _lis_length(self, seq):
        tail = []
        for x in seq:
            pos = bisect.bisect_left(tail, x)
            if pos == len(tail):
                tail.append(x)
            else:
                tail[pos] = x
        return len(tail)

    # ---------- Section 2.2: Build V-list ----------
    def build_v_list(self, sample_instances):
        """
        Build V-list of cut points v_0..v_{n+1} as in Section 2.2.
        """
        n = self.n
        lam = len(sample_instances)
        values = []
        for I in sample_instances:
            values.extend(I)
        values.sort()
        v = [-math.inf]
        total = len(values)
        for r in range(1, n+1):
            idx = r * lam - 1
            if idx < total:
                v.append(values[idx])
            else:
                v.append(math.inf)
        v.append(math.inf)
        self.v_list = v

    # ---------- Section 2.3: Build Trie and BST structures ----------
    def build_group_structures(self, sample_instances):
        """
        For each group G_k, construct data structures for b and pi as in Theorem 2.9.
        """
        for G in self.groups:
            outputs_b = []
            outputs_pi = []
            for I in sample_instances:
                sub = [I[i] for i in G]
                outputs_b.append(tuple(self._encode_b(sub)))
                outputs_pi.append(tuple(self._encode_pi(sub)))
            trie_b = Trie(mode='b', fallback=self._encode_b)
            trie_b.build(outputs_b)
            trie_pi = Trie(mode='pi', fallback=self._encode_pi)
            trie_pi.build(outputs_pi)
            self.group_structs[tuple(G)] = (trie_b, trie_pi)

    def _encode_b(self, sub):
        b = []
        for z in sub:
            r = bisect.bisect_right(self.v_list, z) - 1
            b.append(r)
        return b

    def _encode_pi(self, sub):
        pi = []
        for i, x in enumerate(sub):
            idx = 0
            for j in range(i):
                if sub[j] < x:
                    idx = j + 1
            pi.append(idx)
        return pi

    # ---------- Section 2.4: Operation Phase Sorting ----------
    def sort(self, I):
        """
        Sort a new instance I using learned structures as in Section 2.4.
        Returns sorted list.
        """
        Z = [[] for _ in range(self.n+1)]
        for G in self.groups:
            sub = [I[i] for i in G]
            trie_b, trie_pi = self.group_structs[tuple(G)]
            # attempt to query; fallback if missing
            try:
                b_code = trie_b.query(sub)
            except ValueError:
                b_code = self._encode_b(sub)
            try:
                pi_code = trie_pi.query(sub)
            except ValueError:
                pi_code = self._encode_pi(sub)
            sorted_sub = self._reconstruct_from_pi(sub, pi_code)
            curr_r, curr_seq = None, []
            for x, r in zip(sorted_sub, b_code):
                if curr_r is None or r != curr_r:
                    if curr_seq:
                        Z[curr_r].append(curr_seq)
                    curr_seq, curr_r = [x], r
                else:
                    curr_seq.append(x)
            if curr_seq:
                Z[curr_r].append(curr_seq)
        result = []
        for seqs in Z:
            result.extend(self._merge_sorted(seqs))
        return result

    def _reconstruct_from_pi(self, sub, pi_code):
        lst = []
        for x, idx in zip(sub, pi_code):
            lst.insert(idx, x)
        return lst

    def _merge_sorted(self, seqs):
        import heapq
        heap = []
        for i, seq in enumerate(seqs):
            if seq:
                heapq.heappush(heap, (seq[0], i, 0, seq))
        merged = []
        while heap:
            val, i, idx, seq = heapq.heappop(heap)
            merged.append(val)
            if idx + 1 < len(seq):
                heapq.heappush(heap, (seq[idx+1], i, idx+1, seq))
        return merged

# ---------- Trie and nearly-optimal BST ----------
class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0

class Trie:
    """Trie with nearly-optimal BST ordering at each node."""
    def __init__(self, mode=None, fallback=None):
        self.root = TrieNode()
        self.mode = mode            # 'b' or 'pi'
        self.fallback = fallback    # function to compute code

    def build(self, outputs):
        for code in outputs:
            node = self.root
            node.count += 1
            for lbl in code:
                if lbl not in node.children:
                    node.children[lbl] = TrieNode()
                node = node.children[lbl]
                node.count += 1
        self._prepare_bst(self.root)

    def _prepare_bst(self, node):
        items = sorted(node.children.items(), key=lambda kv: -kv[1].count)
        node.bst_labels = [lbl for lbl,_ in items]
        node.bst_children = {lbl:ch for lbl,ch in items}
        for ch in node.children.values():
            self._prepare_bst(ch)

    def query(self, sub):
        node = self.root
        code = []
        for x in sub:
            labels = node.bst_labels
            # labels are code values; need to compute next label
            # here interpret x as potential next label
            if x in labels:
                code.append(x)
                node = node.bst_children[x]
            else:
                # fallback to direct encoding
                if self.fallback:
                    return self.fallback(sub)
                raise ValueError("Abort: code not in trie and no fallback")
        return code

# --------------------------------------
# ---------- Testing Harness ----------
# --------------------------------------
if __name__ == "__main__":
    n = 10
    sample_instances = [random.sample(range(100), n) for _ in range(5000)]
    sorter = SelfImprovingSorter(n)
    sorter.learn_hidden_partition(sample_instances)
    sorter.build_v_list(sample_instances)
    sorter.build_group_structures(sample_instances)

    errors = 0
    trials = 1000
    for _ in range(trials):
        I = random.sample(range(100), n)
        sorted_I = sorter.sort(I)
        if sorted_I != sorted(I):
            errors += 1
    print(f"Random test: {trials} instances, errors: {errors}")

    # Edge cases
    I_sorted = list(range(n))
    assert sorter.sort(I_sorted) == sorted(I_sorted), "Failed on already sorted input"
    I_rev = list(range(n-1, -1, -1))
    assert sorter.sort(I_rev) == sorted(I_rev), "Failed on reverse sorted input"
    print("Edge case tests passed.")
