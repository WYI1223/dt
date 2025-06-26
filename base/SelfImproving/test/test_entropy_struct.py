# tests/test_entropy_struct.py

import unittest

from ..entropy_struct import B_Pi_Structure

class TestEntropyStruct(unittest.TestCase):

    def test_simple_trie_bst(self):
        # 假设我们有两个训练对：(in, out)
        # 输出字符串出现在 Trie 中：["abc", "abd", "acd"]
        training = [
            ("I1", list("abc")),
            ("I2", list("abd")),
            ("I3", list("acd"))
        ]
        bp = B_Pi_Structure()
        bp.build(training)

        # bp.draw_graphviz()
        # 下面查询时，只需要传 output 的前缀（因为 B_Pi 用的是完全 match）
        # 实际使用时 input_string 是点集编码，而我们这里让 input_string = outp 来模拟。
        self.assertEqual(bp.query(list("abc")), list("abc"))
        self.assertEqual(bp.query(list("abd")), list("abd"))
        self.assertEqual(bp.query(list("acd")), list("acd"))

        # 尝试一个训练集中未出现过的完整 output 路径，应该走到 fallback
        with self.assertRaises(KeyError):
            bp.query(list("abe"))

if __name__ == "__main__":
    unittest.main()

def test_simple_trie_bst():
    # 假设我们有两个训练对：(in, out)
    # 输出字符串出现在 Trie 中：["abc", "abd", "acd"]
    training = [
        ("I1", list("abc")),
        ("I2", list("abd")),
        ("I3", list("acd"))
    ]
    bp = B_Pi_Structure()
    bp.build(training)

    # bp.draw_graphviz()

if __name__ == '__main__':
    test_simple_trie_bst()