import uuid
from typing import List, Optional

class HistoryNode:
    """
    DAG 中的一个节点，存储某个版本的 Face 对象，以及是否为当前叶节点。
    """
    def __init__(self, face_snapshot, operation: str, parents: Optional[List['HistoryNode']] = None):
        self.node_id = uuid.uuid4()
        # 直接存储对 Face 对象的引用，避免深拷贝开销
        self.face = face_snapshot
        self.operation = operation  # 例如："initial", "split", "flip", "merge" 等
        self.parents = parents or []
        self.children: List[HistoryNode] = []
        # 标记当前节点是否为叶节点（未被淘汰）
        self.is_leaf = True
        # 将自己挂到父节点，并将父节点标记为内部节点
        for p in self.parents:
            p.children.append(self)
            p.is_leaf = False

    def __repr__(self):
        pid = [str(p.node_id)[:8] for p in self.parents]
        flag = 'leaf' if self.is_leaf else 'internal'
        return f"<HistoryNode {str(self.node_id)[:8]} op={self.operation} {flag} parents={pid}>"

class HistoryDAG:
    """
    管理所有 HistoryNode，不删除旧节点；被淘汰的旧节点标记为 internal，
    新版本节点默认是 leaf，查询时从 internal 往下查 leaf。
    """
    def __init__(self):
        # key: node_id, value: HistoryNode
        self.nodes = {}

    def add_node(self, face, operation: str, parents: Optional[List[HistoryNode]] = None) -> HistoryNode:
        """添加新版本节点"""
        node = HistoryNode(face, operation, parents)
        self.nodes[node.node_id] = node
        return node

    def find_latest(self, face_id: uuid.UUID) -> Optional[HistoryNode]:
        """
        根据 Face.id 找到最近一次对应的版本节点（按插入顺序逆序遍历）。
        """
        for node in reversed(list(self.nodes.values())):
            if getattr(node.face, 'id', None) == face_id:
                return node
        return None

    def get_current_leaves(self, face_id: uuid.UUID) -> List[HistoryNode]:
        """
        若给定 face_id 已经被分裂或翻边等淘汰，返回所有其叶子子孙节点；
        否则返回自身对应的最新叶节点。
        """
        start = self.find_latest(face_id)
        if not start:
            return []
        leaves: List[HistoryNode] = []
        def dfs(n: HistoryNode):
            if n.is_leaf:
                leaves.append(n)
            else:
                for c in n.children:
                    dfs(c)
        dfs(start)
        return leaves

    def record_split(self, parent_face, new_faces: List, operation: str = "split") -> List[HistoryNode]:
        """
        记录一次三角形拆分：将对应 parent_face 保存的节点标记 internal，
        并为每个 new_face 创建 leaf 子节点。
        """
        parent_node = self.find_latest(parent_face.id)
        children: List[HistoryNode] = []
        for f in new_faces:
            child = self.add_node(f, operation, parents=[parent_node] if parent_node else None)
            children.append(child)
        return children

    def record_flip(self, face1, face2):
        """
        记录一次翻边操作：对两个受影响面分别打上 new node。
        """
        n1 = self.find_latest(face1.id)
        n2 = self.find_latest(face2.id)
        if n1:
            self.add_node(face1, "flip", parents=[n1])
        if n2:
            self.add_node(face2, "flip", parents=[n2])
