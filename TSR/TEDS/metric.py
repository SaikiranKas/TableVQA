
from bs4 import BeautifulSoup
from lxml import etree
import zss

class TEDS:
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def evaluate(self, pred, gt, is_structure=True):
        pred_tree = self._html2tree(pred, is_structure)
        gt_tree = self._html2tree(gt, is_structure)

        if pred_tree is None or gt_tree is None:
            return 0.0

        dist = zss.simple_distance(
            pred_tree, gt_tree,
            get_children=lambda n: n.children,
            get_label=lambda n: n.label
        )
        max_dist = zss.simple_distance(
            gt_tree, None,
            get_children=lambda n: n.children,
            get_label=lambda n: n.label
        )
        score = 1 - (dist / max_dist) if max_dist != 0 else 1.0
        return score

    def _html2tree(self, html, structure_only=True):
        try:
            soup = BeautifulSoup(html, "lxml")
            root = soup.find("table")
            if not root:
                return None
            element = self._build_tree(root, structure_only)
            return element
        except Exception:
            return None

    def _build_tree(self, node, structure_only):
        label = node.name
        if not structure_only:
            if node.name == 'td':
                label += ':' + (node.get_text().strip() or '')
        tree_node = TreeNode(label)

        for child in node.children:
            if isinstance(child, str):
                continue
            subtree = self._build_tree(child, structure_only)
            if subtree:
                tree_node.add_child(subtree)

        return tree_node


class TreeNode:
    def __init__(self, label):
        self.label = label
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def get_children(self):
        return self.children

    def get_label(self):
        return self.label
