import zss  # Zhang-Shasha algorithm library for Tree Edit Distance
from typing import Set
from collections import defaultdict
from typing import Dict, List, Union, Tuple
import hashlib


# Define tree node type
TreeNode = Dict[str, Union[int, str, List["TreeNode"]]]

def compute_subtree_hash(node: TreeNode) -> str:
    """Hashes the structure and content of the subtree for reuse/modularity metrics."""
    if "children" not in node:
        return f"leaf:{node['symbol']}"
    child_hashes = tuple(compute_subtree_hash(child) for child in node["children"])
    return f"prod:{node['production']}|{hash(child_hashes)}"


# Define zss-compatible node class
class ZssNode(zss.Node):
    def __init__(self, label, children=None):
        super().__init__(label)
        if children:
            for child in children:
                self.addkid(child)

def convert_to_zss_tree(tree: TreeNode) -> ZssNode:
    if "symbol" in tree:
        return ZssNode(tree["symbol"])
    label = f"prod:{tree['production']}"
    children = [convert_to_zss_tree(child) for child in tree["children"]]
    return ZssNode(label, children)

def compute_tree_edit_distance(tree1: TreeNode, tree2: TreeNode) -> int:
    t1 = convert_to_zss_tree(tree1)
    t2 = convert_to_zss_tree(tree2)
    return zss.simple_distance(t1, t2)

def extract_productions(tree: TreeNode) -> Set[str]:
    """Extracts productions in the form 'parent -> [child1, child2, ...]' as strings"""
    productions = set()
    if "children" in tree:
        child_labels = []
        for child in tree["children"]:
            if "symbol" in child:
                child_labels.append(f"leaf:{child['symbol']}")
            else:
                child_labels.append(f"prod:{child['production']}")
        productions.add(f"prod:{tree['production']} -> {tuple(child_labels)}")
        for child in tree["children"]:
            productions.update(extract_productions(child))
    return productions

def compute_jaccard_index(tree1: TreeNode, tree2: TreeNode) -> float:
    prods1 = extract_productions(tree1)
    prods2 = extract_productions(tree2)
    intersection = len(prods1 & prods2)
    union = len(prods1 | prods2)
    return intersection / union if union > 0 else 1.0

def extract_all_subtree_hashes(tree: TreeNode) -> Set[str]:
    """Returns a set of all subtree hashes from the tree"""
    hashes = set()

    def traverse(node):
        h = compute_subtree_hash(node)
        hashes.add(h)
        if "children" in node:
            for child in node["children"]:
                traverse(child)

    traverse(tree)
    return hashes

def compute_subtree_overlap(tree1: TreeNode, tree2: TreeNode) -> float:
    subtrees1 = extract_all_subtree_hashes(tree1)
    subtrees2 = extract_all_subtree_hashes(tree2)
    intersection = len(subtrees1 & subtrees2)
    total = len(subtrees1)
    return intersection / total if total > 0 else 1.0


def compute_similarity_metrics(tree1: TreeNode, tree2: TreeNode) -> Tuple[float, float, float]:
    """
    Computes all similarity metrics between two trees:
    - Tree Edit Distance
    - Jaccard Index
    - Subtree Overlap
    """
    ted = compute_tree_edit_distance(tree1, tree2)
    jaccard = compute_jaccard_index(tree1, tree2)
    subtree_overlap = compute_subtree_overlap(tree1, tree2)
    return {
        "tree_edit_distance": ted,
        "jaccard_index": jaccard,
        "subtree_overlap": subtree_overlap
    }

# Define a second example tree to compare with the first one
predicted_tree = {
  "production": 0,
  "children": [
    {
      "production": 2,
      "children": [
        {
          "production": 6,
          "children": [
            {
              "symbol": "1"
            },
            {
              "symbol": "0"
            }
          ]
        },
        {
          "symbol": "2"
        }
      ]
    },
    {
      "production": 3,
      "children": [
        {
          "symbol": "1"
        },
        {
          "production": 5,
          "children": [
            {
              "symbol": "3"
            },
            {
              "symbol": "4"
            }
          ]
        }
      ]
    }
  ]
}

gt_tree = {
  "production": 0,
  "children": [
    {
      "production": 1,
      "children": [
        {
          "production": 4,
          "children": [
            {
              "production": 6,
              "children": [
                {
                  "symbol": "1"
                },
                {
                  "symbol": "0"
                }
              ]
            },
            {
              "symbol": "1"
            },
            {
              "symbol": "2"
            }
          ]
        },
        {
          "symbol": "3"
        }
      ]
    },
    {
      "symbol": "1"
    },
    {
      "symbol": "4"
    }
  ]
}


print(compute_similarity_metrics(predicted_tree, gt_tree))
