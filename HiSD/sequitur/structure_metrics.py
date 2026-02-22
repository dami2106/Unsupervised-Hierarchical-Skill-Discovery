from collections import defaultdict
from typing import Dict, List, Union, Tuple
import hashlib

# Define tree node type
TreeNode = Dict[str, Union[int, str, List["TreeNode"]]]

def compute_hierarchy_depth(tree: TreeNode) -> int:
    if "children" not in tree:
        return 1
    return 1 + max(compute_hierarchy_depth(child) for child in tree["children"])

def count_nodes(tree: TreeNode) -> int:
    if "children" not in tree:
        return 1
    return 1 + sum(count_nodes(child) for child in tree["children"])

def compute_branching_factor(tree: TreeNode) -> Tuple[float, int]:
    if "children" not in tree:
        return 0.0, 0
    
    total_branches = 0
    internal_nodes = 0
    max_branching = 0

    def traverse(node):
        nonlocal total_branches, internal_nodes, max_branching
        if "children" in node:
            num_children = len(node["children"])
            total_branches += num_children
            max_branching = max(max_branching, num_children)
            internal_nodes += 1
            for child in node["children"]:
                traverse(child)

    traverse(tree)
    avg_branching = total_branches / internal_nodes if internal_nodes > 0 else 0.0
    return avg_branching, max_branching

def compute_subtree_hash(node: TreeNode) -> str:
    """Hashes the structure and content of the subtree for reuse/modularity metrics."""
    if "children" not in node:
        return f"leaf:{node['symbol']}"
    child_hashes = tuple(compute_subtree_hash(child) for child in node["children"])
    return f"prod:{node['production']}|{hash(child_hashes)}"

def compute_reuse_score(tree: TreeNode) -> float:
    """Reuse score = unique_subtrees / total_subtree_occurrences"""
    counter = defaultdict(int)

    def traverse(node):
        h = compute_subtree_hash(node)
        counter[h] += 1
        if "children" in node:
            for child in node["children"]:
                traverse(child)

    traverse(tree)
    total = sum(counter.values())
    unique = len(counter)
    return unique / total if total > 0 else 1.0

def compute_modularity_score(tree: TreeNode) -> float:
    """Approximate modularity by measuring how many top-level subtrees can stand independently."""
    if "children" not in tree:
        return 1.0
    root_children = tree["children"]
    independent_count = 0
    for child in root_children:
        reused = compute_reuse_score(child)
        if reused > 0.9:  # high reuse means it's likely a standalone, reusable module
            independent_count += 1
    return independent_count / len(root_children) if root_children else 1.0

def compute_structure_metrics(tree: TreeNode) -> Dict[str, Union[int, float]]:
    """Compute various structure metrics for the given tree."""
    depth = compute_hierarchy_depth(tree)
    size = count_nodes(tree)
    avg_branching, max_branching = compute_branching_factor(tree)
    reuse = compute_reuse_score(tree)
    modularity = compute_modularity_score(tree)

    return {
        "depth": depth,
        "size": size,
        "avg_branching": avg_branching,
        "max_branching": max_branching,
        "reuse": reuse,
        "modularity": modularity
    }

# Example use with a given tree (replace with actual input as needed)
example_tree = {
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

print(compute_structure_metrics(example_tree))