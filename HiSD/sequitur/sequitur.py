from sksequitur import Parser, Grammar, Production, Mark
from graphviz import Digraph
import argparse
from helpers import get_unique_sequence_list, get_all_sequences_list
import json
from pathlib import Path
import os
from structure_metrics import compute_structure_metrics
from similarity_metrics import compute_similarity_metrics
from matching.games import StableMarriage

# --- 3) Build a nested‐dict tree for one sequence‐expansion ---
def build_sequence_tree(grammar, token_list):
    """
    Turn a list of tokens [Prod, 'A', Prod, …] into
    {'production': 0, 'children': [ … ]} recursively.
    """
    node = {"production": 0, "children": []}
    for tok in token_list:
        if isinstance(tok, Production):
            node["children"].append(build_subtree(grammar, tok))
        else:
            node["children"].append({"symbol": tok})
    return node

def build_subtree(grammar, prod):
    """
    Turn a Production(prod) into {'production': n, 'children': […]}.
    Uses the single right‐hand side grammar[prod].
    """
    subtree = {"production": int(prod), "children": []}
    for tok in grammar[prod]:
        if isinstance(tok, Production):
            subtree["children"].append(build_subtree(grammar, tok))
        else:
            subtree["children"].append({"symbol": tok})
    return subtree

def visualize_tree(tree_dict, filename_base, fmt="pdf", root_label=None, label_mapping=None):
    """
    tree_dict: nested dict with keys
      - 'production' → int
      - OR 'symbol' → str
      and optional 'children': [ … ]
    filename_base: e.g. "seq_tree_0"  (no extension)
    fmt: "pdf" (default)
    root_label: if given, use this string as the label for the very top node
    label_mapping: optional dict mapping grammar labels to readable strings
    """
    dot = Digraph(format=fmt)
    dot.attr(rankdir="TB")     # top→bottom
    dot.attr("node", shape="oval", fontsize="12")

    def recurse(node, parent_id=None, is_root=False):
        nid = str(id(node))
        # if this is the root of the entire tree and a custom label was supplied, use it:
        if is_root and root_label is not None:
            label = "Truth"
        else:
            if "production" in node:
                label = f"R{node['production']}"
            else:
                # Use mapping if available, otherwise use original symbol
                label = label_mapping.get(node["symbol"], node["symbol"]).replace("_", " ").title() if label_mapping else node["symbol"]

        dot.node(nid, label)
        if parent_id is not None:
            dot.edge(parent_id, nid)

        for child in node.get("children", []):
            recurse(child, nid)

    # kick off recursion marking the first call as the root
    recurse(tree_dict, parent_id=None, is_root=True)

    # Set DPI to 300 for higher quality PDF
    dot.graph_attr["dpi"] = "300"
    outpath = dot.render(filename_base, cleanup=True)
    # print(f"Written {outpath}")

def build_htn_trees(sequences):
    """
    Build HTN trees for each sequence in the list.
    sequences: list of lists of symbols (not strings)
    """
    # Create a single parser instance
    parser = Parser()

    # Feed each sequence to the parser incrementally
    for seq in sequences:
        # seq is now a list of symbols, not a string
        parser.feed(seq)
        parser.feed([Mark()])

    # Convert the parser output to a grammar
    grammar = Grammar(parser.tree)

    # Print the final grammar
    print(grammar)

    flat_start = grammar[Production(0)]
    seq_expansions = []
    current = []
    for tok in flat_start:
        if isinstance(tok, Mark):
            seq_expansions.append(current)
            current = []
        else:
            current.append(tok)
    # if the last sequence didn't end with a Mark:
    if current:
        seq_expansions.append(current)

    assert len(seq_expansions) == len(sequences), \
        f"expected {len(sequences)} splits, got {len(seq_expansions)}"

    # assemble one tree per original sequence
    trees = [build_sequence_tree(grammar, exp) for exp in seq_expansions]

    return grammar, trees

def construct_hierarchy(args, dataset_dir, hierarchy_output_dir, skill_folder, mapping, vis_mapping):
  
    # Get all episode traces (not just unique)
    sequence_list = get_all_sequences_list(dataset_dir, skill_folder, mapping)
    sequences = [list(seq) for seq in sequence_list]

    grammar, trees = build_htn_trees(sequences)

    grammar_path = hierarchy_output_dir / 'grammar.txt'
    with grammar_path.open('w') as f:
        for head, productions in grammar.items():
            f.write(f"{head}: {productions}\n")

    # Dictionary to store structure metrics for all trees
    structure_metrics_dict = {}

    for idx, (seq, tree) in enumerate(zip(sequences, trees)):
        png_path = hierarchy_output_dir / f'seq_tree_{idx}'
        visualize_tree(tree, png_path, fmt='pdf', root_label=f'H{idx}', label_mapping=vis_mapping)
        json_path = hierarchy_output_dir / f'tree_{idx}.json'
        json_path.write_text(json.dumps(tree, indent=2))
        
        # Compute structure metrics for this tree
        metrics = compute_structure_metrics(tree)
        structure_metrics_dict[f'tree_{idx}'] = metrics

    # Save structure metrics to JSON file
    metrics_path = hierarchy_output_dir / 'structure_metrics.json'
    with metrics_path.open('w') as f:
        json.dump(structure_metrics_dict, f, indent=2)
    
    print(f"Structure metrics saved to {metrics_path}")
    
    # Count unique trees (using their JSON representation)
    tree_jsons = [json.dumps(tree, sort_keys=True) for tree in trees]
    unique_tree_jsons = set(tree_jsons)
    num_unique_trees = len(unique_tree_jsons)
    print(f"Number of unique trees in {skill_folder}: {num_unique_trees} (out of {len(trees)} total)")
    # Save this info to a file
    unique_count_path = hierarchy_output_dir / 'unique_tree_count.txt'
    with unique_count_path.open('w') as f:
        f.write(f"Number of unique trees: {num_unique_trees} (out of {len(trees)} total)\n")

    return trees

def compute_all_similarity_metrics(gt_trees, pred_trees, output_dir):
    """
    Compute similarity metrics between all pairs of ground truth and predicted trees.
    Returns a dictionary with keys (gt_tree_idx, pred_tree_idx) and similarity metrics as values.
    """
    similarity_metrics_dict = {}
    
    for gt_idx, gt_tree in enumerate(gt_trees):
        for pred_idx, pred_tree in enumerate(pred_trees):
            pair_key = f"(seq_tree_{gt_idx}_gt, seq_tree_{pred_idx}_pred)"
            metrics = compute_similarity_metrics(gt_tree, pred_tree)
            similarity_metrics_dict[pair_key] = metrics
    
    # Save similarity metrics to JSON file
    similarity_path = output_dir / 'similarity_metrics.json'
    with similarity_path.open('w') as f:
        json.dump(similarity_metrics_dict, f, indent=2)
    
    print(f"Similarity metrics saved to {similarity_path}")
    return similarity_metrics_dict


def main():
    parser = argparse.ArgumentParser(description='Build HTN hierarchy from sequences')
    parser.add_argument('--predicted-dir', type=Path, required=True, help='Path to input data directory')
    parser.add_argument('--dataset-dir', type=Path, default='Traces/stone_pick_random/stone_pick_random_pixels', help='Name of the folder containing the skill sequences')
    # parser.add_argument('--threshold', type=int, default=2, help='Minimum frequency of a sequence to be considered (number of times it appears in the data)')
    args = parser.parse_args()
    
    predicted_dir = args.predicted_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = (predicted_dir / 'hierarchy_data').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    pred_hierarchy_output_dir = output_dir / 'predicted_hierarchy'
    pred_hierarchy_output_dir.mkdir(parents=True, exist_ok=True)

    gt_hierarchy_output_dir = output_dir / 'ground_truth_hierarchy'
    gt_hierarchy_output_dir.mkdir(parents=True, exist_ok=True)

    matched_hierarchy_output_dir = output_dir / 'matched_hierarchy'
    matched_hierarchy_output_dir.mkdir(parents=True, exist_ok=True)
    


    #Stores the hungarian matching map for predicted -> intermediate
    mapping_path = predicted_dir / 'mapping' / 'mapping.txt'
    with mapping_path.open('r') as f:
        mapping = {}
        for line in f:
            pred_label, gt_label = line.strip().split()
            mapping[pred_label] = gt_label
    
    #Stores the ground truth mapping for intermediate -> truth
    gt_mapping_path = dataset_dir / 'mapping' / 'mapping.txt'
    with gt_mapping_path.open('r') as f:
        gt_mapping = {}
        for line in f:
            pred_label, gt_label = line.strip().split()
            gt_mapping[pred_label] = gt_label

    #Combine both mappings to get predicted -> truth
    pred_mapping = {}
    for pred_label, intermediate_label in mapping.items():
        if intermediate_label in gt_mapping:
            pred_mapping[pred_label] = gt_mapping[intermediate_label]
        else:
            print(f"Warning: intermediate label '{intermediate_label}' not found in gt_mapping")

    pred_trees = construct_hierarchy(args, predicted_dir, pred_hierarchy_output_dir, 'predicted_skills', None, None)

    gt_trees = construct_hierarchy(args, dataset_dir, gt_hierarchy_output_dir, 'groundTruth', gt_mapping, gt_mapping)

    matched_trees = construct_hierarchy(args, predicted_dir, matched_hierarchy_output_dir, 'predicted_skills', None, pred_mapping)
    # similarity_metrics_dict = compute_all_similarity_metrics(gt_trees, pred_trees, output_dir)

    print("Computing average structure metrics for each set...")

    # --- Compute average structure metrics for each set and save to summary file ---
    def compute_average_metrics(metrics_json_path):
        with open(metrics_json_path, 'r') as f:
            metrics_dict = json.load(f)
        # metrics_dict: {tree_id: {metric: value, ...}, ...}
        if not metrics_dict:
            return {}
        # Get all metric names
        metric_names = list(next(iter(metrics_dict.values())).keys())
        sums = {k: 0.0 for k in metric_names}
        count = 0
        for tree_metrics in metrics_dict.values():
            for k in metric_names:
                sums[k] += tree_metrics.get(k, 0.0)
            count += 1
        avgs = {k: (sums[k] / count if count > 0 else 0.0) for k in metric_names}
        return avgs

    gt_metrics_path = gt_hierarchy_output_dir / 'structure_metrics.json'
    pred_metrics_path = pred_hierarchy_output_dir / 'structure_metrics.json'
    matched_metrics_path = matched_hierarchy_output_dir / 'structure_metrics.json'

    # Read unique tree counts
    def read_unique_tree_count(path):
        try:
            with open(path, 'r') as f:
                return f.readline().strip()
        except Exception:
            return 'Number of unique trees: N/A'

    gt_unique_count = read_unique_tree_count(gt_hierarchy_output_dir / 'unique_tree_count.txt')
    pred_unique_count = read_unique_tree_count(pred_hierarchy_output_dir / 'unique_tree_count.txt')
    matched_unique_count = read_unique_tree_count(matched_hierarchy_output_dir / 'unique_tree_count.txt')

    gt_avg = compute_average_metrics(gt_metrics_path)
    pred_avg = compute_average_metrics(pred_metrics_path)
    matched_avg = compute_average_metrics(matched_metrics_path)

    summary_path = output_dir / 'structure_metrics_summary.txt'
    with summary_path.open('w') as f:
        f.write('Ground Truth Average Structure Metrics:\n')
        f.write(f'  {gt_unique_count}\n')
        for k, v in gt_avg.items():
            f.write(f'  {k}: {v:.4f}\n')
        f.write('\nPredicted Average Structure Metrics:\n')
        f.write(f'  {pred_unique_count}\n')
        for k, v in pred_avg.items():
            f.write(f'  {k}: {v:.4f}\n')
        f.write('\nMatched Average Structure Metrics:\n')
        f.write(f'  {matched_unique_count}\n')
        for k, v in matched_avg.items():
            f.write(f'  {k}: {v:.4f}\n')
    print(f"Average structure metrics summary saved to {summary_path}")


if __name__ == '__main__':
    main()