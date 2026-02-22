import os 
from collections import Counter


def get_unique_sequence_list(trace_directory, skill_folder = "predicted_skills", mapping = None):
    """
    Get a list of unique sequences from the trace directory.
    """
    sequence_list = []

    if mapping is not None:
        # Reverse the mapping dictionary
        mapping = {v: k for k, v in mapping.items()}

    for filename in os.listdir(f"{trace_directory}/{skill_folder}/"):
        with open(os.path.join(f"{trace_directory}/{skill_folder}/", filename), 'r') as file:
            sequence = file.read().strip().split()

            result = []
            prev = None
            for item in sequence:
                if item != prev:
                    result.append(item)
                    prev = item


            
            if mapping is not None:
                sequence_list.append(tuple(mapping[skill] for skill in result))
            else:
                sequence_list.append(tuple(skill for skill in result))
    
    counts = Counter(sequence_list)

    return counts
    

def get_all_sequences_list(trace_directory, skill_folder = "predicted_skills", mapping = None):
    """
    Get a list of all processed sequences (one per episode) from the trace directory.
    Each sequence is a tuple of skills, with consecutive duplicates removed and mapping applied if provided.
    """
    sequence_list = []

    if mapping is not None:
        # Reverse the mapping dictionary
        mapping = {v: k for k, v in mapping.items()}

    for filename in os.listdir(f"{trace_directory}/{skill_folder}/"):
        with open(os.path.join(f"{trace_directory}/{skill_folder}/", filename), 'r') as file:
            sequence = file.read().strip().split()

            result = []
            prev = None
            for item in sequence:
                if item != prev:
                    result.append(item)
                    prev = item

            if mapping is not None:
                sequence_list.append(tuple(mapping[skill] for skill in result))
            else:
                sequence_list.append(tuple(skill for skill in result))

    return sequence_list


if __name__ == "__main__":
    # trace_directory = "runs/stone_pick_random/stone_pick_random_pixels_stone_pick_random_pixels/version_0"
    trace_directory = "Traces/stone_pick_random/stone_pick_random_pixels"

    mapping = {
        'table': '0',
        'wood': '1',
        'wood_pickaxe': '2',
        'stone': '3',
        'stone_pickaxe': '4',
    }

    counts = get_unique_sequence_list(trace_directory, "groundTruth", mapping)
    print(counts)

    