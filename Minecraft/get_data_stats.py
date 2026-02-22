from argparse import ArgumentParser
import os
from collections import Counter, defaultdict
import json

if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--data-dir", type=str, default="Data/minecraft_cobblestone_mapped")

    args = parser.parse_args()

    episode_lengths = []
    ground_truth_distribution = Counter()
    ground_truth_skills = set()

    # New structures
    skill_lengths = defaultdict(list)       # list of consecutive run lengths per skill
    skill_start_timesteps = defaultdict(list)  # list of start indices per skill

    # List all the files in the args.data_dir directory
    files = os.listdir(os.path.join(args.data_dir, "asot_skills"))
    mapping_dir = os.path.join(args.data_dir, "asot_mapping")
    os.makedirs(mapping_dir, exist_ok=True)

    for file in files:
        with open(os.path.join(args.data_dir, "asot_skills", file), 'r') as f:
            data = f.read().strip()
            if not data:
                continue

            skills = [s for s in data.split("\n") if s]
            episode_length = len(skills)
            episode_lengths.append(episode_length)

            # Count total occurrences
            for s in skills:
                ground_truth_distribution[s] += 1
                ground_truth_skills.add(s)

            # Track skill run lengths and start times
            prev_skill = None
            run_length = 0
            for i, skill in enumerate(skills):
                if skill != prev_skill:
                    # new run starts
                    if prev_skill is not None:
                        skill_lengths[prev_skill].append(run_length)
                    skill_start_timesteps[skill].append(i)
                    run_length = 1
                    prev_skill = skill
                else:
                    run_length += 1
            # add the last run
            if prev_skill is not None:
                skill_lengths[prev_skill].append(run_length)

    # Compute averages
    avg_skill_length = {
        skill: sum(lengths) / len(lengths) for skill, lengths in skill_lengths.items()
    }
    avg_skill_start_timestep = {
        skill: sum(starts) / len(starts) for skill, starts in skill_start_timesteps.items()
    }

    stats = {
        "total_episodes": len(episode_lengths),
        "unique_skills": len(ground_truth_skills),
        "min_episode_length": min(episode_lengths) if episode_lengths else 0,
        "avg_episode_length": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0,
        "max_episode_length": max(episode_lengths) if episode_lengths else 0,
        "ground_truth_distribution": dict(ground_truth_distribution),
        "average_skill_length": avg_skill_length,
        "average_skill_start_timestep": avg_skill_start_timestep,
    }

    # Save the stats to a JSON file
    with open(os.path.join(args.data_dir, "asot_stats.json"), 'w') as f:
        json.dump(stats, f, indent=4)

    # Create a mapping of i -> skill
    skill_mapping = {i: skill for i, skill in enumerate(sorted(ground_truth_skills))}

    with open(os.path.join(mapping_dir, "mapping.txt"), "w") as f:
        for i, skill in skill_mapping.items():
            line = f"{i} {skill}"
            if i < len(skill_mapping) - 1:
                f.write(line + "\n")
            else:
                f.write(line)
