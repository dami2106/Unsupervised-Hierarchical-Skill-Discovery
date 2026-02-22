# Define your list of seeds
seeds=(888 0 333 9 42)

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo "Starting PPO training with seed: $seed"
    echo "=========================================="

    python ppo_hierarchy.py --skill_list wooden_pickaxe stone_pickaxe wood stone table --root Traces/stone_pick_static --bc_checkpoint_dir bc_checkpoints_gt\
    --pca_model_path pca_models/pca_model_650.joblib --pu_start_models_dir pu_start_models_gt --pu_end_models_dir pu_end_models_gt --run_name ppo_hierarchy_groundtruth --hierarchy_dir Traces/stone_pick_static/hierarchy_data/ground_truth_hierarchy --ppo_seed "$seed"

    echo "Finished run for seed: $seed"
    echo
done

echo "All runs completed!"
