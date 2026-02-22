# Define your list of seeds
seeds=(888 0 333 9 42)

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo "Starting PPO training with seed: $seed"
    echo "=========================================="

    python ppo_skills.py --skill_list 0 1 2 3 4 --root Traces/stone_pick_static --bc_checkpoint_dir bc_checkpoints_ompn\
    --pca_model_path pca_models/pca_model_650.joblib --pu_start_models_dir pu_start_models_ompn --pu_end_models_dir pu_end_models_ompn --run_name ppo_options_ompn --ppo_seed "$seed"

    echo "Finished run for seed: $seed"
    echo
done

echo "All runs completed!"
