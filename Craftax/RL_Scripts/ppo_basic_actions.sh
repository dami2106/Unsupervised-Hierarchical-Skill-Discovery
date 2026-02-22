# Define your list of seeds
seeds=(888 0 333 9 42)

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo "Starting PPO training with seed: $seed"
    echo "=========================================="

    python ppo_basic_actions.py --ppo_seed "$seed"

    echo "Finished run for seed: $seed"
    echo
done

echo "All runs completed!"
