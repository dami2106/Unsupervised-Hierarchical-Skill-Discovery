#!/bin/bash

for skill in 0 1 2 3 4; do
    python Skill_Learning/bc_resnet.py --skill "$skill" --dir 'Traces/stone_pick_static' \
    --image_dir_name 'top_down_obs' --backbone 'resnet34' --skills_name 'ompn_skills' --save_dir 'bc_checkpoints_ompn'
    echo "Done $skill"
done

