#!/bin/bash

python Skill_Learning/train_start_model_pulearning.py --dir 'Traces/stone_pick_static' --skills_dirname 'compile_skills' \
--features_name 'pca_features_650' --old_data_mode --save_dir 'pu_start_models_compile'
