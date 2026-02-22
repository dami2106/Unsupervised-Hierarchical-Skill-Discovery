# #!/bin/bash

# export CUBLAS_WORKSPACE_CONFIG=:4096:8


ALPHA_TRAIN=0.11
ALPHA_EVAL=0.14

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.1
LAMBDA_ACTIONS_TRAIN=0.1
LAMBDA_FRAMES_EVAL=0.03
LAMBDA_ACTIONS_EVAL=0.1

EPS_TRAIN=0.022
EPS_EVAL=0.382
RADIUS_GW=0.098

# --- Dataset parameters ---
DATASET="../Craftax/Traces/stone_pick_static/stone_pick_static/stone_pick_static_pixels_big"
FEATURE_NAME="pca_features"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="stone_pickaxe_static"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=30
BATCH_SIZE=2
N_FRAMES=135
LEARNING_RATE=1e-05
WEIGHT_DECAY=0.001
LOG=true
VISUALIZE=true
SEED=0
RHO=0.182
N_CLUSTERS=5
LAYERS="650 300 40"

# --- Build the command ---
CMD="python src/train.py \
  --alpha-train $ALPHA_TRAIN \
  --alpha-eval $ALPHA_EVAL \
  --lambda-frames-train $LAMBDA_FRAMES_TRAIN \
  --lambda-actions-train $LAMBDA_ACTIONS_TRAIN \
  --lambda-frames-eval $LAMBDA_FRAMES_EVAL \
  --lambda-actions-eval $LAMBDA_ACTIONS_EVAL \
  --eps-train $EPS_TRAIN \
  --eps-eval $EPS_EVAL \
  --radius-gw $RADIUS_GW \
  --dataset $DATASET \
  --n-frames $N_FRAMES \
  --save-directory $SAVE_DIRECTORY \
  --n-epochs $N_EPOCHS \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --weight-decay $WEIGHT_DECAY \
  --layers $LAYERS \
  --rho $RHO \
  --n-clusters $N_CLUSTERS \
  --val-freq $VAL_FREQ \
  --seed $SEED \
  --run $RUN \
  --feature-name $FEATURE_NAME"

# Append boolean flags if enabled
if [ "$UB_FRAMES" = true ]; then
  CMD="$CMD --ub-frames"
fi

if [ "$UB_ACTIONS" = true ]; then
  CMD="$CMD --ub-actions"
fi

if [ "$STD_FEATS" = true ]; then
  CMD="$CMD --std-feats"
fi

if [ "$VISUALIZE" = true ]; then
  CMD="$CMD --visualize"
fi

if [ "$LOG" = true ]; then
  CMD="$CMD --log"
fi

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
