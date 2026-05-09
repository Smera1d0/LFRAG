#!/bin/bash
export PYTHONPATH=/your-path-to/LFRAG:$PYTHONPATH
# ==================== Configurable Parameters ====================
MODEL_PATH="./models/colqwen2.5-v0.2"
OUTPUT_DIR="./ckpts"
BATCH_SIZE=8
EPOCH=2
LR=2e-4
TAU=0.02

# Dataset paths
TRAIN_JSONL="./datasets/train/docmatix_train.jsonl"
TRAIN_IMAGE_DIR="./datasets/train"
EVAL_SIZE=1000

# WandB settings
WANDB_PROJECT="your-wandb-project-name"
WANDB_NAME="lfrag-training"
WANDB_ENTITY="your-wandb-entity-name"
WANDB_API_KEY="your-wandb-api-key"
# =================================================================

accelerate launch \
  --config_file ./scripts/train_config.yaml \
  scripts/train_lfrag.py \
  --output_dir ${OUTPUT_DIR} \
  --model_path ${MODEL_PATH} \
  --lr ${LR} \
  --tau ${TAU} \
  --epoch ${EPOCH} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --dataset_jsonl_path ${TRAIN_JSONL} \
  --dataset_image_path ${TRAIN_IMAGE_DIR} \
  --eval_size ${EVAL_SIZE} \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_name ${WANDB_NAME} \
  --wandb_entity ${WANDB_ENTITY} \
  --wandb_api_key ${WANDB_API_KEY} \
  --peft