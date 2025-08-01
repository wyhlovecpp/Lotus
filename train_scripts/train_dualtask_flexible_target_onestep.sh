#!/bin/bash

# Ensure the script is run from the Lotus directory or adjust paths accordingly
# Get the absolute path of the parent directory (Lotus)
LOTUS_DIR=$(cd "$(dirname "$0")/.." && pwd) # Adjusted for script in train_scripts/

# Add Lotus directory to PYTHONPATH to resolve maisi.scripts imports
export PYTHONPATH="$LOTUS_DIR:$PYTHONPATH"

# --- User-defined Paths and Parameters (PLEASE EDIT THESE) ---
export PATH_TO_MAISI_CHECKPOINT="/data1/yuhan/mri/Lotus/models/input_unet3d_data-all_steps1000size512ddpm_random_current_inputx_v1.pt"
export PATH_TO_TRAINED_AUTOENCODER="/data1/yuhan/mri/Lotus/models/autoencoder_epoch273.pt"
export PATH_TO_CSV_DATA_LIST="/data1/yuhan/mri/breast/encode/encoded_info.csv" # CSV for flexible target data
export PATH_TO_EMBEDDING_BASE_DIR="/data1/yuhan/mri/breast/encode"
export CUDA_DEVICE_IDS_CONCAT="01234567" # Example: Use all 8 GPUs

# MAISI Configs (assuming they are in Lotus/configs/)
export MAISI_ENV_CONFIG="$LOTUS_DIR/configs/environment_maisi_diff_model.json"
export MAISI_MODEL_CONFIG="$LOTUS_DIR/configs/config_maisi_diff_model.json"
export MAISI_MODEL_DEF="$LOTUS_DIR/configs/config_maisi.json"

# Training Hyperparameters (adjust as needed)
export TRAIN_BATCH_SIZE=1 # Per device (effective batch doubles due to dual task)
export GRADIENT_ACCUMULATION_STEPS=1
export LEARNING_RATE=1e-5
export MAX_TRAIN_STEPS=100000 
export NUM_TRAIN_EPOCHS=1000 
export LR_SCHEDULER="constant"
export LR_WARMUP_STEPS=0
export ONESTEP_TIMESTEP=999 
export MIXED_PRECISION="fp16" 

export OUTPUT_DIR_BASE="$LOTUS_DIR/output/train-dualtask-flexible-target"
export OUTPUT_DIR="${OUTPUT_DIR_BASE}-bsz$(($TRAIN_BATCH_SIZE * ${#CUDA_DEVICE_IDS_CONCAT} * $GRADIENT_ACCUMULATION_STEPS))"
export CHECKPOINTING_STEPS=5000
export VALIDATION_STEPS=1000 # Note: script does not currently have a validation loop
export SEED=42
export DATALOADER_NUM_WORKERS=4 
export MAISI_CACHE_RATE=1.0
export MAISI_NUM_WORKERS=4 

# Calculate number of GPUs from CUDA_DEVICE_IDS_CONCAT string length
export NUM_GPUS=${#CUDA_DEVICE_IDS_CONCAT}

# Accelerator config file (relative to the LOTUS_DIR)
export ACCELERATE_CONFIG_FILE="$LOTUS_DIR/accelerate_configs/${CUDA_DEVICE_IDS_CONCAT}.yaml"

echo "Starting Dual-Task Flexible Target One-Step Training..."
echo "Using $NUM_GPUS GPUs: $CUDA_DEVICE_IDS_CONCAT"
echo "Accelerator config: $ACCELERATE_CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"

# Check if accelerate config file exists
if [ ! -f "$ACCELERATE_CONFIG_FILE" ]; then
    echo "Error: Accelerate config file $ACCELERATE_CONFIG_FILE not found!"
    echo "Please ensure you have an accelerate config file for $NUM_GPUS GPU(s) named appropriately."
    exit 1
fi

accelerate launch --config_file=$ACCELERATE_CONFIG_FILE --mixed_precision=$MIXED_PRECISION \
  --main_process_port="29503" \  # Changed port to avoid conflict
  "$LOTUS_DIR/train_dualtask_flexible_target_onestep.py" \
  --env_config=$MAISI_ENV_CONFIG \
  --model_config=$MAISI_MODEL_CONFIG \
  --model_def=$MAISI_MODEL_DEF \
  --existing_ckpt_filepath="$PATH_TO_MAISI_CHECKPOINT" \
  --trained_autoencoder_path="$PATH_TO_TRAINED_AUTOENCODER" \
  --csv_data_list="$PATH_TO_CSV_DATA_LIST" \
  --embedding_base_dir="$PATH_TO_EMBEDDING_BASE_DIR" \
  --maisi_cache_rate=$MAISI_CACHE_RATE \
  --maisi_num_workers=$MAISI_NUM_WORKERS \
  --output_dir="$OUTPUT_DIR" \
  --seed=$SEED \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --onestep_timestep=$ONESTEP_TIMESTEP \
  --validation_steps=$VALIDATION_STEPS \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --gradient_checkpointing \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler="$LR_SCHEDULER" \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  --mixed_precision=$MIXED_PRECISION \
  --report_to="tensorboard" \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=5 \
  --resume_from_checkpoint="latest" \
  --tracker_project_name="train_dualtask_flexible_target_onestep" \
  --num_gpus=$NUM_GPUS

echo "Dual-Task Flexible Target Training script finished." 