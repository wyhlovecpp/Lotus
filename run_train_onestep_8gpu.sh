#!/bin/bash

# Script to run train_onestep.py with 8 GPUs using accelerate

# --- Accelerate Configuration ---
# Path to your accelerate config file (if not using default location or want to specify)
# If you have a default config from `accelerate config` and it's set for 8 GPUs, you might not need to specify this.
ACCELERATE_CONFIG_FILE="accelerate_default_config.yaml"

# --- Script Configuration ---
# Path to your training script
TRAIN_SCRIPT_PATH="train_onestep.py" # IMPORTANT: Update this path

# --- MAISI Model & Config Paths (Update these) ---
ENV_CONFIG="./configs/environment_maisi_diff_model.json"
MODEL_CONFIG="./configs/config_maisi_diff_model.json"
MODEL_DEF="./configs/config_maisi.json"
EXISTING_UNET_CKPT="/data1/yuhan/mri/Lotus/models/input_unet3d_data-all_steps1000size512ddpm_random_current_inputx_v1.pt" # IMPORTANT: Update this path
TRAINED_VAE_CKPT="/data1/yuhan/mri/Lotus/models/autoencoder_epoch273.pt"     # IMPORTANT: Update this path

# --- Data Loader Configuration (from data_loader.py cfig) ---
CSV_ROOT="./meta_files/meta_data.csv"                # IMPORTANT: Update this path
SCALE_DOSE_DICT="./meta_files/PTV_DICT.json"          # IMPORTANT: Update this path
PAT_OBJ_DICT="./meta_files/Pat_Obj_DICT.json"        # IMPORTANT: Update this path
DOWN_HU=-1000
UP_HU=1000
DENOM_NORM_HU=500
IN_SIZE_X=96
IN_SIZE_Y=128
IN_SIZE_Z=144
OUT_SIZE_X=96
OUT_SIZE_Y=128
OUT_SIZE_Z=144
NORM_OAR=True          # For boolean flags, the script passes them if True. Change to False if needed.
CAT_STRUCTURES=True   # For boolean flags, the script passes them if True. Change to False if needed.
DOSE_DIV_FACTOR=10.0

# --- Training Hyperparameters ---
OUTPUT_DIR="./output_train_onestep_8gpu"
SEED=42
TRAIN_BATCH_SIZE=1 # This is per-device batch size
NUM_TRAIN_EPOCHS=50 # Adjust as needed
MAX_TRAIN_STEPS=100000 # Set to a number if you prefer step-based training, e.g., 100000
ONESTEP_TIMESTEP=999
GRAD_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-5
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=500
DATALOADER_NUM_WORKERS=4 # Number of workers per GPU for data loading

# --- Optimizer & Precision ---
# USE_8BIT_ADAM="--use_8bit_adam" # Uncomment to use 8-bit Adam
USE_8BIT_ADAM="" # Comment out or leave empty to not use 8-bit Adam
ALLOW_TF32="--allow_tf32"        # Uncomment to allow TF32 on Ampere+ GPUs
MIXED_PRECISION="fp16"         # Choices: "no", "fp16", "bf16". Must match accelerate config if specified there.

# --- Checkpointing & Logging ---
VALIDATION_STEPS=1000 # Example, adapt if you add validation later
CHECKPOINTING_STEPS=1000
CHECKPOINTS_TOTAL_LIMIT=5
REPORT_TO="tensorboard"
TRACKER_PROJECT_NAME="train_onestep_8gpu_project"

# --- Optional features ---
# GRADIENT_CHECKPOINTING="--gradient_checkpointing" # Uncomment to enable
GRADIENT_CHECKPOINTING=""
# ENABLE_XFORMERS="--enable_xformers_memory_efficient_attention" # Uncomment to enable (if UNet supports)
ENABLE_XFORMERS=""

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Construct the command
CMD="accelerate launch --config_file $ACCELERATE_CONFIG_FILE $TRAIN_SCRIPT_PATH \\
    --env_config $ENV_CONFIG \\
    --model_config $MODEL_CONFIG \\
    --model_def $MODEL_DEF \\
    --existing_ckpt_filepath $EXISTING_UNET_CKPT \\
    --trained_autoencoder_path $TRAINED_VAE_CKPT \\
    --csv_root $CSV_ROOT \\
    --scale_dose_dict $SCALE_DOSE_DICT \\
    --pat_obj_dict $PAT_OBJ_DICT \\
    --down_HU $DOWN_HU \\
    --up_HU $UP_HU \\
    --denom_norm_HU $DENOM_NORM_HU \\
    --in_size_x $IN_SIZE_X --in_size_y $IN_SIZE_Y --in_size_z $IN_SIZE_Z \\
    --out_size_x $OUT_SIZE_X --out_size_y $OUT_SIZE_Y --out_size_z $OUT_SIZE_Z \\
    --CatStructures $CAT_STRUCTURES \\
    --dose_div_factor $DOSE_DIV_FACTOR \\
    --output_dir $OUTPUT_DIR \\
    --seed $SEED \\
    --train_batch_size $TRAIN_BATCH_SIZE \\
    --num_train_epochs $NUM_TRAIN_EPOCHS \\
    $(if [ "$MAX_TRAIN_STEPS" != "null" ] ; then echo --max_train_steps $MAX_TRAIN_STEPS; fi) \\
    --onestep_timestep $ONESTEP_TIMESTEP \\
    --gradient_accumulation_steps $GRAD_ACCUMULATION_STEPS \\
    --learning_rate $LEARNING_RATE \\
    --lr_scheduler $LR_SCHEDULER \\
    --lr_warmup_steps $LR_WARMUP_STEPS \\
    --norm_oar $NORM_OAR \\
    $USE_8BIT_ADAM \\
    $ALLOW_TF32 \\
    --mixed_precision $MIXED_PRECISION \\
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \\
    --validation_steps $VALIDATION_STEPS \\
    --checkpointing_steps $CHECKPOINTING_STEPS \\
    --checkpoints_total_limit $CHECKPOINTS_TOTAL_LIMIT \\
    --report_to $REPORT_TO \\
    $GRADIENT_CHECKPOINTING \\
    $ENABLE_XFORMERS \\
    --tracker_project_name $TRACKER_PROJECT_NAME
"

# Print the command to be executed
echo "Running command:"
echo "$CMD"

# Execute the command
eval "$CMD"

echo "Training script finished." 