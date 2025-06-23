#!/bin/bash

# Activate conda environment (adjust if your env name is different)
# source activate vae # Or: conda activate vae

# --- Configuration ---
# Model Paths
MODEL_DIR="/data1/yuhan/mri/Lotus/train_scripts/output/train-maisi-onestep-finetuned-bsz8/checkpoint-5000/"
MODEL_FILENAME="unet_onestep_checkpoint.pt"
TRAINED_AUTOENCODER_PATH="/data1/yuhan/mri/Lotus/models/autoencoder_epoch273.pt" 

# Data Paths
TEST_CSV_DATA_LIST="/data1/yuhan/mri/prostate_dataset/maisi_test_encode/encoded_info.csv"
CONDITION_LATENT_BASE_DIR="/data1/yuhan/mri/prostate_dataset/maisi_test_encode"

# MAISI Config Paths (THESE REMAIN MAISI-SPECIFIC)
ENV_CONFIG="configs/environment_maisi_diff_model.json"
MODEL_CONFIG="configs/config_maisi_diff_model.json"
MODEL_DEF="configs/config_maisi.json"

# Inference Parameters
ONESTEP_TIMESTEP=999 
LATENT_CHANNELS=4
TEST_BATCH_SIZE=1
MIXED_PRECISION="fp16" # "no", "fp16", "bf16"
SEED=42
DATALOADER_NUM_WORKERS=4

# Output Directory
OUTPUT_DIR="output/infer-onestep-5000-results" # Generic name

# GPU setting (for non-accelerate scripts, not used by this script if using accelerate defaults)
# CUDA_VISIBLE_DEVICES=0 

# Ensure the script is run from the Lotus directory or adjust paths accordingly
LOTUS_DIR=$(cd "$(dirname "$0")" && pwd) # Assuming script is in Lotus/ or Lotus/train_scripts/
PYTHON_SCRIPT="$LOTUS_DIR/infer_onestep.py" # Updated script name

# Add Lotus directory to PYTHONPATH if MAISI scripts are imported relatively
export PYTHONPATH="$LOTUS_DIR:$PYTHONPATH"

echo "Starting One-Step Inference..."
echo "Output directory: $OUTPUT_DIR"

python $PYTHON_SCRIPT \
  --env_config=$ENV_CONFIG \
  --model_config=$MODEL_CONFIG \
  --model_def=$MODEL_DEF \
  --trained_autoencoder_path=$TRAINED_AUTOENCODER_PATH \
  --model_dir=$MODEL_DIR \
  --model_filename=$MODEL_FILENAME \
  --test_csv_data_list=$TEST_CSV_DATA_LIST \
  --condition_latent_base_dir=$CONDITION_LATENT_BASE_DIR \
  --onestep_timestep=$ONESTEP_TIMESTEP \
  --latent_channels=$LATENT_CHANNELS \
  --seed=$SEED \
  --test_batch_size=$TEST_BATCH_SIZE \
  --mixed_precision=$MIXED_PRECISION \
  --output_dir=$OUTPUT_DIR \
  --dataloader_num_workers=$DATALOADER_NUM_WORKERS

echo "Inference script finished." 