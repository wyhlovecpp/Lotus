#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
# import csv # No longer needed for load_filenames

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn
import torch.distributed as dist
import transformers
import datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from packaging import version
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from scripts.diff_model_setting import initialize_distributed, load_config # MAISI config loading
from scripts.utils import define_instance # MAISI define_instance
# from monai.data import DataLoader, partition_dataset, CacheDataset # MONAI specific parts removed
# from monai.transforms import Compose, Lambdad, EnsureChannelFirstd # MONAI specific parts removed
from monai.utils import first # Still used for calculate_scale_factor

# <<< Add imports for visualization >>>
import torchvision.utils
from PIL import Image # Optional, but good practice
# <<< End Add >>>

# <<< Imports for new data loader >>>
from data_loader import GetLoader
# <<< End Add >>>


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------
# Old MAISI Data Loading Utilities, RawVAEDataset, create_transforms, round_number, prepare_data
# are removed as we are switching to data_loader.py
# ---------------------------------------------------------------------

def calculate_scale_factor(args: argparse.Namespace, train_loader: torch.utils.data.DataLoader, vae_model: nn.Module, device: torch.device, logger_instance: logging.Logger) -> torch.Tensor:
    """
    Calculate the scaling factor for the dataset, adapted for VAE-encoded labels.
    The VAE model is used to encode the 'label' from the first batch.
    """
    if not train_loader or not len(train_loader.dataset): # type: ignore
        logger_instance.warning("Train loader is empty, cannot calculate scale factor. Defaulting to 1.0")
        return torch.tensor(1.0, device=device)
        
    check_data = first(train_loader)
    if check_data is None or "label" not in check_data:
        logger_instance.warning("First batch from train loader is invalid or lacks 'label' key, cannot calculate scale factor. Defaulting to 1.0")
        return torch.tensor(1.0, device=device)

    raw_labels = check_data["label"].to(device) # [B, 1, D, H, W] or [B, D, H, W]
    if raw_labels.ndim == 4: # Ensure channel dim [B, C, D, H, W] for VAE
        raw_labels = raw_labels.unsqueeze(1)
    
    with torch.no_grad():
        # Ensure VAE is on the correct device and in eval mode for this calculation
        vae_model.to(device).eval()
        # Assuming vae.encode_stage_2_inputs expects [B, C_img_in, D, H, W]
        # The label from data_loader.py is likely [B, 1, D, H, W] after ToTensor and unsqueeze if needed.
        # Or it could be [B, D, H, W] if EnsureChannelFirstd not applied to label in data_loader
        # For MAISI VAE, input is typically [B,1,D,H,W] for single channel image.
        z = vae_model.encode_stage_2_inputs(raw_labels) # Output: [B, C_latent, D_latent, H_latent, W_latent]
        # Move VAE back to CPU if it was temporarily moved and is managed elsewhere
        # Or expect VAE to be on 'device' already. For simplicity, assume it's managed by main training loop's device placement.

    if torch.std(z) == 0:
        logger_instance.warning("Std of first batch (VAE-encoded labels) is 0. Scale factor set to 1.0 to avoid division by zero.")
        scale_factor = torch.tensor(1.0, device=device)
    else:
        scale_factor = 1.0 / torch.std(z)
    
    logger_instance.info(f"Initial scaling factor (from VAE-encoded labels) set to {scale_factor}.")

    if accelerator.num_processes > 1:
        scale_factor_tensor = scale_factor.clone().detach().to(device)
        torch.distributed.all_reduce(scale_factor_tensor, op=torch.distributed.ReduceOp.AVG)
        scale_factor = scale_factor_tensor
        logger_instance.info(f"Final scale_factor after all_reduce: {scale_factor}.")
    else:
        logger_instance.info(f"Final scale_factor (single process or DDP not fully initialized for this part): {scale_factor}.")

    return scale_factor

def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D UNet for one-step dose prediction using VAE-encoded dose maps.")

    # MAISI specific arguments (for UNet and VAE definition)
    parser.add_argument("--env_config", type=str, default="configs/environment_maisi_diff_model.json",
                        help="Path to MAISI environment configuration file.")
    parser.add_argument("--model_config", type=str, default="configs/config_maisi_diff_model.json",
                        help="Path to MAISI model training/inference configuration.")
    parser.add_argument("--model_def", type=str, default="configs/config_maisi.json",
                        help="Path to MAISI model definition file.")
    parser.add_argument("--existing_ckpt_filepath", type=str, required=True,
                        help="Path to the existing MAISI 3D UNet checkpoint.")
    parser.add_argument("--trained_autoencoder_path", type=str, required=True,
                        help="Path to the trained MAISI autoencoder (VAE) checkpoint (used for encoding labels and visualization).")

    # New arguments for data_loader.py configuration
    parser.add_argument('--csv_root', type=str, default='meta_files/meta_data.csv', help='Path to the meta data CSV file for data_loader.py')
    parser.add_argument('--scale_dose_dict', type=str, default='meta_files/PTV_DICT.json', help='Path to PTV dose dictionary JSON for data_loader.py')
    parser.add_argument('--pat_obj_dict', type=str, default='meta_files/Pat_Obj_DICT.json', help='Path to patient object dictionary JSON for data_loader.py')
    parser.add_argument('--down_HU', type=int, default=-1000, help='Bottom clip of CT HU value for data_loader.py')
    parser.add_argument('--up_HU', type=int, default=1000, help='Upper clip of CT HU value for data_loader.py')
    parser.add_argument('--denom_norm_HU', type=int, default=500, help='Denominator for CT normalization for data_loader.py')
    parser.add_argument('--in_size_x', type=int, default=96, help='Input size X for data_loader.py transforms')
    parser.add_argument('--in_size_y', type=int, default=128, help='Input size Y for data_loader.py transforms')
    parser.add_argument('--in_size_z', type=int, default=144, help='Input size Z for data_loader.py transforms')
    parser.add_argument('--out_size_x', type=int, default=96, help='Output size X for data_loader.py transforms')
    parser.add_argument('--out_size_y', type=int, default=128, help='Output size Y for data_loader.py transforms')
    parser.add_argument('--out_size_z', type=int, default=144, help='Output size Z for data_loader.py transforms')
    parser.add_argument('--norm_oar', type=bool, default=True, help='Normalize OAR channel in data_loader.py') # Or use action='store_true'
    parser.add_argument('--CatStructures', type=bool, default=False, help='Concatenate PTVs/OARs in multi-channels or merge in data_loader.py') # Or use action='store_true'
    parser.add_argument('--dose_div_factor', type=float, default=10.0, help='Dose division factor for data_loader.py')
    # Removed: --csv_data_list, --data_dir, --maisi_cache_rate, --maisi_num_workers


    # General training arguments (mostly retained)
    parser.add_argument(
        "--output_dir", type=str, default="onestep-finetuned",
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs."
    )
    parser.add_argument(
        "--onestep_timestep", type=int, default=1,
        help="The single timestep to use for the one-step prediction."
    )
    parser.add_argument(
        "--validation_steps", type=int, default=500,
        help="Run validation every X steps (validation logic to be adapted/removed if not used)."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Initial learning rate."
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant",
        help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32", action="store_true",
        help=("Whether or not to allow TF32 on Ampere GPUs.")
    )
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=4, # Used by GetLoader config
        help=("Number of subprocesses to use for data loading.")
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir", type=str, default="logs",
        help=("[TensorBoard] log directory. Will default to output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    )
    parser.add_argument(
        "--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision."
    )
    parser.add_argument(
        "--report_to", type=str, default="tensorboard",
        help=('The integration to report the results and logs to. Supported platforms are "tensorboard", "wandb", "comet_ml".')
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps", type=int, default=500,
        help="Save a checkpoint of the training state every X updates."
    )
    parser.add_argument(
        "--checkpoints_total_limit", type=int, default=None, help=("Max number of checkpoints to store.")
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help=('Whether training should be resumed from a previous checkpoint.')
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--tracker_project_name", type=str, default="train_onestep_new_loader",
        help="The `project_name` argument passed to Accelerator.init_trackers."
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (for MAISI's initialize_distributed if ever needed, less relevant now).")


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if not args.existing_ckpt_filepath:
        raise ValueError("`--existing_ckpt_filepath` for the MAISI 3D UNet is required.")
    if not args.trained_autoencoder_path:
        raise ValueError("`--trained_autoencoder_path` for the MAISI VAE is required for encoding labels.")
    # Add checks for new required args for data_loader.py config if necessary e.g. csv_root
    if not args.csv_root:
        raise ValueError("`--csv_root` is required for the data loader.")

    return args

def main(args):
    # --- 1. Initialize Accelerator and Basic Setup ---
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200)) 
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) 

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[init_kwargs, ddp_kwargs], 
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Example check for CatStructures, moved here after logger is initialized
    if args.CatStructures:
        logger.warning("Warning: --CatStructures is True. The current data slicing logic in the training loop " +
                       "assumes CatStructures is False for extracting ptv, oar, img conditions. " +
                       "This may lead to incorrect channel selection. Please verify or adapt the slicing logic.")

    if args.seed is not None:
        set_seed(args.seed)
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. Load MAISI Configurations for UNet & VAE --- 
    if accelerator.num_processes > 1:
        os.environ["RANK"] = str(accelerator.process_index)
        os.environ["WORLD_SIZE"] = str(accelerator.num_processes)
        os.environ["LOCAL_RANK"] = str(accelerator.local_process_index) 
        if torch.cuda.is_available():
            torch.cuda.set_device(accelerator.device) 

    maisi_args = load_config(args.env_config, args.model_config, args.model_def)
    
    # --- 3. Load Autoencoder (VAE) --- 
    # VAE is used to encode target 'label' and for visualization.
    logger.info(f"Loading VAE from: {args.trained_autoencoder_path}")
    autoencoder = None
    try:
        autoencoder = define_instance(maisi_args, "autoencoder_def") 
        ckpt_vae = torch.load(args.trained_autoencoder_path, map_location="cpu") 
        vae_state_dict = ckpt_vae.get("state_dict", ckpt_vae) 
        if any(key.startswith("module.") for key in vae_state_dict.keys()):
            vae_state_dict = {k.replace("module.", ""): v for k, v in vae_state_dict.items()}
        autoencoder.load_state_dict(vae_state_dict)
        autoencoder.to(accelerator.device).eval() # Move to device and set to eval
        logger.info("VAE loaded successfully.")
        # Get C_latent: number of channels from VAE output
        # This depends on VAE architecture, often quant_conv.out_channels or similar.
        # For MAISI AutoencoderKL_EMA, it's likely autoencoder.latent_channels or derived from z_channels.
        # Let's assume it's accessible via a property or a known fixed value for MAISI VAE.
        # If autoencoder_def in maisi_args has 'z_channels':
        default_z_channels = maisi_args.autoencoder_def.get('params', {}).get('z_channels', 4) # Corrected access
        C_latent = getattr(autoencoder, 'latent_channels', default_z_channels)
        logger.info(f"VAE latent channels (C_latent): {C_latent}")

    except Exception as e:
        logger.error(f"Failed to load VAE from {args.trained_autoencoder_path}: {e}", exc_info=True)
        raise RuntimeError("VAE is essential and could not be loaded.")


    # --- 4. Load UNet and Modify Input Layer ---
    unet = define_instance(maisi_args, "diffusion_unet_def")
    logger.info(f"MAISI UNet architecture defined. Type: {type(unet)}")

    # New input channels: 3 VAE-encoded conditions + 1 VAE-encoded noisy dose map
    new_total_in_channels = 4 * C_latent
    logger.info(f"Calculated new total input channels for UNet conv_in: {new_total_in_channels} (4 * {C_latent})")

    original_conv_in_module_name = 'conv_in' # Default name
    
    # Load checkpoint first to get original conv_in weights
    logger.info(f"Loading existing MAISI UNet checkpoint from: {args.existing_ckpt_filepath}")
    checkpoint = torch.load(args.existing_ckpt_filepath, map_location="cpu")
    unet_state_dict = checkpoint.get("unet_state_dict", checkpoint.get("state_dict", checkpoint))
    if any(key.startswith("module.") for key in unet_state_dict.keys()):
        unet_state_dict = {k.replace("module.", ""): v for k, v in unet_state_dict.items()}

    # Identify original conv_in configuration from the UNet definition
    if hasattr(unet, original_conv_in_module_name):
        original_conv_in_block = getattr(unet, original_conv_in_module_name)
        actual_original_conv3d = None
        is_monai_conv_block = False

        if isinstance(original_conv_in_block, nn.Sequential) and len(original_conv_in_block) > 0 and isinstance(original_conv_in_block[0], nn.Conv3d):
            actual_original_conv3d = original_conv_in_block[0]
            is_monai_conv_block = True
            logger.info("Detected unet.conv_in as a MONAI Convolution block (nn.Sequential).")
        elif isinstance(original_conv_in_block, nn.Conv3d):
            actual_original_conv3d = original_conv_in_block
            logger.info("Detected unet.conv_in as nn.Conv3d.")
        else:
            logger.error(f"unet.{original_conv_in_module_name} is of unexpected type: {type(original_conv_in_block)}. Cannot reliably get parameters for replacement.")
            raise RuntimeError(f"UNet.{original_conv_in_module_name} has unexpected structure.")

        original_out_channels = actual_original_conv3d.out_channels
        kernel_size = actual_original_conv3d.kernel_size
        stride = actual_original_conv3d.stride
        padding = actual_original_conv3d.padding
        dilation = actual_original_conv3d.dilation
        groups = actual_original_conv3d.groups
        padding_mode = actual_original_conv3d.padding_mode
        has_bias = actual_original_conv3d.bias is not None

        # Create the new Conv3d layer with the new number of input channels
        new_conv3d_layer = nn.Conv3d(
            in_channels=new_total_in_channels,
            out_channels=original_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=has_bias,
            padding_mode=padding_mode
        )

        # Initialize weights for the new conv3d_layer using checkpoint weights
        # The error message "size mismatch for conv_in.conv.weight" implies the key.
        checkpoint_conv_in_weight_key = 'conv_in.conv.weight'
        checkpoint_conv_in_bias_key = 'conv_in.conv.bias'

        if checkpoint_conv_in_weight_key in unet_state_dict:
            checkpoint_weight = unet_state_dict[checkpoint_conv_in_weight_key]
            checkpoint_original_in_channels = checkpoint_weight.shape[1]
            
            if new_total_in_channels % checkpoint_original_in_channels != 0:
                raise ValueError(
                    f"New total input channels ({new_total_in_channels}) must be a multiple of "
                    f"checkpoint's conv_in input channels ({checkpoint_original_in_channels})."
                )
            num_repeats = new_total_in_channels // checkpoint_original_in_channels
            
            expanded_weight = checkpoint_weight.repeat(1, num_repeats, 1, 1, 1)
            expanded_weight *= (1.0 / num_repeats) # Scale weights

            new_conv3d_layer.weight = nn.Parameter(expanded_weight)
            logger.info(f"Initialized new conv_in.weight by repeating checkpoint weights {num_repeats} times and scaling.")

            if has_bias:
                if checkpoint_conv_in_bias_key in unet_state_dict:
                    checkpoint_bias = unet_state_dict[checkpoint_conv_in_bias_key]
                    new_conv3d_layer.bias = nn.Parameter(checkpoint_bias.clone()) # Bias is typically not repeated along input channels
                    logger.info("Initialized new conv_in.bias from checkpoint.")
                else:
                    logger.warning(f"Checkpoint does not contain bias key '{checkpoint_conv_in_bias_key}' for conv_in, but new layer expects bias. Bias will be Kaiming initialized.")
            
            # Remove these keys from state_dict as they are now handled
            del unet_state_dict[checkpoint_conv_in_weight_key]
            if checkpoint_conv_in_bias_key in unet_state_dict:
                del unet_state_dict[checkpoint_conv_in_bias_key]
        else:
            logger.warning(f"Could not find weight key '{checkpoint_conv_in_weight_key}' in checkpoint for conv_in. Weights will be Kaiming initialized.")

        # Replace the old conv_in layer in unet
        if is_monai_conv_block:
            unet.conv_in[0] = new_conv3d_layer
            logger.info(f"Replaced unet.conv_in[0] with new Conv3d layer. New input channels: {new_conv3d_layer.in_channels}")
        else:
            setattr(unet, original_conv_in_module_name, new_conv3d_layer)
            logger.info(f"Replaced unet.{original_conv_in_module_name} with new Conv3d layer. New input channels: {new_conv3d_layer.in_channels}")

        # Load the rest of the UNet weights
        load_result = unet.load_state_dict(unet_state_dict, strict=False)
        logger.info(f"UNet state_dict loaded. Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")
        if any(k.startswith(original_conv_in_module_name) for k in load_result.missing_keys) or \
           any(k.startswith(original_conv_in_module_name) for k in load_result.unexpected_keys):
            logger.warning(f"There are still missing or unexpected keys related to '{original_conv_in_module_name}' after manual handling. Check UNet structure and checkpoint keys.")

    else:
        logger.error(f"Could not find '{original_conv_in_module_name}' in UNet. Cannot modify input channels or load weights correctly.")
        raise RuntimeError(f"UNet does not have expected layer '{original_conv_in_module_name}'.")


    # --- 5. Load MAISI Noise Scheduler ---
    noise_scheduler_maisi = define_instance(maisi_args, "noise_scheduler")
    logger.info(f"MAISI Noise Scheduler loaded: {type(noise_scheduler_maisi)}")

    # --- 6. XFormers and Gradient Checkpointing ---
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available() and hasattr(unet, "enable_xformers_memory_efficient_attention"):
            try:
                unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention for UNet.")
            except Exception as e:
                logger.warning(f"Could not enable xformers for UNet: {e}")
        else:
            logger.info("XFormers not enabled (unavailable, UNet doesn't support it, or flag not set).")

    if args.gradient_checkpointing and hasattr(unet, 'enable_gradient_checkpointing'):
        unet.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing for UNet.")

    # --- 7. Optimizer ---    
    optimizer_cls = torch.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )
    logger.info(f"Optimizer created: {type(optimizer)}")

    # --- 8. Data Preparation using GetLoader ---
    # Create config for GetLoader from args
    loader_cfig = {
        'csv_root': args.csv_root,
        'scale_dose_dict': args.scale_dose_dict,
        'pat_obj_dict': args.pat_obj_dict,
        'down_HU': args.down_HU,
        'up_HU': args.up_HU,
        'denom_norm_HU': args.denom_norm_HU,
        'in_size': (args.in_size_x, args.in_size_y, args.in_size_z),
        'out_size': (args.out_size_x, args.out_size_y, args.out_size_z),
        'norm_oar': args.norm_oar,
        'CatStructures': args.CatStructures, # This will affect how we extract conditions
        'dose_div_factor': args.dose_div_factor,
        'train_bs': args.train_batch_size, # For GetLoader internal use if it creates dataloaders
        'val_bs': args.train_batch_size, # Assuming val_bs is same as train for this script
        'num_workers': args.dataloader_num_workers
    }
    logger.info(f"Config for GetLoader: {loader_cfig}")
    
    # Instantiate GetLoader
    data_loader_manager = GetLoader(loader_cfig)
    train_dataloader = data_loader_manager.train_dataloader() # This directly returns the DataLoader instance
    
    # Ensure num_dataloader_workers in accelerator.prepare matches if GetLoader doesn't handle it.
    # PyTorch DataLoader is created by GetLoader, accelerator will wrap it.
    
    logger.info(f"Train Dataloader prepared via GetLoader. Num batches: {len(train_dataloader)} (on this process if DDP used by GetLoader)")

    # --- 9. Scale Factor ---
    # Calculated from VAE-encoded 'label'
    # VAE (autoencoder) should already be on accelerator.device and in eval mode
    scale_factor = calculate_scale_factor(args, train_dataloader, autoencoder, accelerator.device, logger)
    logger.info(f"Using scale_factor for VAE latents: {scale_factor.item()}")


    # --- 10. LR Scheduler & Accelerator Prepare ---
    len_train_dataloader_local = len(train_dataloader)
    gathered_dataloader_lengths = accelerator.gather(torch.tensor(len_train_dataloader_local, device=accelerator.device))
    
    num_update_steps_per_epoch = 0
    if accelerator.is_main_process:
        if gathered_dataloader_lengths.numel() == 0:
             logger.error("Gathered dataloader lengths tensor is empty.")
             num_update_steps_per_epoch = 0
        else:
            num_update_steps_per_epoch = gathered_dataloader_lengths[0].item()
            if num_update_steps_per_epoch == 0 and torch.any(gathered_dataloader_lengths > 0):
                logger.warning("Main rank dataloader is empty, but other ranks have data. Using max dataloader length.")
                num_update_steps_per_epoch = torch.max(gathered_dataloader_lengths).item()
            elif torch.all(gathered_dataloader_lengths == 0) and len(train_dataloader.dataset) > 0 : # type: ignore
                 logger.error("All dataloaders reported as empty, but global dataset seems non-empty.")
        logger.info(f"Num update steps per epoch on main rank (for LR scheduler): {num_update_steps_per_epoch}")
        num_optimizer_steps_per_epoch = math.ceil(num_update_steps_per_epoch / args.gradient_accumulation_steps)

    num_opt_steps_tensor = torch.tensor([0 if accelerator.is_main_process else 0], dtype=torch.long, device=accelerator.device)
    if accelerator.is_main_process:
        num_opt_steps_tensor[0] = num_optimizer_steps_per_epoch # type: ignore
    
    if accelerator.num_processes > 1:
        torch.distributed.broadcast(num_opt_steps_tensor, src=0)

    num_optimizer_steps_per_epoch = num_opt_steps_tensor[0].item()
    logger.info(f"Synchronized num_optimizer_steps_per_epoch = {num_optimizer_steps_per_epoch} across all ranks.")

    if args.max_train_steps is None:
        if num_optimizer_steps_per_epoch == 0:
             logger.warning("num_optimizer_steps_per_epoch is 0. Setting max_train_steps to 0.")
             args.max_train_steps = 0
        elif args.num_train_epochs is None:
             raise ValueError("Must provide either --max_train_steps or --num_train_epochs")
        else:
             args.max_train_steps = args.num_train_epochs * num_optimizer_steps_per_epoch
    else:
        if num_optimizer_steps_per_epoch > 0:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_optimizer_steps_per_epoch)
        else:
            args.num_train_epochs = 0

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # autoencoder is already on device. unet is not yet.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler # autoencoder is NOT prepared by accelerator here
    )
    # VAE (autoencoder) is used with no_grad, so it doesn't need to be prepared for DDP by accelerator
    # if its parameters are not being optimized. Ensure it's on the correct device.
    autoencoder.to(accelerator.device).eval()


    logger.info("Accelerator .prepare() called for UNet, optimizer, train_dataloader, lr_scheduler.")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        raw_tracker_config = dict(vars(args))
        for k, v in maisi_args.__dict__.items(): # type: ignore
            if k not in raw_tracker_config: raw_tracker_config[f"maisi_{k}"] = v
        
        tracker_config = {}
        for k, v in raw_tracker_config.items():
            if v is None: continue
            if isinstance(v, (int, float, str, bool, torch.Tensor)): tracker_config[k] = v
            elif isinstance(v, list):
                if v and all(isinstance(elem, (int, float, str, bool)) for elem in v): tracker_config[k] = str(v) 
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
        logger.info("Trackers initialized.")

    # --- 11. Training Loop ---
    # Need to know number of channels from PTV, OAR to correctly slice them from batch['data']
    # Assuming comb_optptv, comb_ptv, comb_oar, Body are all single channel from data_loader.py
    # If CatStructures is False:
    # data = [comb_optptv (1), comb_ptv (1), comb_oar (1), Body (1), img (1), beam_plate (1), angle_plate (1), prompt_extend (prompt_channels)]
    # Indices:
    # comb_optptv: 0
    # comb_ptv:    1  (Condition 1 for UNet)
    # comb_oar:    2  (Condition 2 for UNet)
    # Body:        3
    # img:         4  (Condition 3 for UNet)
    # beam_plate:  5
    # angle_plate: 6
    # prompt_extend: 7 onwards
    
    # Determine channel indices based on args.CatStructures
    # This logic assumes fixed single channels for the initial structure parts.
    if args.CatStructures:
        # cat_optptv, cat_ptv, cat_oar, Body, img, beam_plate, angle_plate, prompt_extend
        # The number of channels in cat_optptv, cat_ptv, cat_oar can vary.
        # This makes fixed indexing hard without knowing their channel counts.
        # For simplicity, this refactor will assume args.CatStructures is False,
        # which matches the default in data_loader.py and user's cursor context.
        # If True, this slicing logic needs to be more dynamic.
        logger.error("args.CatStructures=True is not yet fully supported by the current slicing logic in the training loop. Please adapt.")
        raise NotImplementedError("Dynamic channel slicing for CatStructures=True is not implemented.")
    
    # Assuming args.CatStructures is False:
    idx_comb_ptv = 1
    idx_comb_oar = 2
    idx_img = 4
    # Slices for single channel extraction
    slice_comb_ptv = slice(idx_comb_ptv, idx_comb_ptv + 1)
    slice_comb_oar = slice(idx_comb_oar, idx_comb_oar + 1)
    slice_img = slice(idx_img, idx_img + 1)


    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}") # type: ignore
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint) if args.resume_from_checkpoint != "latest" else None
        if args.resume_from_checkpoint == "latest":
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info("Checkpoint 'latest' for resuming given but no checkpoints found.")
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[-1])
            if num_optimizer_steps_per_epoch > 0 : # Avoid division by zero
                first_epoch = global_step // num_optimizer_steps_per_epoch
            else: # If dataloader was empty and caused 0 steps per epoch
                first_epoch = 0 
            logger.info(f"Resumed from step {global_step}, epoch {first_epoch}.")
    
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process, desc="Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train() # Set UNet to train mode
        autoencoder.eval() # Ensure VAE is in eval mode for encoding labels
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # --- Extract Conditions and Label ---
                # batch['data'] shape: [B, TotalChannels, D, H, W]
                # batch['label'] shape: [B, 1, D, H, W] (dose map)
                
                # Ensure data is on correct device and dtype
                raw_conditions_data = batch['data'].to(accelerator.device, dtype=weight_dtype)
                raw_label_dose_map = batch['label'].to(accelerator.device, dtype=weight_dtype) # This is the raw dose map

                # Extract individual raw conditions
                raw_condition_ptv = raw_conditions_data[:, slice_comb_ptv, ...] # [B, 1, D, H, W]
                raw_condition_oar = raw_conditions_data[:, slice_comb_oar, ...] # [B, 1, D, H, W]
                raw_condition_img = raw_conditions_data[:, slice_img, ...]       # [B, 1, D, H, W]
                
                # --- Encode Label and Conditions with VAE --- 
                with torch.no_grad():
                    # VAE expects [B, C_in_img, D, H, W]
                    target_latents = autoencoder.encode_stage_2_inputs(raw_label_dose_map)
                    condition_ptv_latents = autoencoder.encode_stage_2_inputs(raw_condition_ptv)
                    condition_oar_latents = autoencoder.encode_stage_2_inputs(raw_condition_oar)
                    condition_img_latents = autoencoder.encode_stage_2_inputs(raw_condition_img)
                
                # Apply scaling factor to all VAE latents
                s_factor = scale_factor.to(target_latents.device, dtype=target_latents.dtype)
                scaled_target_latents = target_latents * s_factor
                scaled_condition_ptv_latents = condition_ptv_latents * s_factor
                scaled_condition_oar_latents = condition_oar_latents * s_factor
                scaled_condition_img_latents = condition_img_latents * s_factor

                # --- Diffusion Process on Target Latents ---
                current_batch_size = scaled_target_latents.shape[0]
                timesteps = torch.full((current_batch_size,), args.onestep_timestep, device=accelerator.device, dtype=torch.long)
                noise = torch.randn_like(scaled_target_latents)
                
                noisy_target_latents = noise_scheduler_maisi.add_noise(
                    original_samples=scaled_target_latents, noise=noise, timesteps=timesteps
                )
                noisy_target_latents = noisy_target_latents.to(weight_dtype)


                # --- UNet Input and Prediction ---
                # Input: Concatenated VAE-encoded conditions and VAE-encoded noisy target latent
                model_input = torch.cat([
                    scaled_condition_ptv_latents,
                    scaled_condition_oar_latents,
                    scaled_condition_img_latents,
                    noisy_target_latents
                ], dim=1)

                # UNet call - no MAISI-specific conditioning tensors like top/bottom_region_index
                unet_output = unet(
                    x=model_input,
                    timesteps=timesteps,
                    top_region_index_tensor=None, 
                    bottom_region_index_tensor=None,
                    spacing_tensor=None 
                )
                model_pred = unet_output.sample if hasattr(unet_output, 'sample') else unet_output

                # --- Loss Calculation ---
                # Target is the scaled VAE latent of the dose map (before noising)
                loss = F.mse_loss(model_pred.float(), scaled_target_latents.float(), reduction="mean")
                
                # Gather loss for logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # --- Backward Pass and Optimization ---
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm) # type: ignore
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0 # Reset accumulated loss

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        torch.save(
                            {
                                "step": global_step, "epoch": epoch,
                                "unet_state_dict": unwrapped_unet.state_dict(),
                                "scale_factor": scale_factor.cpu(), "args": args
                            },
                            os.path.join(save_path, "unet_onestep_checkpoint.pt")
                        )
                        logger.info(f"Saved state and UNet checkpoint to {save_path}")

                        # <<< Visualization Logic (Needs Update) >>>
                        if autoencoder is not None: # autoencoder is VAE
                            logger.info(f"Generating visualization images for step {global_step}...")
                            vis_save_dir = os.path.join(args.output_dir, "visualizations")
                            os.makedirs(vis_save_dir, exist_ok=True)
                            
                            autoencoder.eval() # Ensure VAE is in eval

                            def decode_latent_and_save(latent_tensor, filename_prefix, axis_name, recon_model, current_scale_factor):
                                try:
                                    # Latent tensor is from UNet output or target (already scaled by overall scale_factor)
                                    # Recon model needs to unscale it before VAE decoding.
                                    vae_input = (latent_tensor / current_scale_factor.to(latent_tensor.device)).to(dtype=weight_dtype)
                                    decoded_img_tensor = recon_model.decode_stage_2_outputs(vae_input) # [1, C_img, D_img, H_img, W_img]
                                    
                                    if decoded_img_tensor.shape[1] != 1: # Assuming dose map is single channel
                                        logger.warning(f"VAE output for {filename_prefix} has {decoded_img_tensor.shape[1]} channels, visualizing first.")
                                    
                                    img_volume = decoded_img_tensor[0, 0] # First item in batch, first channel
                                    D, H, W = img_volume.shape
                                    
                                    if axis_name == "axial": middle_slice_idx, img_slice_2d = D // 2, img_volume[D // 2, :, :]
                                    elif axis_name == "sagittal": middle_slice_idx, img_slice_2d = W // 2, img_volume[:, :, W // 2]
                                    elif axis_name == "coronal": middle_slice_idx, img_slice_2d = H // 2, img_volume[:, H // 2, :]
                                    else: return False
                                    
                                    img_min, img_max = torch.min(img_slice_2d), torch.max(img_slice_2d)
                                    normalized_slice = (img_slice_2d - img_min) / (img_max - img_min + 1e-5)
                                    
                                    vis_filename = os.path.join(vis_save_dir, f"step_{global_step}_{filename_prefix}_{axis_name}.png")
                                    torchvision.utils.save_image(normalized_slice, vis_filename)
                                    return True
                                except Exception as e:
                                    logger.error(f"Failed to process/save {filename_prefix} ({axis_name}): {e}", exc_info=True)
                                    return False

                            def save_condition_slice(condition_tensor, filename_prefix, axis_name): # condition_tensor is [1,1,D,H,W]
                                try:
                                    img_volume = condition_tensor[0, 0] # First item, first channel
                                    D, H, W = img_volume.shape
                                    if axis_name == "axial": img_slice_2d = img_volume[D // 2, :, :]
                                    elif axis_name == "sagittal": img_slice_2d = img_volume[:, :, W // 2]
                                    elif axis_name == "coronal": img_slice_2d = img_volume[:, H // 2, :]
                                    else: return False
                                    
                                    img_min, img_max = torch.min(img_slice_2d), torch.max(img_slice_2d)
                                    normalized_slice = (img_slice_2d - img_min) / (img_max - img_min + 1e-5) # Normalize for visualization
                                    
                                    vis_filename = os.path.join(vis_save_dir, f"step_{global_step}_{filename_prefix}_{axis_name}.png")
                                    torchvision.utils.save_image(normalized_slice, vis_filename)
                                    return True
                                except Exception as e:
                                    logger.error(f"Failed to save condition {filename_prefix} ({axis_name}): {e}", exc_info=True)
                                    return False

                            try:
                                # VAE (autoencoder) is already on accelerator.device, eval()
                                with torch.no_grad():
                                    # Select first sample from batch for visualization
                                    sample_pred_latent = model_pred[0].unsqueeze(0)         # From UNet output
                                    sample_target_latent = scaled_target_latents[0].unsqueeze(0) # Target for UNet (scaled VAE(label))
                                    
                                    # Raw conditions from this batch (first sample) - these are now latents
                                    sample_cond_ptv_latent = scaled_condition_ptv_latents[0].unsqueeze(0)
                                    sample_cond_oar_latent = scaled_condition_oar_latents[0].unsqueeze(0)
                                    sample_cond_img_latent = scaled_condition_img_latents[0].unsqueeze(0)
                                    
                                    latents_to_decode = {
                                        "pred_dose": sample_pred_latent, 
                                        "target_dose": sample_target_latent,
                                        "cond_ptv_recon": sample_cond_ptv_latent,
                                        "cond_oar_recon": sample_cond_oar_latent,
                                        "cond_img_recon": sample_cond_img_latent
                                    }
                                    axes_to_visualize = ["axial", "sagittal", "coronal"]

                                    for prefix, latent_data in latents_to_decode.items():
                                        for axis_name in axes_to_visualize:
                                            decode_latent_and_save(latent_data, prefix, axis_name, autoencoder, scale_factor)
                                    
                                    logger.info(f"Finished attempting visualization saves for step {global_step}.")
                            except Exception as e:
                                logger.error(f"Error during visualization at step {global_step}: {e}", exc_info=True)
                            # No need to move autoencoder to CPU as it's managed by accelerator.device now
                        # <<< End Visualization Logic >>>
            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        
    # --- 12. Save the Final Model & Clean Up ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_final = accelerator.unwrap_model(unet)
        final_save_path = os.path.join(args.output_dir, "final_unet_onestep.pt")
        torch.save(
            {
                "step": global_step, "epoch": args.num_train_epochs -1, # Or current epoch
                "unet_state_dict": unet_final.state_dict(),
                "scale_factor": scale_factor.cpu(), "args": args
            },
            final_save_path
        )
        logger.info(f"Saved final UNet one-step model to {final_save_path}")

    accelerator.end_training()
    logger.info("Training finished.")

if __name__ == "__main__":
    args = parse_args()
    # Example check for CatStructures, as current slicing is fixed for False
    # This block has been moved into main()
    main(args) 