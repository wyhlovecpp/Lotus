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
import csv 
from collections import defaultdict

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn
import torch.distributed as dist
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from packaging import version
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

# Diffusers imports
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# MAISI imports (assuming scripts. and monai. are in PYTHONPATH or accessible)
from scripts.diff_model_setting import initialize_distributed, load_config, setup_logging
from scripts.utils import define_instance
from monai.data import DataLoader, partition_dataset, CacheDataset
from monai.transforms import Compose, Lambdad
from monai.utils import first

# Visualization imports
import torchvision.utils
from PIL import Image

check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# --- Data Loading Utilities for Dual-Task Flexible Target ---
def load_filenames_flexible_target(csv_data_list: str, embedding_base_dir: str) -> list:
    """
    Load filenames from a CSV for dual-task learning with flexible primary target.
    Each CSV row should have 'input_path' (condition, e.g., *_0000_encoded.npy) 
    and 'target_path' (primary DCE target, e.g., *_0000_encoded.npy or *_0001_encoded.npy).

    Determines 'dce_target_type' based on 'target_path' suffix.
    Constructs absolute paths and checks for existence.
    Returns a list of dicts: 
    [{
        "condition_path": abs_path_cond,
        "dce_target_path": abs_path_dce,
        "dce_target_type": "0000" or "0001",
        "top_region_index": "", "bottom_region_index": "", "spacing": ""
    }, ...]
    """
    paths_list = []
    if not os.path.exists(csv_data_list):
        raise FileNotFoundError(f"CSV file not found: {csv_data_list}")

    with open(csv_data_list, newline="") as f:
        reader = csv.DictReader(f)
        if "input_path" not in reader.fieldnames or "target_path" not in reader.fieldnames:
            raise ValueError("CSV file must contain 'input_path' and 'target_path' columns.")
        
        for row in reader:
            condition_p_rel = row["input_path"]
            dce_target_p_rel = row["target_path"]

            abs_condition_path = os.path.join(embedding_base_dir, condition_p_rel)
            abs_dce_target_path = os.path.join(embedding_base_dir, dce_target_p_rel)

            dce_target_type = None
            if dce_target_p_rel.endswith("_0000_encoded.npy"):
                dce_target_type = "0000"
            elif dce_target_p_rel.endswith("_0001_encoded.npy"):
                dce_target_type = "0001"
            # Add more types here if needed, e.g., for "_0002_encoded.npy"
            # elif dce_target_p_rel.endswith("_0002_encoded.npy"):
            #     dce_target_type = "0002"
            else:
                logger.warning(f"Unknown DCE target suffix for {dce_target_p_rel}. Skipping row.")
                continue

            if os.path.exists(abs_condition_path) and os.path.exists(abs_dce_target_path):
                paths_list.append({
                    "condition_path": abs_condition_path,
                    "dce_target_path": abs_dce_target_path,
                    "dce_target_type": dce_target_type,
                    "top_region_index": "",  # Placeholder
                    "bottom_region_index": "",  # Placeholder
                    "spacing": ""  # Placeholder
                })
            else:
                logger.warning(f"Skipping entry due to missing files: "
                               f"Cond: {abs_condition_path} (Exists: {os.path.exists(abs_condition_path)}), "
                               f"DCE Target: {abs_dce_target_path} (Exists: {os.path.exists(abs_dce_target_path)})")
    
    if not paths_list:
        logger.warning(f"No valid (condition, dce_target) pairs found from {csv_data_list} and {embedding_base_dir}.")
    return paths_list

def load_and_squeeze(x):
    """Loads .npy file and squeezes the first dimension if it's 1."""
    data = np.load(x) if isinstance(x, str) else x
    if data.ndim > 0 and data.shape[0] == 1: # Check ndim before shape
        data = np.squeeze(data, axis=0)
    return data

def prepare_data_flexible_target(
    args: argparse.Namespace,
    train_files_dicts: list, 
    is_distributed: bool,
    local_rank: int = 0
) -> DataLoader:
    train_transforms = Compose([
        Lambdad(keys=["condition_path", "dce_target_path"], func=load_and_squeeze),
        # dce_target_type is passed through, no transform needed for it here.
        # Metadata placeholders are filled by these Lambdad transforms.
        Lambdad(keys="top_region_index", func=lambda _: [0, 0, 1, 0]),
        Lambdad(keys="bottom_region_index", func=lambda _: [0, 0, 1, 0]), # Actual value set in training loop
        Lambdad(keys="spacing", func=lambda _: [1.0, 1.0, 1.2]),
    ])

    if is_distributed:
        data_partition = partition_dataset(
            data=train_files_dicts, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True
        )[local_rank]
    else:
        data_partition = train_files_dicts

    if not data_partition:
        logger.warning(f"Rank {local_rank} received an empty data partition. Returning an empty DataLoader.")
        return DataLoader([], batch_size=args.train_batch_size, shuffle=False)

    train_ds = CacheDataset(
        data=data_partition, 
        transform=train_transforms, 
        cache_rate=args.maisi_cache_rate, 
        num_workers=args.maisi_num_workers
    )
    
    if not train_ds:
        logger.warning(f"CacheDataset for rank {local_rank} is empty.")
        return DataLoader([], batch_size=args.train_batch_size, shuffle=False)

    return DataLoader(
        train_ds, 
        num_workers=args.dataloader_num_workers, 
        batch_size=args.train_batch_size, 
        shuffle=True
    )

def calculate_scale_factor(args: argparse.Namespace, train_loader: DataLoader, device: torch.device, logger_instance: logging.Logger) -> torch.Tensor:
    if not train_loader or not len(train_loader.dataset):
        logger_instance.warning("Train loader is empty, cannot calculate scale factor. Defaulting to 1.0")
        return torch.tensor(1.0, device=device)
        
    check_data = first(train_loader)
    if check_data is None or "condition_path" not in check_data: 
        logger_instance.warning("First batch from train loader is invalid or lacks 'condition_path' key. Defaulting to 1.0")
        return torch.tensor(1.0, device=device)

    # Note: condition_path from batch is actually the loaded tensor now
    z = check_data["condition_path"].to(device) 
    if torch.std(z) == 0:
        logger_instance.warning("Std of first batch (condition latent) is 0. Scale factor set to 1.0.")
        scale_factor = torch.tensor(1.0, device=device)
    else:
        scale_factor = 1.0 / torch.std(z)
    
    logger_instance.info(f"Initial scaling factor set to {scale_factor.item()}.")

    if accelerator.num_processes > 1:
        scale_factor_tensor = scale_factor.clone().detach().to(device)
        torch.distributed.all_reduce(scale_factor_tensor, op=torch.distributed.ReduceOp.AVG)
        scale_factor = scale_factor_tensor
        logger_instance.info(f"Final scale_factor after all_reduce: {scale_factor.item()}.")
    else:
        logger_instance.info(f"Final scale_factor (single process): {scale_factor.item()}.")
    return scale_factor

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D UNet for one-step dual-task prediction with flexible targets.")
    # MAISI specific arguments
    parser.add_argument("--env_config", type=str, default="configs/environment_maisi_diff_model.json")
    parser.add_argument("--model_config", type=str, default="configs/config_maisi_diff_model.json")
    parser.add_argument("--model_def", type=str, default="configs/config_maisi.json")
    parser.add_argument("--existing_ckpt_filepath", type=str, required=True, help="Path to existing MAISI 3D UNet checkpoint.")
    parser.add_argument("--trained_autoencoder_path", type=str, default="models/autoencoder_epoch273.pt", help="Path to VAE for visualization.")
    # Data arguments
    parser.add_argument("--csv_data_list", type=str, required=True, help="Path to CSV file for data.")
    parser.add_argument("--embedding_base_dir", type=str, required=True, help="Base directory for VAE encoded npy files.")
    parser.add_argument("--maisi_cache_rate", type=float, default=1.0)
    parser.add_argument("--maisi_num_workers", type=int, default=4)
    # General training arguments
    parser.add_argument("--output_dir", type=str, default="dualtask-flexible-onestep-finetuned")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=2) 
    parser.add_argument("--num_train_epochs", type=int, default=100) 
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--onestep_timestep", type=int, default=999) 
    parser.add_argument("--validation_steps", type=int, default=1000) 
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0) 
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default="train_dualtask_flexible_target_onestep")
    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if not args.csv_data_list or not args.embedding_base_dir:
        raise ValueError("`--csv_data_list` and `--embedding_base_dir` are required.")
    if not args.existing_ckpt_filepath:
        raise ValueError("`--existing_ckpt_filepath` for the MAISI 3D UNet is required.")
    return args

# --- Fixed Metadata Values (examples, customize as needed) ---
# These define the specific vectors for bottom_region_index per task type
BOTTOM_REGION_TARGET_TYPE_0000 = [1, 0, 0, 0]  # Example for target type "0000"
BOTTOM_REGION_TARGET_TYPE_0001 = [0, 0, 1, 0]  # Example for target type "0001"
# BOTTOM_REGION_TARGET_TYPE_0002 = [1, 1, 0, 0] # Example if you add a "0002" type
BOTTOM_REGION_RECONSTRUCTION_TASK = [0, 1, 0, 0] # For original image reconstruction task
DEFAULT_BOTTOM_REGION = [0,0,0,0] # Fallback if type is unknown

FIXED_TOP_REGION_INDEX_VALUE = [0, 0, 1, 0]
FIXED_SPACING_VALUE = [1.0, 1.0, 1.2]

# --- Main Training Function ---
def main(args):
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

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if accelerator.num_processes > 1:
        os.environ["RANK"] = str(accelerator.process_index)
        os.environ["WORLD_SIZE"] = str(accelerator.num_processes)
        os.environ["LOCAL_RANK"] = str(accelerator.local_process_index)
        if torch.cuda.is_available():
            torch.cuda.set_device(accelerator.device)

    maisi_args = load_config(args.env_config, args.model_config, args.model_def)
    unet = define_instance(maisi_args, "diffusion_unet_def")
    logger.info(f"MAISI UNet architecture defined. Type: {type(unet)}")
    checkpoint = torch.load(args.existing_ckpt_filepath, map_location="cpu")
    unet_state_dict = checkpoint.get("unet_state_dict", checkpoint.get("state_dict", checkpoint))
    if any(key.startswith("module.") for key in unet_state_dict.keys()):
        unet_state_dict = {k.replace("module.", ""): v for k, v in unet_state_dict.items()}

    original_conv_in = None
    if hasattr(unet, 'conv_in'):
        if isinstance(unet.conv_in, nn.Conv3d):
            original_conv_in = unet.conv_in
        elif isinstance(unet.conv_in, nn.Sequential) and len(unet.conv_in) > 0 and isinstance(unet.conv_in[0], nn.Conv3d):
             original_conv_in = unet.conv_in[0]
             logger.info("Detected conv_in wrapped in nn.Sequential. Accessing internal Conv3d.")
        else:
            logger.error(f"unet.conv_in type {type(unet.conv_in)} not modifiable.")
    else:
        logger.error("Could not find 'unet.conv_in'.")

    if original_conv_in is not None:
        original_in_channels = original_conv_in.in_channels
        load_result = unet.load_state_dict(unet_state_dict, strict=False)
        logger.info(f"UNet state_dict loaded. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")
        
        loaded_weight = original_conv_in.weight.clone().detach()
        if loaded_weight.shape[1] != original_in_channels:
             logger.error(f"Loaded conv_in weight has {loaded_weight.shape[1]} vs expected {original_in_channels} channels. Modification aborted.")
        else:
            new_in_channels = original_in_channels * 2 # For [condition, noisy_target]
            new_weight_tensor = loaded_weight.repeat(1, 2, 1, 1, 1) * 0.5
            new_bias_tensor = original_conv_in.bias.clone().detach() if original_conv_in.bias is not None else None
            new_conv_in_layer = nn.Conv3d(
                in_channels=new_in_channels, out_channels=original_conv_in.out_channels,
                kernel_size=original_conv_in.kernel_size, stride=original_conv_in.stride,
                padding=original_conv_in.padding, dilation=original_conv_in.dilation,
                groups=original_conv_in.groups, bias=(original_conv_in.bias is not None),
                padding_mode=original_conv_in.padding_mode )
            new_conv_in_layer.weight = nn.Parameter(new_weight_tensor)
            if new_bias_tensor is not None: new_conv_in_layer.bias = nn.Parameter(new_bias_tensor)
            if isinstance(unet.conv_in, nn.Sequential): unet.conv_in[0] = new_conv_in_layer
            else: unet.conv_in = new_conv_in_layer
            logger.info(f"Modified UNet conv_in to {new_in_channels} input channels.")
    else:
        load_result = unet.load_state_dict(unet_state_dict, strict=True) 
        logger.info(f"UNet state_dict loaded (no conv_in mod). Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")

    noise_scheduler_maisi = define_instance(maisi_args, "noise_scheduler")
    logger.info(f"MAISI Noise Scheduler loaded: {type(noise_scheduler_maisi)}")

    autoencoder = None
    if args.trained_autoencoder_path and os.path.exists(args.trained_autoencoder_path):
        logger.info(f"Loading VAE from: {args.trained_autoencoder_path}")
        try:
            autoencoder = define_instance(maisi_args, "autoencoder_def") 
            ckpt_vae = torch.load(args.trained_autoencoder_path, map_location="cpu") 
            vae_state_dict = ckpt_vae.get("state_dict", ckpt_vae) 
            if any(key.startswith("module.") for key in vae_state_dict.keys()):
                vae_state_dict = {k.replace("module.", ""): v for k, v in vae_state_dict.items()}
            autoencoder.load_state_dict(vae_state_dict)
            autoencoder.eval()
            logger.info("VAE loaded successfully for visualization.")
        except Exception as e:
            logger.error(f"Failed to load VAE: {e}. Visualization disabled.")
            autoencoder = None
    else:
        logger.warning("VAE path not provided or not found. Visualization disabled.")

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        try: 
            if hasattr(unet, "enable_xformers_memory_efficient_attention"):
                unet.enable_xformers_memory_efficient_attention()
        except Exception as e: logger.warning(f"Could not enable xformers: {e}")
    if args.gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    optimizer_cls = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError: logger.error("bitsandbytes not found for 8-bit Adam.")
    optimizer = optimizer_cls(unet.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    logger.info(f"Loading training filenames from: {args.csv_data_list} with base dir: {args.embedding_base_dir}")
    train_files_dicts = load_filenames_flexible_target(args.csv_data_list, args.embedding_base_dir)
    if not train_files_dicts: 
        raise ValueError(f"No valid file pairs found from {args.csv_data_list} and {args.embedding_base_dir}. Please check paths and CSV content.")
    logger.info(f"Found {len(train_files_dicts)} unique file pairs for flexible target training.")

    train_dataloader = prepare_data_flexible_target(args=args, train_files_dicts=train_files_dicts, is_distributed=(accelerator.num_processes > 1), local_rank=accelerator.local_process_index)
    logger.info(f"Train Dataloader prepared. Batches on this process: {len(train_dataloader)}")

    scale_factor = checkpoint.get("scale_factor", None)
    if scale_factor is not None:
        scale_factor = torch.tensor(scale_factor, device=accelerator.device) if not isinstance(scale_factor, torch.Tensor) else scale_factor.to(accelerator.device)
        logger.info(f"Loaded scale_factor from checkpoint: {scale_factor.item()}")
    else:
        logger.info("Calculating scale_factor from dataset...")
        scale_factor = calculate_scale_factor(args, train_dataloader, accelerator.device, logger)
    logger.info(f"Using scale_factor: {scale_factor.item()}")

    len_train_dataloader_local = len(train_dataloader)
    gathered_dataloader_lengths = accelerator.gather(torch.tensor(len_train_dataloader_local, device=accelerator.device))
    num_update_steps_per_epoch = 0
    if accelerator.is_main_process:
        if gathered_dataloader_lengths.numel() > 0:
            num_update_steps_per_epoch = gathered_dataloader_lengths[0].item()
            if num_update_steps_per_epoch == 0 and torch.any(gathered_dataloader_lengths > 0):
                num_update_steps_per_epoch = torch.max(gathered_dataloader_lengths).item()
        num_optimizer_steps_per_epoch = math.ceil(num_update_steps_per_epoch / args.gradient_accumulation_steps)
    num_opt_steps_tensor = torch.tensor([0 if not accelerator.is_main_process else num_optimizer_steps_per_epoch], dtype=torch.long, device=accelerator.device)
    if accelerator.num_processes > 1: torch.distributed.broadcast(num_opt_steps_tensor, src=0)
    num_optimizer_steps_per_epoch = num_opt_steps_tensor[0].item()

    if args.max_train_steps is None:
        if num_optimizer_steps_per_epoch == 0: args.max_train_steps = 0
        else: args.max_train_steps = args.num_train_epochs * num_optimizer_steps_per_epoch
    if num_optimizer_steps_per_epoch > 0: args.num_train_epochs = math.ceil(args.max_train_steps / num_optimizer_steps_per_epoch)
    else: args.num_train_epochs = 0
    
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, num_training_steps=args.max_train_steps * accelerator.num_processes)
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        tracker_config = {k: str(v) if isinstance(v, list) else v for k, v in dict(vars(args)).items() if isinstance(v, (int, float, str, bool, torch.Tensor)) or (isinstance(v, list) and all(isinstance(elem, (int, float, str, bool)) for elem in v))}
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_files_dicts)} (globally)")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. DDP, accum) = {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint) if args.resume_from_checkpoint != "latest" else None
        if not path:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            if dirs: path = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[-1]
        if path:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[-1])
            if num_optimizer_steps_per_epoch > 0 : first_epoch = global_step // num_optimizer_steps_per_epoch
            logger.info(f"Resumed from step {global_step}, epoch {first_epoch}.")
        else:
            logger.info("No checkpoint found. Training from scratch.")

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process, desc="Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        log_loss_dce_primary, log_loss_recon = 0.0, 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Unpack batch. batch["condition_path"] is now loaded tensor.
                condition_latents_loaded = batch["condition_path"].to(weight_dtype)
                dce_target_latents_loaded = batch["dce_target_path"].to(weight_dtype)
                # batch["dce_target_type"] is a list of strings for items in the batch
                batch_dce_target_types = batch["dce_target_type"]
                bsz_single_task_actual = condition_latents_loaded.shape[0]

                # Scale latents
                s_factor = scale_factor.to(condition_latents_loaded.device, dtype=condition_latents_loaded.dtype)
                scaled_condition_latents = condition_latents_loaded * s_factor
                scaled_dce_target_latents = dce_target_latents_loaded * s_factor

                # Prepare combined targets for the two tasks
                # Task 1: Primary DCE Target, Task 2: Condition Reconstruction
                combined_target_latents = torch.cat([
                    scaled_dce_target_latents, 
                    scaled_condition_latents
                ], dim=0)
                
                # Condition for UNet is always the scaled condition_latent (duplicated for each task)
                combined_scaled_condition_for_unet = torch.cat([
                    scaled_condition_latents, 
                    scaled_condition_latents.clone()
                ], dim=0)

                # Prepare metadata
                # Top region index (fixed for both tasks in this setup)
                top_region_tensor_single = torch.tensor(FIXED_TOP_REGION_INDEX_VALUE, device=accelerator.device, dtype=weight_dtype).unsqueeze(0).repeat(bsz_single_task_actual, 1)
                combined_top_region_index = torch.cat([top_region_tensor_single, top_region_tensor_single.clone()], dim=0)

                # Spacing (fixed for both tasks)
                spacing_tensor_single = torch.tensor(FIXED_SPACING_VALUE, device=accelerator.device, dtype=weight_dtype).unsqueeze(0).repeat(bsz_single_task_actual, 1)
                combined_spacing_tensor = torch.cat([spacing_tensor_single, spacing_tensor_single.clone()], dim=0)

                # Bottom region index - dynamic for primary DCE task, fixed for reconstruction task
                bottom_region_primary_dce_list = []
                for i in range(bsz_single_task_actual):
                    target_type = batch_dce_target_types[i]
                    if target_type == "0000":
                        bottom_region_primary_dce_list.append(BOTTOM_REGION_TARGET_TYPE_0000)
                    elif target_type == "0001":
                        bottom_region_primary_dce_list.append(BOTTOM_REGION_TARGET_TYPE_0001)
                    # elif target_type == "0002": # Example if you add more types
                    #    bottom_region_primary_dce_list.append(BOTTOM_REGION_TARGET_TYPE_0002)
                    else:
                        logger.warning(f"Unexpected dce_target_type '{target_type}' in batch item {i}. Using default bottom region.")
                        bottom_region_primary_dce_list.append(DEFAULT_BOTTOM_REGION)
                
                bottom_primary_dce_tensor_batch = torch.tensor(bottom_region_primary_dce_list, device=accelerator.device, dtype=weight_dtype)
                bottom_recon_tensor_batch = torch.tensor(BOTTOM_REGION_RECONSTRUCTION_TASK, device=accelerator.device, dtype=weight_dtype).unsqueeze(0).repeat(bsz_single_task_actual, 1)
                combined_bottom_region_index = torch.cat([bottom_primary_dce_tensor_batch, bottom_recon_tensor_batch], dim=0)
                
                # Diffusion process
                current_combined_batch_size = combined_target_latents.shape[0]
                timesteps = torch.full((current_combined_batch_size,), args.onestep_timestep, device=accelerator.device, dtype=torch.long)
                noise = torch.randn_like(combined_target_latents)
                noisy_combined_latents = noise_scheduler_maisi.add_noise(original_samples=combined_target_latents, noise=noise, timesteps=timesteps).to(weight_dtype)
                
                model_input = torch.cat([combined_scaled_condition_for_unet, noisy_combined_latents], dim=1)
                
                unet_output = unet(
                    x=model_input, timesteps=timesteps,
                    top_region_index_tensor=combined_top_region_index,
                    bottom_region_index_tensor=combined_bottom_region_index,
                    spacing_tensor=combined_spacing_tensor
                )
                model_pred = unet_output.sample if hasattr(unet_output, 'sample') else unet_output

                pred_task_primary_dce = model_pred[:bsz_single_task_actual]
                pred_task_recon = model_pred[bsz_single_task_actual:]

                target_task_primary_dce = combined_target_latents[:bsz_single_task_actual]
                target_task_recon = combined_target_latents[bsz_single_task_actual:]

                loss_primary_dce = F.mse_loss(pred_task_primary_dce.float(), target_task_primary_dce.float(), reduction="mean")
                loss_recon = F.mse_loss(pred_task_recon.float(), target_task_recon.float(), reduction="mean")
                
                loss = loss_primary_dce + loss_recon

                log_loss_dce_primary += accelerator.gather(loss_primary_dce.repeat(bsz_single_task_actual)).mean().item() / args.gradient_accumulation_steps
                log_loss_recon += accelerator.gather(loss_recon.repeat(bsz_single_task_actual)).mean().item() / args.gradient_accumulation_steps
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "total_train_loss": log_loss_dce_primary + log_loss_recon,
                    "train_loss_primary_dce": log_loss_dce_primary,
                    "train_loss_recon": log_loss_recon,
                    "lr": lr_scheduler.get_last_lr()[0]
                }, step=global_step)
                log_loss_dce_primary, log_loss_recon = 0.0, 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        torch.save({
                            "step": global_step, "epoch": epoch,
                            "unet_state_dict": unwrapped_unet.state_dict(),
                            "scale_factor": scale_factor.cpu(), "args": args
                        }, os.path.join(save_path, "unet_onestep_checkpoint.pt"))
                        logger.info(f"Saved checkpoint to {save_path}")

                        if autoencoder is not None:
                            logger.info(f"Generating visualization for step {global_step}...")
                            vis_save_dir = os.path.join(args.output_dir, "visualizations")
                            os.makedirs(vis_save_dir, exist_ok=True)
                            
                            def decode_and_save_vis(latent_tensor, filename_prefix, axis_name, recon_model):
                                try:
                                    decoded_tensor = recon_model(latent_tensor)
                                    img_volume = decoded_tensor[0, 0]
                                    D, H, W = img_volume.shape
                                    if axis_name == "axial": img_slice_2d = img_volume[D // 2, :, :]
                                    elif axis_name == "sagittal": img_slice_2d = img_volume[:, :, W // 2]
                                    elif axis_name == "coronal": img_slice_2d = img_volume[:, H // 2, :]
                                    else: return False
                                    normalized_slice = (img_slice_2d - torch.min(img_slice_2d)) / (torch.max(img_slice_2d) - torch.min(img_slice_2d) + 1e-5)
                                    vis_filename = os.path.join(vis_save_dir, f"step_{global_step}_{filename_prefix}_{axis_name}.png")
                                    torchvision.utils.save_image(normalized_slice, vis_filename)
                                    return True
                                except Exception as e:
                                    logger.error(f"Vis error ({filename_prefix}, {axis_name}): {e}", exc_info=True)
                                    return False

                            autoencoder.to(accelerator.device, dtype=weight_dtype).eval()
                            class ReconModelVis(nn.Module):
                                def __init__(self, ae, sf): super().__init__(); self.autoencoder = ae; self.scale_factor = sf
                                def forward(self, z): return self.autoencoder.decode_stage_2_outputs((z / self.scale_factor).to(dtype=weight_dtype))
                            recon_model_vis = ReconModelVis(autoencoder, scale_factor).to(accelerator.device, dtype=weight_dtype).eval()

                            with torch.no_grad():
                                latents_to_vis = {
                                    "pred_primary_dce": pred_task_primary_dce[0].unsqueeze(0),
                                    "pred_recon": pred_task_recon[0].unsqueeze(0),
                                    "gt_primary_dce": target_task_primary_dce[0].unsqueeze(0),
                                    "gt_recon_orig": target_task_recon[0].unsqueeze(0) 
                                }
                                axes_to_vis = ["axial", "sagittal", "coronal"]
                                for prefix, latent in latents_to_vis.items():
                                    for axis in axes_to_vis:
                                        decode_and_save_vis(latent, prefix, axis, recon_model_vis)
                            autoencoder.cpu()
                            logger.info(f"Visualizations saved for step {global_step}.")
            
            if global_step >= args.max_train_steps: break
        if global_step >= args.max_train_steps: break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        final_save_path = os.path.join(args.output_dir, "final_unet_onestep.pt")
        torch.save({
            "step": global_step, "epoch": epoch,
            "unet_state_dict": unwrapped_unet.state_dict(),
            "scale_factor": scale_factor.cpu(), "args": args
        }, final_save_path)
        logger.info(f"Saved final model to {final_save_path}")
    accelerator.end_training()
    logger.info("Training finished.")

if __name__ == "__main__":
    args = parse_args()
    main(args) 