#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import random
import datetime
import logging
import csv # For loading test data list

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

from monai.utils import set_determinism
from monai.data import DataLoader, CacheDataset # Reusing MONAI data loading
from monai.transforms import Compose, Lambdad

# MAISI imports (ensure these paths are correct relative to your execution context)
from scripts.diff_model_setting import initialize_distributed, load_config, setup_logging
from scripts.sample import ReconModel, check_input
from scripts.utils import define_instance

# Basic logging setup
logger = logging.getLogger(__name__)

def set_random_seed(seed):
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed

def load_filenames_test(csv_data_list: str) -> list:
    """
    Load filenames from a test CSV file.
    Expects at least a 'condition_path' column.
    Returns a list of dictionaries, e.g.,
    [{"condition_path": "relative/path/cond1.npy", "id": "cond1"}, ...]
    Uses the basename of the condition path as an ID if 'id' column is not present.
    """
    paths_list = []
    seen_ids = set()
    if not os.path.exists(csv_data_list):
        raise FileNotFoundError(f"Test CSV file not found: {csv_data_list}")
        
    with open(csv_data_list, newline="") as f:
        reader = csv.DictReader(f)
        if "input_path" not in reader.fieldnames:
            raise ValueError("CSV file must contain 'input_path' column.")
            
        has_id_col = "id" in reader.fieldnames
        
        for row in reader:
            condition_path = row["input_path"]
            
            if has_id_col:
                sample_id = row["id"]
            else:
                # Generate ID from filename if 'id' column is missing
                sample_id = os.path.splitext(os.path.basename(condition_path))[0]
            
            if sample_id not in seen_ids:
                paths_list.append({"condition_path": condition_path, "id": sample_id})
                seen_ids.add(sample_id)
            else:
                 logger.warning(f"Duplicate ID '{sample_id}' found in CSV. Skipping.")
                 
    if not paths_list:
         logger.warning(f"No valid entries found or loaded from {csv_data_list}")
         
    return paths_list

def load_and_squeeze_leading_dim(x):
    """Loads .npy file and removes leading singleton dimension if present."""
    if not isinstance(x, str) or not os.path.exists(x):
        raise FileNotFoundError(f"Input file not found: {x}")
    data = np.load(x)
    if data.shape[0] == 1:
        logger.debug(f"Squeezing leading dimension from shape {data.shape} for file {x}")
        data = np.squeeze(data, axis=0) # Remove leading singleton dimension
    # Expected shape now: (C, D, H, W)
    return data

def prepare_test_data(
    args: argparse.Namespace,
    test_files_dicts: list # List of dicts like [{"condition_path": "path/cond.npy", "id": "sample1"}, ...]
) -> DataLoader:
    """
    Prepare test data loader for one-step inference. Loads only condition latents.
    """
    test_transforms = Compose([
        Lambdad(keys=["condition"], func=load_and_squeeze_leading_dim),
        # We don't need metadata like spacing/region index in the dataloader itself
        # as they are fixed and will be prepared separately.
    ])

    # No partitioning needed for inference unless running distributed inference
    # For simplicity, assume non-distributed inference first.
    
    # Check if dataset list is empty
    if not test_files_dicts:
        logger.warning("Test file list is empty. Returning an empty DataLoader.")
        return DataLoader([], batch_size=args.test_batch_size)

    test_ds = CacheDataset(
        data=test_files_dicts, 
        transform=test_transforms, 
        cache_rate=0.0, # No caching needed for inference usually
        num_workers=args.dataloader_num_workers
    )
    
    if not test_ds:
        logger.warning("CacheDataset for test data is empty.")
        return DataLoader([], batch_size=args.test_batch_size)

    return DataLoader(
        test_ds, 
        num_workers=args.dataloader_num_workers, 
        batch_size=args.test_batch_size, 
        shuffle=False # No shuffling for inference
    )


def load_models(args, device, logger):
    """Loads the VAE and the fine-tuned One-Step UNet model."""
    # Load VAE (autoencoder)
    logger.info(f"Loading VAE from: {args.trained_autoencoder_path}")
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ckpt_vae = torch.load(args.trained_autoencoder_path, map_location=device) 
    # Adapt key if necessary, assume state_dict is top level or under a common key
    vae_state_dict = ckpt_vae.get("state_dict", ckpt_vae) 
    autoencoder.load_state_dict(vae_state_dict)
    logger.info("VAE loaded successfully.")

    # Load the fine-tuned One-Step UNet
    unet_ckpt_path = os.path.join(args.model_dir, args.model_filename)
    logger.info(f"Loading One-Step UNet from: {unet_ckpt_path}")
    unet = define_instance(args, "diffusion_unet_def").to(device) # Defines the base UNet architecture
    ckpt_unet = torch.load(unet_ckpt_path, map_location=device)
    
    # --- Crucial Part: Modify UNet architecture if needed BEFORE loading state_dict ---
    # This modification MUST match how it was done in train_maisi_onestep.py
    # Find the original conv_in layer and its parameters
    original_conv_in = None
    conv_in_attribute_path = None # To store how to access the layer e.g., unet.conv_in or unet.conv_in[0]
    
    if hasattr(unet, 'conv_in'):
        if isinstance(unet.conv_in, torch.nn.Conv3d):
            original_conv_in = unet.conv_in
            conv_in_attribute_path = ('conv_in',)
        elif isinstance(unet.conv_in, torch.nn.Sequential) and len(unet.conv_in) > 0 and isinstance(unet.conv_in[0], torch.nn.Conv3d):
             original_conv_in = unet.conv_in[0] 
             conv_in_attribute_path = ('conv_in', 0)
             logger.info("Detected conv_in wrapped in nn.Sequential. Will modify conv_in[0].")
        else:
            logger.error(f"unet.conv_in found but type {type(unet.conv_in)} is not directly modifiable by this script.")
    else:
        logger.error("Could not find 'unet.conv_in'. Cannot modify input channels.")

    if original_conv_in is not None:
        original_in_channels = original_conv_in.in_channels
        original_out_channels = original_conv_in.out_channels
        kernel_size = original_conv_in.kernel_size
        stride = original_conv_in.stride
        padding = original_conv_in.padding
        dilation = original_conv_in.dilation
        groups = original_conv_in.groups
        has_bias = original_conv_in.bias is not None
        padding_mode = original_conv_in.padding_mode
        
        new_in_channels = original_in_channels * 2 # Based on train_maisi_onestep.py
        logger.info(f"Modifying UNet definition for inference: original_in_channels={original_in_channels}, new_in_channels={new_in_channels}")

        # Create the new Conv3D layer definition
        new_conv_in_layer = torch.nn.Conv3d(
            in_channels=new_in_channels,
            out_channels=original_out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=has_bias, padding_mode=padding_mode
        )
        
        # Replace the layer in the UNet definition *before* loading state dict
        if len(conv_in_attribute_path) == 1:
            setattr(unet, conv_in_attribute_path[0], new_conv_in_layer)
            logger.info(f"Replaced unet.{conv_in_attribute_path[0]} definition.")
        elif len(conv_in_attribute_path) == 2:
            parent_module = getattr(unet, conv_in_attribute_path[0])
            parent_module[conv_in_attribute_path[1]] = new_conv_in_layer
            logger.info(f"Replaced unet.{conv_in_attribute_path[0]}[{conv_in_attribute_path[1]}] definition.")
            
    else:
         logger.error("UNet input layer not modified. State dict loading might fail if checkpoint expects modified layer.")

    # Now load the state dict into the (potentially modified) UNet architecture
    unet_state_dict = ckpt_unet.get("unet_state_dict", ckpt_unet.get("state_dict", ckpt_unet))
    # Handle 'module.' prefix if saved from DDP
    if any(key.startswith("module.") for key in unet_state_dict.keys()):
        logger.info("Removing 'module.' prefix from UNet state_dict keys.")
        unet_state_dict = {k.replace("module.", ""): v for k, v in unet_state_dict.items()}
        
    load_result = unet.load_state_dict(unet_state_dict, strict=True)
    logger.info(f"UNet state_dict loaded. Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")
    
    # Load scale factor
    if "scale_factor" in ckpt_unet:
        scale_factor = ckpt_unet["scale_factor"]
        if not isinstance(scale_factor, torch.Tensor):
            scale_factor = torch.tensor(scale_factor, device=device)
        else:
            scale_factor = scale_factor.to(device)
        logger.info(f"Loaded scale_factor from UNet checkpoint: {scale_factor.item()}")
    else:
        # Try loading from VAE checkpoint as fallback? Or require it in UNet checkpoint.
        logger.warning("Scale factor not found in UNet checkpoint. Attempting to load from VAE checkpoint.")
        scale_factor = ckpt_vae.get("scale_factor", torch.tensor(1.0, device=device)) # Default to 1.0 if not found anywhere
        logger.info(f"Using scale_factor: {scale_factor.item()} (loaded from VAE ckpt or default)")

    return autoencoder, unet, scale_factor


def prepare_metadata_tensors(args, batch_size, device, dtype):
    """Prepares fixed metadata tensors for the batch."""
    # Use metadata values intended for the target task (e.g., DCE prediction)
    # These values should ideally come from config or args
    # Using placeholders matching train_maisi_onestep.py for now
    # TODO: Make these configurable via args or loaded config
    fixed_top_region_index_value = [0, 0, 1, 0] 
    # Use DCE task bottom region index
    bottom_region_index_dce_task_value = [0, 0, 1, 0] 
    fixed_spacing_value = [1.0, 1.0, 1.2] 

    top_tensor = torch.tensor(fixed_top_region_index_value, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
    bottom_tensor = torch.tensor(bottom_region_index_dce_task_value, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
    spacing_tensor = torch.tensor(fixed_spacing_value, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
    
    return top_tensor, bottom_tensor, spacing_tensor

def save_image(data, out_spacing, output_path):
    """Saves image data as NIfTI file."""
    affine = np.eye(4)
    # Use output spacing from config if available, otherwise default or assume 1.0
    if out_spacing and len(out_spacing) == 3:
        for i in range(3):
            affine[i, i] = out_spacing[i]
    else:
        logger.warning(f"Output spacing '{out_spacing}' is invalid. Using identity affine.")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), output_path)
    logger.info(f"Saved image to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="One-Step Diffusion Model Inference using MAISI models")
    # MAISI Configs
    parser.add_argument("--env_config", type=str, default="configs/environment_maisi_diff_model.json")
    parser.add_argument("--model_config", type=str, default="configs/config_maisi_diff_model.json")
    parser.add_argument("--model_def", type=str, default="configs/config_maisi.json")
    # Model Paths
    parser.add_argument("--trained_autoencoder_path", type=str, default="models/autoencoder_epoch273.pt",
                        help="Path to the trained autoencoder (VAE) checkpoint.")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory where the fine-tuned one-step UNet checkpoint is saved.")
    parser.add_argument("--model_filename", type=str, required=True, # Make UNet checkpoint required
                        help="Filename for the fine-tuned one-step UNet checkpoint.")
    # Data
    parser.add_argument("--test_csv_data_list", type=str, required=True,
                        help="Path to CSV file containing test data paths ('condition_path' column required).")
    parser.add_argument("--condition_latent_base_dir", type=str, required=True,
                        help="Base directory for the VAE encoded condition npy files listed in the CSV.")
    # Inference Parameters
    parser.add_argument("--onestep_timestep", type=int, default=1,
                        help="The single timestep 't' to use for the UNet prediction (should match training).")
    parser.add_argument("--latent_channels", type=int, default=4,
                        help="Number of latent channels expected by VAE/UNet (default: 4).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for noise generation.")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
                        help="Whether to use mixed precision ('fp16', 'bf16', or 'no').")
    # Output
    parser.add_argument("--output_dir", type=str, default="results_onestep",
                        help="Directory to save the synthetic images.")
    # Others
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs (currently inference is non-distributed).")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Num workers for data loading.")
    
    args_cli = parser.parse_args()
    
    # Load config files and merge CLI args
    args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)
    # Manually override config values with CLI arguments where provided
    args.trained_autoencoder_path = args_cli.trained_autoencoder_path
    args.model_dir = args_cli.model_dir
    args.model_filename = args_cli.model_filename
    args.test_csv_data_list = args_cli.test_csv_data_list
    args.condition_latent_base_dir = args_cli.condition_latent_base_dir
    args.onestep_timestep = args_cli.onestep_timestep
    args.latent_channels = args_cli.latent_channels # Ensure this matches model def
    args.seed = args_cli.seed
    args.test_batch_size = args_cli.test_batch_size
    args.mixed_precision = args_cli.mixed_precision
    args.output_dir = args_cli.output_dir
    args.num_gpus = args_cli.num_gpus
    args.dataloader_num_workers = args_cli.dataloader_num_workers

    # Add latent_channels to args if not present from configs
    if not hasattr(args, 'latent_channels'):
         args.latent_channels = args_cli.latent_channels

    return args

def main():
    args = parse_args()
    
    # Setup logging, device, seed
    # Note: Distributed setup might be overkill if only running on one GPU, but keep for consistency
    local_rank, world_size, device = initialize_distributed(args.num_gpus) 
    global logger # Make logger global after setup
    logger = setup_logging("one_step_inference")
    seed = set_random_seed(args.seed)
    logger.info(f"Running on device: {device} with seed {seed}")
    logger.info(f"Output directory set to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine dtype for mixed precision
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    logger.info(f"Using mixed precision: {args.mixed_precision} ({dtype})")
    
    # Load models (VAE and modified UNet)
    autoencoder, unet, scale_factor = load_models(args, device, logger)

    # <<< Cast models to the target dtype AND device for mixed precision >>>
    if args.mixed_precision != "no":
        logger.info(f"Casting UNet and Autoencoder to device='{device}', dtype={dtype}...")
        unet = unet.to(device=device, dtype=dtype)
        autoencoder = autoencoder.to(device=device, dtype=dtype)
    else:
        # If not using mixed precision, still ensure models are on the correct device
        logger.info(f"Moving UNet and Autoencoder to device='{device}'...")
        unet = unet.to(device=device)
        autoencoder = autoencoder.to(device=device)

    autoencoder.eval()
    unet.eval()

    # Prepare VAE decoder wrapper (after autoencoder is on correct device/dtype)
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device).eval()

    # Prepare Data Loader
    logger.info(f"Loading test file list from: {args.test_csv_data_list}")
    test_filenames_relative = load_filenames_test(args.test_csv_data_list)
    
    test_files_dicts = []
    for file_pair_dict in test_filenames_relative:
        full_cond_path = os.path.join(args.condition_latent_base_dir, file_pair_dict["condition_path"])
        if os.path.exists(full_cond_path):
            test_files_dicts.append({
                "condition": full_cond_path, # Key for condition latent path
                "id": file_pair_dict["id"]    # ID for saving output file
            })
        else:
            logger.warning(f"Condition latent file not found and skipped: {full_cond_path}")

    if not test_files_dicts:
        logger.error("No valid test files found. Exiting.")
        return
        
    logger.info(f"Prepared {len(test_files_dicts)} test samples.")
    test_dataloader = prepare_test_data(args, test_files_dicts)

    # Inference loop
    # Output spacing from config for saving NIfTI
    output_spacing = tuple(args.diffusion_unet_inference.get("spacing", (1.0, 1.0, 1.0))) 
    target_timestep = torch.tensor([args.onestep_timestep], device=device) # Prepare timestep tensor once

    with torch.no_grad():
        autocast_context = torch.autocast(device_type=device.type, dtype=dtype, enabled=(args.mixed_precision != "no"))
        with autocast_context:
            for batch in tqdm(test_dataloader, desc="Running One-Step Inference"):
                raw_condition_latents = batch["condition"] # This is likely a MONAI MetaTensor
                ids = batch["id"] # List of IDs in the batch
                current_batch_size = raw_condition_latents.shape[0] # Get batch size from raw tensor

                # <<< Convert MetaTensor to plain torch.Tensor and ensure correct device/dtype >>>
                if hasattr(raw_condition_latents, 'array'): # MONAI MetaTensor stores data in .array
                    plain_tensor_data = raw_condition_latents.array
                elif hasattr(raw_condition_latents, 'as_tensor'): # Fallback for other tensor-like objects
                    plain_tensor_data = raw_condition_latents.as_tensor()
                else: # Assume it's already a torch.Tensor or NumPy array
                    plain_tensor_data = raw_condition_latents
                
                # Create a new, plain torch.Tensor with the correct device and dtype
                condition_latents = torch.as_tensor(plain_tensor_data, device=device, dtype=dtype)

                # Scale condition latents
                # Ensure scale_factor is also on the correct device and dtype before multiplication
                scaled_condition_latents = condition_latents * scale_factor.to(device=device, dtype=dtype)

                # Sample noise (same shape as condition latents)
                # Noise should also be a plain tensor on the correct device and dtype
                noise = torch.randn_like(condition_latents, device=device, dtype=dtype)

                # Prepare UNet input: [Condition Latent, Noise]
                # All components of model_input are now plain torch.Tensors with correct properties
                model_input = torch.cat([scaled_condition_latents, noise], dim=1)

                # Prepare metadata tensors for the batch
                top_tensor, bottom_tensor, spacing_tensor = prepare_metadata_tensors(args, current_batch_size, device, dtype)
                
                # <<< Sanity check for conv_in layer device and dtype >>>
                try:
                    conv_in_layer_to_check = unet.conv_in[0] if isinstance(unet.conv_in, torch.nn.Sequential) else unet.conv_in
                    logger.debug(f"UNet conv_in weight device: {conv_in_layer_to_check.weight.device}, dtype: {conv_in_layer_to_check.weight.dtype}")
                except Exception as e:
                    logger.debug(f"Could not check conv_in layer details: {e}")
                
                # Run the UNet
                # Ensure timesteps tensor is correctly shaped if model expects (B,)
                timesteps_batch = target_timestep.repeat(current_batch_size) 
                
                unet_output = unet(
                    x=model_input,
                    timesteps=timesteps_batch,
                    top_region_index_tensor=top_tensor,
                    bottom_region_index_tensor=bottom_tensor, 
                    spacing_tensor=spacing_tensor
                )
                # print(unet_output.shape)
                # print(torch.mean(unet_output), torch.std(unet_output))
                # Assuming UNet output is the predicted x0 latent, based on training loss
                pred_x0_latent = unet_output.sample if hasattr(unet_output, 'sample') else unet_output
                # print(pred_x0_latent.shape)
                # print(torch.mean(pred_x0_latent), torch.std(pred_x0_latent))
                # <<< Add Debugging for pred_x0_latent >>>
                if torch.isnan(pred_x0_latent).any():
                    logger.warning(f"NaN detected in pred_x0_latent for batch starting with ID: {ids[0]}")
                elif torch.isinf(pred_x0_latent).any():
                    logger.warning(f"Inf detected in pred_x0_latent for batch starting with ID: {ids[0]}")
                else:
                    logger.debug(
                        f"pred_x0_latent stats (before VAE decode) - Min: {torch.min(pred_x0_latent):.4f}, "
                        f"Max: {torch.max(pred_x0_latent):.4f}, "
                        f"Mean: {torch.mean(pred_x0_latent):.4f}, "
                        f"Std: {torch.std(pred_x0_latent):.4f}"
                    )
                # <<< End Debugging >>>

                # Decode the predicted latent using the VAE wrapper
                # recon_model handles the scaling factor internally
                decoded_images = recon_model(pred_x0_latent) # Output shape should be [B, C_img, D, H, W]
                # print(decoded_images.shape)
                # print(torch.mean(decoded_images), torch.std(decoded_images))
                # <<< Add Debugging for decoded_images (before numpy conversion) >>>
                if torch.is_tensor(decoded_images):
                    logger.debug(
                        f"decoded_images (from recon_model) stats - Min: {torch.min(decoded_images):.4f}, "
                        f"Max: {torch.max(decoded_images):.4f}, "
                        f"Mean: {torch.mean(decoded_images):.4f}, "
                        f"Std: {torch.std(decoded_images):.4f}, "
                        f"dtype: {decoded_images.dtype}"
                    )
                else:
                    logger.warning(f"decoded_images is not a tensor. Type: {type(decoded_images)}")
                # <<< End Debugging >>>
                
                # Move to CPU, convert to numpy, handle channel dim, save each image in batch
                decoded_images_np = decoded_images.cpu().float().numpy()

                for i in range(current_batch_size):
                    img_id = ids[i]
                    # Assuming output is [C, D, H, W], squeeze C if it's 1, otherwise handle multi-channel output
                    output_data = decoded_images_np[i]
                    if output_data.shape[0] == 1: # Single channel image
                         output_data = np.squeeze(output_data, axis=0) 
                    else: # Multi-channel, save as is or decide how to handle
                         logger.warning(f"Output for {img_id} has multiple channels ({output_data.shape[0]}). Saving as is.")
                         # NIfTI typically expects [X, Y, Z] or [X, Y, Z, T/C]. Adjust if needed.
                         # Example: move channel to last dim: output_data = np.moveaxis(output_data, 0, -1)
                         
                    output_filename = f"{img_id}_onestep_t{args.onestep_timestep}.nii.gz"
                    output_path = os.path.join(args.output_dir, output_filename)
                    save_image(output_data, output_spacing, output_path)

    logger.info("One-step inference finished.")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main() 