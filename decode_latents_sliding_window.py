#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import glob
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm
import torch.multiprocessing as mp
from math import ceil
import logging

from monai.inferers import sliding_window_inference
# Assuming MAISI 'scripts' directory is in PYTHONPATH or accessible
# e.g., export PYTHONPATH=/path/to/Lotus:$PYTHONPATH if this script is in Lotus/
try:
    from scripts.diff_model_setting import load_config
    from scripts.sample import ReconModel
    from scripts.utils import define_instance
except ImportError as e:
    print("Failed to import MAISI scripts. Ensure 'scripts' dir is in PYTHONPATH.")
    print("Example: If your project root is 'Lotus', run: export PYTHONPATH='$PWD/Lotus:$PYTHONPATH' from outside Lotus dir, or export PYTHONPATH='$PWD:$PYTHONPATH' from Lotus dir.")
    raise e

# Basic logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_autoencoder_model(device, model_def_config_path, autoencoder_checkpoint_path):
    """Loads the VAE model using MAISI's define_instance and config system."""
    logger.info(f"Loading MAISI model definition from: {model_def_config_path}")
    # load_config expects env_config, model_config, model_def paths.
    # For autoencoder_def, it's usually in the model_def file (e.g., config_maisi.json).
    # We pass model_def_config_path for all three as a simplification,
    # assuming define_instance will find 'autoencoder_def' within it.
    config_args = load_config(model_def_config_path, model_def_config_path, model_def_config_path)
    
    autoencoder = define_instance(config_args, "autoencoder_def").to(device)
    
    logger.info(f"Loading VAE checkpoint from: {autoencoder_checkpoint_path}")
    ckpt = torch.load(autoencoder_checkpoint_path, map_location=device)
    
    vae_state_dict = ckpt.get("state_dict", ckpt)
    autoencoder.load_state_dict(vae_state_dict)
    autoencoder.eval()
    logger.info("VAE model loaded successfully.")
    return autoencoder, config_args

def get_output_affine_from_config(config_args):
    """Tries to extract output spacing from config_args to create an affine matrix."""
    affine = np.eye(4) # Default identity affine
    try:
        # Attempt to get spacing, path might vary based on MAISI config structure
        # This example path is based on infer_onestep.py's usage
        # args.diffusion_unet_inference.get("spacing", (1.0, 1.0, 1.0)))
        # Check if 'diffusion_unet_inference' attribute and then 'spacing' key exist
        if hasattr(config_args, 'diffusion_unet_inference') and isinstance(config_args.diffusion_unet_inference, dict):
            spacing = config_args.diffusion_unet_inference.get("spacing", (1.0, 1.0, 1.0))
        elif hasattr(config_args, 'autoencoder_def') and 'spacing' in config_args.autoencoder_def: # Another common place
             spacing = config_args.autoencoder_def['spacing']
        else: # Fallback to a common default or check other potential config locations
            spacing = (1.0,1.0,1.0) # Default spacing
            logger.info("Spacing not found in common config locations (e.g., config_args.diffusion_unet_inference.spacing or config_args.autoencoder_def.spacing). Using default 1.0 for spacing.")


        if spacing and len(spacing) == 3:
            for i in range(3):
                affine[i, i] = float(spacing[i])
            logger.info(f"Using output spacing from config: {spacing} for NIfTI affine.")
        else:
            logger.warning(f"Could not get valid 3D spacing from config (found: {spacing}). Using identity affine.")
    except Exception as e:
        logger.warning(f"Error extracting spacing from config: {e}. Using identity affine.")
    return affine

def save_nifti(data, output_path, affine):
    """Saves data as NIfTI file with given affine."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), output_path)
    logger.info(f"Saved NIfTI image to {output_path}")

def worker_process(latent_files_chunk, process_id, args):
    device_idx = args.gpu_ids[process_id % len(args.gpu_ids)] # Assign GPU from the list
    device = torch.device(f"cuda:{device_idx}")
    logger.info(f"Worker process {process_id} assigned to GPU cuda:{device_idx}")

    try:
        autoencoder, config_args_for_metadata = load_autoencoder_model(device, args.model_def_config, args.trained_autoencoder_path)
        
        scale_factor_tensor = torch.tensor(args.scale_factor, device=device)
        recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor_tensor).to(device)
        recon_model.eval()

        output_affine = get_output_affine_from_config(config_args_for_metadata)

        dtype_decode = torch.float32
        if args.mixed_precision_decode == "fp16":
            dtype_decode = torch.float16
            recon_model = recon_model.to(dtype=dtype_decode)
        elif args.mixed_precision_decode == "bf16":
            dtype_decode = torch.bfloat16
            recon_model = recon_model.to(dtype=dtype_decode)
        logger.info(f"Worker {process_id} on GPU {device_idx}: Using mixed precision {args.mixed_precision_decode} (dtype: {dtype_decode})")


        for latent_file_path in tqdm(latent_files_chunk, desc=f"GPU {device_idx} decoding", position=process_id):
            try:
                latent_data_np = np.load(latent_file_path) # Expected shape: [C, D, H, W]
                latent_tensor = torch.from_numpy(latent_data_np).to(device=device, dtype=dtype_decode).unsqueeze(0) # Add batch_dim [1, C, D, H, W]

                with torch.no_grad():
                    autocast_enabled = (args.mixed_precision_decode != "no")
                    with torch.autocast(device_type=device.type, dtype=dtype_decode if autocast_enabled else None, enabled=autocast_enabled):
                        decoded_image_tensor = sliding_window_inference(
                            inputs=latent_tensor,
                            roi_size=tuple(args.roi_size),
                            sw_batch_size=args.sw_batch_size,
                            predictor=recon_model, # Pass the ReconModel instance
                            overlap=args.overlap,
                            mode="gaussian",
                            progress=False # tqdm is in outer loop per GPU
                        )
                
                output_data_np = decoded_image_tensor.squeeze(0).cpu().float().numpy() # Back to float32 for saving
                
                if output_data_np.shape[0] == 1: # Single channel image [D, H, W]
                    output_data_np = np.squeeze(output_data_np, axis=0)
                else: # Multi-channel image [C, D, H, W]
                    logger.info(f"Output for {os.path.basename(latent_file_path)} has {output_data_np.shape[0]} channels. Saving as is.")

                base_name = os.path.splitext(os.path.basename(latent_file_path))[0]
                output_nifti_filename = f"{base_name}.nii.gz"
                output_nifti_path = os.path.join(args.output_dir, output_nifti_filename)
                
                save_nifti(output_data_np, output_nifti_path, affine=output_affine)

            except Exception as e:
                logger.error(f"GPU {device_idx} failed to process {latent_file_path}: {e}", exc_info=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error in worker_process {process_id} on GPU {device_idx}: {e}", exc_info=True)


def main_decode():
    parser = argparse.ArgumentParser(description="Decode .npy latents to NIfTI using Sliding Window Inference and VAE.")
    parser.add_argument("--input_latent_dir", type=str, required=True, help="Directory containing .npy latent files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save decoded NIfTI images.")
    parser.add_argument("--trained_autoencoder_path", type=str, required=True, help="Path to the trained VAE checkpoint (.pt).")
    parser.add_argument("--model_def_config", type=str, required=True, help="Path to MAISI model definition JSON (e.g., configs/config_maisi.json for autoencoder_def).")
    parser.add_argument("--scale_factor", type=float, default=1.0317915678024292, 
                        help="Scale factor for ReconModel, should be consistent with how latents were generated/scaled. (Default from maisi/decode_pipeline.py)")
    
    # Sliding window parameters
    parser.add_argument("--roi_size", type=int, nargs=3, default=[64, 64, 64], help="ROI size for sliding window (D H W).")
    parser.add_argument("--sw_batch_size", type=int, default=1, help="Batch size for sliding window.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio for sliding window (0.0 to 1.0).")

    # GPU and processing
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). Uses these GPUs for multiprocessing.")
    parser.add_argument("--mixed_precision_decode", type=str, default="no", choices=["no", "fp16", "bf16"], 
                        help="Mixed precision ('fp16', 'bf16', 'no') for VAE decoding stage.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        args.gpu_ids = [int(gid.strip()) for gid in args.gpu_ids.split(',') if gid.strip()]
        if not args.gpu_ids:
            raise ValueError("No GPU IDs provided or all are invalid.")
        if not all(gid >= 0 for gid in args.gpu_ids):
            raise ValueError("GPU IDs must be non-negative integers.")
    except ValueError as e:
        logger.error(f"Invalid GPU IDs: {args.gpu_ids}. Error: {e}")
        return
        
    num_available_gpus = torch.cuda.device_count()
    for gid in args.gpu_ids:
        if gid >= num_available_gpus:
            logger.error(f"GPU ID {gid} is not available. Available GPUs: {num_available_gpus}.")
            return
    
    num_gpus_to_use = len(args.gpu_ids)
    logger.info(f"Using {num_gpus_to_use} GPU(s): {args.gpu_ids}")

    npy_files = sorted(glob.glob(os.path.join(args.input_latent_dir, "*.npy")))
    if not npy_files:
        logger.error(f"No .npy files found in {args.input_latent_dir}")
        return
    logger.info(f"Found {len(npy_files)} .npy files to process.")

    if num_gpus_to_use == 1:
        logger.info(f"Running decoding on a single GPU: cuda:{args.gpu_ids[0]}")
        worker_process(npy_files, 0, args) # Process ID 0 for single GPU mode
    else:
        logger.info(f"Distributing decoding over {num_gpus_to_use} GPUs: {args.gpu_ids}")
        # mp.set_start_method("spawn", force=True) # Set start method for multiprocessing

        files_per_process = ceil(len(npy_files) / num_gpus_to_use)
        processes = []
        for i in range(num_gpus_to_use):
            start_idx = i * files_per_process
            end_idx = start_idx + files_per_process
            files_chunk = npy_files[start_idx:end_idx]
            if not files_chunk:
                continue
            
            # 'i' is the process_id, used to pick from args.gpu_ids
            p = mp.Process(target=worker_process, args=(files_chunk, i, args))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

    logger.info("Decoding finished.")

if __name__ == "__main__":
    # It's generally recommended to set the start method for multiprocessing
    # if you plan to use CUDA in subprocesses, especially on some OSes.
    # 'spawn' is often the most robust.
    # Do this before any CUDA context is created in the parent process if possible.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing start method to spawn: {e}. Using default.")
    main_decode() 