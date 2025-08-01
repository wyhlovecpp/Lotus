#!/usr/bin/env python
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

import monai
from monai.data import DataLoader, partition_dataset, CacheDataset
from monai.transforms import Compose, Lambdad, EnsureChannelFirstd
from monai.utils import first

from scripts.diff_model_setting import initialize_distributed, load_config, setup_logging
from scripts.utils import define_instance


def load_filenames(csv_data_list: str) -> list:
    """
    Load unique filenames from a CSV file.
    Each row should have columns: input_path and target_path.
    Returns a list of unique relative file paths.
    """
    paths = set()
    with open(csv_data_list, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.add(row["input_path"])
            paths.add(row["target_path"])
    return list(paths)

def load_and_squeeze(x):
    data = np.load(x) if isinstance(x, str) else x
    # 如果第一维是1，则去掉这一维
    if data.shape[0] == 1:
        data = np.squeeze(data, axis=0)
    return data

def prepare_data(
    train_files: list, device: torch.device, cache_rate: float, num_workers: int = 2, batch_size: int = 1
) -> DataLoader:
    """
    Prepare training data.
    Loads npy files and attaches fixed metadata.
    """
    train_transforms = Compose([
        Lambdad(keys=["image"], func=load_and_squeeze),
        Lambdad(keys="top_region_index", func=lambda _: [0, 0, 1, 0]),
        Lambdad(keys="bottom_region_index", func=lambda _: [0, 0, 1, 0]),
        Lambdad(keys="spacing", func=lambda _: [1.0, 1.0, 1.2]),
    ])


    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(train_ds, num_workers=6, batch_size=batch_size, shuffle=True)


def load_unet(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> torch.nn.Module:
    """
    Load the UNet model.
    """
    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    if dist.is_initialized():
        unet = DistributedDataParallel(unet, device_ids=[device], find_unused_parameters=True)

    if args.existing_ckpt_filepath is None:
        logger.info("Training from scratch.")
    else:
        checkpoint_unet = torch.load(args.existing_ckpt_filepath, map_location=device)
        if dist.is_initialized():
            unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        else:
            unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        logger.info(f"Pretrained checkpoint {args.existing_ckpt_filepath} loaded.")

    return unet


def calculate_scale_factor(train_loader: DataLoader, device: torch.device, logger: logging.Logger) -> torch.Tensor:
    """
    Calculate the scaling factor for the dataset.
    """
    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
    logger.info(f"Scaling factor set to {scale_factor}.")

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    logger.info(f"scale_factor -> {scale_factor}.")
    return scale_factor


def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Create optimizer for training.
    """
    return torch.optim.Adam(params=model.parameters(), lr=lr)


def create_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.PolynomialLR:
    """
    Create learning rate scheduler.
    """
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)


def train_one_epoch(
    epoch: int,
    unet: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
    loss_pt: torch.nn.L1Loss,
    scaler: GradScaler,
    scale_factor: torch.Tensor,
    noise_scheduler: torch.nn.Module,
    num_images_per_batch: int,
    num_train_timesteps: int,
    device: torch.device,
    logger: logging.Logger,
    local_rank: int,
    amp: bool = True,
) -> torch.Tensor:
    """
    Train the model for one epoch.
    """
    if local_rank == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch + 1}, lr {current_lr}.")

    _iter = 0
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)
    unet.train()
    for train_data in train_loader:
        current_lr = optimizer.param_groups[0]["lr"]
        _iter += 1
        images = train_data["image"].to(device)
        images = images * scale_factor

        # Convert fixed metadata into tensors.
        top_region_index_tensor = torch.tensor(train_data["top_region_index"], dtype=torch.half, device=device)
        bottom_region_index_tensor = torch.tensor(train_data["bottom_region_index"], dtype=torch.half, device=device)
        spacing_tensor = torch.tensor(train_data["spacing"], dtype=torch.half, device=device)


        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            noise = torch.randn(
                (num_images_per_batch, 4, images.size(-3), images.size(-2), images.size(-1)), device=device
            )
            timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()
            noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
            noise_pred = unet(
                x=noisy_latent,
                timesteps=timesteps,
                top_region_index_tensor=top_region_index_tensor,
                bottom_region_index_tensor=bottom_region_index_tensor,
                spacing_tensor=spacing_tensor,
            )
            loss = loss_pt(noise_pred.float(), noise.float())

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        if local_rank == 0:
            logger.info(
                "[{0}] epoch {1}, iter {2}/{3}, loss: {4:.4f}, lr: {5:.12f}.".format(
                    str(datetime.now())[:19], epoch + 1, _iter, len(train_loader), loss.item(), current_lr
                )
            )

    if dist.is_initialized():
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)
    return loss_torch


def save_checkpoint(
    epoch: int,
    unet: torch.nn.Module,
    loss_torch_epoch: float,
    num_train_timesteps: int,
    scale_factor: torch.Tensor,
    ckpt_folder: str,
    args: argparse.Namespace,
) -> None:
    """
    Save checkpoint.
    """
    unet_state_dict = unet.module.state_dict() if dist.is_initialized() else unet.state_dict()
    torch.save(
        {
            "epoch": epoch + 1,
            "loss": loss_torch_epoch,
            "num_train_timesteps": num_train_timesteps,
            "scale_factor": scale_factor,
            "unet_state_dict": unet_state_dict,
        },
        f"{ckpt_folder}/{args.model_filename}",
    )


def diff_model_train(env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int, amp: bool = True) -> None:
    """
    Main function to train a diffusion model.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("training")

    logger.info(f"Using {device} of {world_size}")
    if local_rank == 0:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        logger.info(f"[config] data_root -> {args.embedding_base_dir}.")
        logger.info(f"[config] data_list -> {args.csv_data_list}.")
        logger.info(f"[config] lr -> {args.diffusion_unet_train['lr']}.")
        logger.info(f"[config] num_epochs -> {args.diffusion_unet_train['n_epochs']}.")
        logger.info(f"[config] num_train_timesteps -> {args.noise_scheduler['num_train_timesteps']}.")
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # Load unique file paths from the CSV.
    filenames_train = load_filenames(args.csv_data_list)
    if local_rank == 0:
        logger.info(f"num_files_train: {len(filenames_train)}")

    train_files = []
    for f in filenames_train:
        str_img = os.path.join(args.embedding_base_dir, f)
        if not os.path.exists(str_img):
            continue
        # Build sample dictionary; metadata will be attached via transforms.
        train_files.append({
            "image": str_img,
            "top_region_index": "",
            "bottom_region_index": "",
            "spacing": ""
        })

    if dist.is_initialized():
        train_files = partition_dataset(
            data=train_files, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True
        )[local_rank]

    train_loader = prepare_data(
        train_files,
        device,
        args.diffusion_unet_train["cache_rate"],
        batch_size=args.diffusion_unet_train["batch_size"]
    )

    unet = load_unet(args, device, logger)
    noise_scheduler = define_instance(args, "noise_scheduler")

    scale_factor = calculate_scale_factor(train_loader, device, logger)
    optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])
    total_steps = (args.diffusion_unet_train["n_epochs"] * len(train_loader.dataset)) / args.diffusion_unet_train["batch_size"]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    loss_pt = torch.nn.L1Loss()
    scaler = GradScaler()

    torch.set_float32_matmul_precision("highest")
    logger.info("torch.set_float32_matmul_precision -> highest.")

    for epoch in range(args.diffusion_unet_train["n_epochs"]):
        loss_torch = train_one_epoch(
            epoch,
            unet,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_pt,
            scaler,
            scale_factor,
            noise_scheduler,
            args.diffusion_unet_train["batch_size"],
            args.noise_scheduler["num_train_timesteps"],
            device,
            logger,
            local_rank,
            amp=amp,
        )
        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or local_rank == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            logger.info(f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}.")
            save_checkpoint(
                epoch,
                unet,
                loss_torch_epoch,
                args.noise_scheduler["num_train_timesteps"],
                scale_factor,
                args.model_dir,
                args,
            )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument("--env_config", type=str, default="configs/environment_maisi_diff_model.json",
                        help="Path to environment configuration file")
    parser.add_argument("--model_config", type=str, default="configs/config_maisi_diff_model.json",
                        help="Path to model training/inference configuration")
    parser.add_argument("--model_def", type=str, default="configs/config_maisi.json",
                        help="Path to model definition file")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable automatic mixed precision training")
    # New arguments for CSV data list and base directory
    parser.add_argument("--csv_data_list", type=str, default="/data3/wyh/mri/breast/encode/encoded_info.csv",
                        help="Path to CSV file containing data paths")
    parser.add_argument("--embedding_base_dir", type=str, default="/data3/wyh/mri/breast/encode",
                        help="Base directory for encoded npy files")
    # Model filename for saving checkpoints.
    parser.add_argument("--model_filename", type=str, default="diffusion_model_checkpoint.pt",
                        help="Filename for saving model checkpoint")
    args = parser.parse_args()
    diff_model_train(args.env_config, args.model_config, args.model_def, args.num_gpus, args.amp)
