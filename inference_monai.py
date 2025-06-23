#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, random, datetime
import numpy as np, torch, nibabel as nib
from tqdm import tqdm
from monai.utils import set_determinism
from scripts.diff_model_setting import initialize_distributed, load_config, setup_logging
from scripts.sample import ReconModel, check_input
from scripts.utils import define_instance
import torch.nn.functional as F

def set_random_seed(seed):
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed

def load_models(args, device, logger):
    # 加载 VAE (autoencoder)
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ckpt_vae = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=True)
    autoencoder.load_state_dict(ckpt_vae)
    # 加载训练好的 diffusion UNet
    unet = define_instance(args, "diffusion_unet_def").to(device)
    ckpt = torch.load(os.path.join(args.model_dir, args.model_filename), map_location=device, weights_only=False)
    unet.load_state_dict(ckpt["unet_state_dict"], strict=True)
    scale_factor = ckpt["scale_factor"]
    logger.info(f"Loaded checkpoint {os.path.join(args.model_dir, args.model_filename)} with scale_factor {scale_factor}.")
    return autoencoder, unet, scale_factor

def prepare_tensors(args, device):
    # 固定输入设计：top_region_index, bottom_region_index, spacing
    top_tensor = torch.tensor(args.diffusion_unet_inference["top_region_index"], dtype=torch.half, device=device)[None, :]
    bottom_tensor = torch.tensor(args.diffusion_unet_inference["bottom_region_index"], dtype=torch.half, device=device)[None, :]
    spacing_tensor = torch.tensor(args.diffusion_unet_inference["spacing"], dtype=torch.half, device=device)[None, :]
    return top_tensor, bottom_tensor, spacing_tensor

def run_inference(args, device, autoencoder, unet, scale_factor, top_tensor, bottom_tensor, spacing_tensor):
    # 根据配置生成初始 latent
    latent_shape = (
        1,
        args.latent_channels,
        args.diffusion_unet_inference["dim"][0] // 4,
        args.diffusion_unet_inference["dim"][1] // 4,
        args.diffusion_unet_inference["dim"][2] // 4,
    )
    noise = torch.randn(latent_shape, device=device)
    # 设置 noise scheduler
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])
    # 用 ReconModel 包装 autoencoder 用于直接解码
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    autoencoder.eval()
    unet.eval()
    image = noise
    with torch.amp.autocast("cuda", enabled=True):
        for t in tqdm(noise_scheduler.timesteps, ncols=100):
            t_tensor = torch.tensor([t], device=device)
            model_out = unet(
                x=image,
                timesteps=t_tensor,
                top_region_index_tensor=top_tensor,
                bottom_region_index_tensor=bottom_tensor,
                spacing_tensor=spacing_tensor,
            )
            image, _ = noise_scheduler.step(model_out, t, image)
        # 若 diffusion 过程中尺寸下降过多，则上采样回原始 latent 尺寸
        expected_shape = latent_shape
        # if image.shape[2:] != expected_shape[2:]:
        #     image = F.interpolate(image, size=expected_shape[2:], mode="trilinear", align_corners=False)
        # # 直接调用 ReconModel 解码，不使用滑动窗口
        print(image.shape)
        synthetic = recon_model(image)
    data = synthetic.squeeze().cpu().detach().numpy()
    return data.astype(np.int16)

def save_image(data, out_spacing, output_path):
    affine = np.eye(4)
    for i in range(3):
        affine[i, i] = out_spacing[i]
    nib.save(nib.Nifti1Image(data, affine), output_path)

def main():
    parser = argparse.ArgumentParser(description="Diffusion Model Inference with VAE & Training UNet for MRI")
    parser.add_argument("--env_config", type=str, default="configs/environment_maisi_diff_model.json")
    parser.add_argument("--model_config", type=str, default="configs/config_maisi_diff_model.json")
    parser.add_argument("--model_def", type=str, default="configs/config_maisi.json")
    parser.add_argument("--num_gpus", type=int, default=1)
    # 模型及输出相关参数
    parser.add_argument("--trained_autoencoder_path", type=str, default="models/autoencoder_epoch273.pt",
                        help="Path to the trained autoencoder (VAE) checkpoint.")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory where the diffusion UNet checkpoint is saved.")
    parser.add_argument("--model_filename", type=str, default="input_unet3d_data-all_steps1000size512ddpm_random_current_inputx_v1.pt",
                        help="Filename for the diffusion UNet checkpoint.")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save the synthetic image.")
    parser.add_argument("--output_prefix", type=str, default="synthetic",
                        help="Prefix for the output filename.")
    parser.add_argument("--latent_channels", type=int, default=4,
                        help="Number of latent channels (default: 4).")
    args_cli = parser.parse_args()
    
    # 使用 load_config 加载配置文件合并后的 args 对象
    args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)
    # 用命令行参数覆盖配置中对应字段
    args.trained_autoencoder_path = args_cli.trained_autoencoder_path
    args.model_dir = args_cli.model_dir
    args.model_filename = args_cli.model_filename
    args.output_dir = args_cli.output_dir
    args.output_prefix = args_cli.output_prefix
    args.latent_channels = args_cli.latent_channels

    local_rank, world_size, device = initialize_distributed(args_cli.num_gpus)
    logger = setup_logging("inference")
    seed = set_random_seed(args.diffusion_unet_inference.get("random_seed"))
    logger.info(f"Using {device} with seed {seed}.")

    output_size = tuple(args.diffusion_unet_inference["dim"])
    out_spacing = tuple(args.diffusion_unet_inference["spacing"])

    check_input(None, None, None, output_size, out_spacing, None)

    autoencoder, unet, scale_factor = load_models(args, device, logger)
    top_tensor, bottom_tensor, spacing_tensor = prepare_tensors(args, device)
    data = run_inference(args, device, autoencoder, unet, scale_factor, top_tensor, bottom_tensor, spacing_tensor)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"{args.output_prefix}_seed{seed}_size{'x'.join(map(str, output_size))}_{timestamp}_rank{local_rank}.nii.gz"
    output_path = os.path.join(args.output_dir, output_filename)
    os.makedirs(args.output_dir, exist_ok=True)
    save_image(data, out_spacing, output_path)
    logger.info(f"Saved synthetic image to {output_path}.")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()