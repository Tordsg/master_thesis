#!/usr/bin/env python3
# This script is a simplified version of compare_novel_learning.py
# It only measures for one loss (normal_loss) and removes all novel loss functionality and loops over multiple losses.

import os
import torch
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import skimage
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
import tikzplotlib
from networks import Siren, HashRelu, HashSIREN
from utils import plot_activations_spectrum_and_gradients_weights_with_tikz, transform_normalized, record_activation, capture_activations, plot_compare_steps, plot_compare_time, save_metrics_to_csv
from utils import normal_loss
import time

torch.set_float32_matmul_precision('high')

psnr_metric = PeakSignalNoiseRatio(data_range=2.0).cuda()
ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).cuda()

output_base_dir = "novel_single_loss"
test_name = "novel_single_loss"
os.makedirs(output_base_dir, exist_ok=True)

total_steps = 3000
image_interval = 1000
psnr_interval = 50
activation_interval = 3000
activation_start = False
print_loss_interval = 1000
n_runs = 5

models = {
    "HashSIREN": HashSIREN,
    "HashReLU": HashRelu,
}
images = {
    "cameraman": transform_normalized(Image.fromarray(skimage.data.camera())),
    "tokyo": transform_normalized(Image.open("tokyo_crop.jpg").convert("L"))
}

for image_name, image_tensor in images.items():
    for model_name, model_class in models.items():
        visual_input = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, steps=256),
            torch.linspace(0, 1, steps=256),
            indexing = "ij"
        ), dim=-1).reshape(-1, 2).cuda()
        model_input = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, steps=image_tensor.shape[1]),
            torch.linspace(0, 1, steps=image_tensor.shape[2]),
            indexing = "ij"
        ), dim=-1).reshape(-1, 2).cuda()
        
        psnr_dict = {}
        time_dict = {}
        ssim_dict = {}
        model_dir = os.path.join(output_base_dir, f"{model_name}_{image_name}")
        os.makedirs(model_dir, exist_ok=True)

        # Only use normal_loss
        hyperparameter = 'normal_loss'
        loss_function = normal_loss
        psnr_dict[hyperparameter] = []
        time_dict[hyperparameter] = []
        ssim_dict[hyperparameter] = []
        init_dir = os.path.join(model_dir, f"{hyperparameter}")
        os.makedirs(init_dir, exist_ok=True)
        model_input = model_input.detach().requires_grad_(True)
        ground_truth = image_tensor.squeeze(0).cuda().detach()
        for i in range(n_runs):
            model = model_class().cuda()
            optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
            psnr_values = []
            ssim_values = []
            time_values = []
            model.eval()
            for step in tqdm(range(0, total_steps + 1), desc=f"Training INR {model_name} on {image_name} with normal_loss"):
                start_time = time.time()
                optim.zero_grad()
                model_input = model_input.detach().requires_grad_(True)
                model_output = model(model_input)
                loss = loss_function(model_output, ground_truth, step=step, print_loss_interval=print_loss_interval)
                loss.backward()
                optim.step()
                end_time = time.time()
                time_values.append(end_time - start_time)
                if step % psnr_interval == 0:
                    psnr_value = psnr_metric(model_output.detach().squeeze(), ground_truth.detach().flatten())
                    psnr_values.append(psnr_value.item())
                    reshaped_model_output = model_output.detach().reshape(image_tensor.shape[1], image_tensor.shape[2]) 
                    reshaped_ground_truth = ground_truth.detach().reshape(image_tensor.shape[1], image_tensor.shape[2])
                    ssim_value = ssim_metric(reshaped_model_output.unsqueeze(0).unsqueeze(0), reshaped_ground_truth.unsqueeze(0).unsqueeze(0))
                    ssim_values.append(ssim_value.item())
            psnr_dict[hyperparameter].append(psnr_values)
            ssim_dict[hyperparameter].append(ssim_values)
            time_dict[hyperparameter].append(time_values)
        psnr_mean_std = {}
        ssim_mean_std = {}
        mean_time_ticks = {}
        for hyperparameter, values in psnr_dict.items():
            if values: 
                mean_psnr = np.mean(values, axis=0)
                std_psnr = np.std(values, axis=0)
                mean_ssim = np.mean(ssim_dict[hyperparameter], axis=0)
                std_ssim = np.std(ssim_dict[hyperparameter], axis=0)
                if np.isscalar(mean_psnr):
                    mean_psnr = np.array([mean_psnr])
                if np.isscalar(std_psnr):
                    std_psnr = np.array([std_psnr])
                if np.isscalar(mean_ssim):
                    mean_ssim = np.array([mean_ssim])
                if np.isscalar(std_ssim):
                    std_ssim = np.array([std_ssim])
                psnr_mean_std[hyperparameter] = (mean_psnr, std_psnr)
                ssim_mean_std[hyperparameter] = (mean_ssim, std_ssim)
                mean_time_ticks[hyperparameter] = np.cumsum(np.mean(time_dict[hyperparameter], axis=0))[::psnr_interval]
        save_metrics_to_csv(psnr_mean_std, ssim_mean_std, mean_time_ticks, model_dir, image_name)
        plot_compare_steps(model_name, model_dir, psnr_mean_std, ssim_mean_std, psnr_interval, output_base_dir, image_name, test_name, n_runs)
        plot_compare_time(model_name, model_dir, psnr_mean_std, ssim_mean_std, mean_time_ticks, psnr_interval, output_base_dir, image_name, test_name, n_runs) 