import os
import torch
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import skimage
from PIL import Image
from tqdm import tqdm
import numpy as np
from networks import HashRelu, HashSIREN, Siren
from utils import transform_normalized, plot_compare_steps, plot_compare_time, save_metrics_to_csv
import time

torch.set_float32_matmul_precision('high')

# Initialize metrics
psnr_metric = PeakSignalNoiseRatio(data_range=2.0).cuda()
ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).cuda()

output_base_dir = "train"
test_name = "train"
os.makedirs(output_base_dir, exist_ok=True)

# Training parameters
total_steps = 3000
image_interval = 1000
psnr_interval = 50
activation_interval = 3000
activation_start = False
print_loss_interval = 1000
n_runs = 5

# Model and image setup
models = {
    "HashSIREN": HashSIREN,
    "Siren": Siren,
    "HashReLU": HashRelu,
}
images = {
    "cameraman": transform_normalized(Image.fromarray(skimage.data.camera())),
    "eagle": transform_normalized(Image.fromarray(skimage.data.eagle())), 
    "tokyo": transform_normalized(Image.open("tokyo_crop.jpg").convert("L"))
}

for image_name, image_tensor in images.items():
    for model_name, model_class in models.items():
        # Prepare input grid
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
        
        model_dir = os.path.join(output_base_dir, f"{model_name}_{image_name}")
        os.makedirs(model_dir, exist_ok=True)
        init_dir = os.path.join(model_dir, "normal_loss")
        os.makedirs(init_dir, exist_ok=True)
        model_input = model_input.detach().requires_grad_(True)
        ground_truth = image_tensor.squeeze(0).cuda().detach()
        psnr_values = []
        ssim_values = []
        time_values = []
        for i in range(n_runs):
            model = model_class().cuda()
            optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
            run_psnr_values = []
            run_ssim_values = []
            run_time_values = []
            model.eval()
            for step in tqdm(range(0, total_steps + 1), desc=f"Training INR {model_name} on {image_name}"):
                start_time = time.time()
                optim.zero_grad()
                model_input = model_input.detach().requires_grad_(True)
                model_output = model(model_input)
                loss = ((model_output - ground_truth)**2).mean()
                loss.backward()
                optim.step()
                end_time = time.time()
                run_time_values.append(end_time - start_time)
                if step % psnr_interval == 0:
                    psnr_value = psnr_metric(model_output.detach().squeeze(), ground_truth.detach().flatten())
                    run_psnr_values.append(psnr_value.item())
                    reshaped_model_output = model_output.detach().reshape(image_tensor.shape[1], image_tensor.shape[2]) 
                    reshaped_ground_truth = ground_truth.detach().reshape(image_tensor.shape[1], image_tensor.shape[2])
                    ssim_value = ssim_metric(reshaped_model_output.unsqueeze(0).unsqueeze(0), reshaped_ground_truth.unsqueeze(0).unsqueeze(0))
                    run_ssim_values.append(ssim_value.item())
            psnr_values.append(run_psnr_values)
            ssim_values.append(run_ssim_values)
            time_values.append(run_time_values)
        # Compute mean and std for metrics
        mean_psnr = np.mean(psnr_values, axis=0)
        std_psnr = np.std(psnr_values, axis=0)
        mean_ssim = np.mean(ssim_values, axis=0)
        std_ssim = np.std(ssim_values, axis=0)
        mean_time_ticks = np.cumsum(np.mean(time_values, axis=0))[::psnr_interval]
        psnr_mean_std = {"normal_loss": (mean_psnr, std_psnr)}
        ssim_mean_std = {"normal_loss": (mean_ssim, std_ssim)}
        mean_time_ticks_dict = {"normal_loss": mean_time_ticks}
        # Save results and plots
        save_metrics_to_csv(psnr_mean_std, ssim_mean_std, mean_time_ticks_dict, model_dir, image_name)
        plot_compare_steps(model_name, model_dir, psnr_mean_std, ssim_mean_std, psnr_interval, output_base_dir, image_name, test_name, n_runs)
        plot_compare_time(model_name, model_dir, psnr_mean_std, ssim_mean_std, mean_time_ticks_dict, psnr_interval, output_base_dir, image_name, test_name, n_runs) 