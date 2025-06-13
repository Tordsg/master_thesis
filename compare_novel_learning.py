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
from utils import grad, amplitude_grad, low_pass_frequency, high_pass_frequency, KERNEL_X, KERNEL_Y, SOBEL_X, SOBEL_Y
from utils import fourier_grad_loss, fourier_grad_loss_inverse, fourier_amplitude_grad_loss, fourier_amplitude_grad_loss_inverse, fourier_weighted_frequency_loss, fourier_weighted_frequency_loss_inverse, sobel_grad_loss, kernel_grad_loss, auto_grad_loss, grad_loss, normal_loss
from utils import visualize_fourier_grad_loss, visualize_fourier_grad_loss_inverse, visualize_fourier_amplitude_grad_loss, visualize_fourier_amplitude_grad_loss_inverse, visualize_fourier_weighted_frequency_loss, visualize_fourier_weighted_frequency_loss_inverse, visualize_sobel_grad_loss, visualize_kernel_grad_loss, visualize_auto_grad_loss, visualize_grad_loss
import time

# Set high precision for float32 matrix multiplication
# and record CUDA memory history
torch.set_float32_matmul_precision('high')
# torch.cuda.memory._record_memory_history(max_entries=100000)

# Initialize metrics and containers
psnr_metric = PeakSignalNoiseRatio(data_range=2.0).cuda()
ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).cuda()
activations = {}
activations_gradients = {}

# Output and training parameters
output_base_dir = "novel"
test_name = "novel"
os.makedirs(output_base_dir, exist_ok=True)

total_steps = 3000
image_interval = 1000
psnr_interval = 50
activation_interval = 3000
activation_start = False
print_loss_interval = 1000
n_runs = 5
LAMBDA = 0
LAMBDA_LOW = 0.1

# Model, image, and loss setup
models = {
    # "SIREN": Siren,
    "HashSIREN": HashSIREN,
    "HashReLU": HashRelu,
}
images = {
    "cameraman": transform_normalized(Image.fromarray(skimage.data.camera())),
    # "eagle": transform_normalized(Image.fromarray(skimage.data.eagle())),
    "tokyo": transform_normalized(Image.open("tokyo_crop.jpg").convert("L"))
}

hyperparameters = {'normal_loss': normal_loss, 
                   'grad_loss': grad_loss,
                   'auto_grad_loss': auto_grad_loss,
                   'sobel_grad_loss': sobel_grad_loss,
                   'kernel_grad_loss': kernel_grad_loss,
                   'fourier_grad_loss_inverse': fourier_grad_loss_inverse, 
                   'fourier_grad_loss': fourier_grad_loss,
                   'fourier_amplitude_grad_loss': fourier_amplitude_grad_loss,
                   'fourier_amplitude_grad_loss_inverse': fourier_amplitude_grad_loss_inverse,
                   'fourier_weighted_frequency_loss': fourier_weighted_frequency_loss,
                   'fourier_weighted_frequency_loss_inverse': fourier_weighted_frequency_loss_inverse,
}

# Prepare visualization grids
visual_gt = torch.zeros(256, 256).cuda()
visual_gt_complex = torch.complex(visual_gt, torch.zeros(256, 256).cuda())
visual_kx, visual_ky = torch.meshgrid(
    torch.fft.fftfreq(256),
    torch.fft.fftfreq(256),
    indexing='ij'
)
visual_kx, visual_ky = torch.fft.fftshift(visual_kx.cuda()), torch.fft.fftshift(visual_ky.cuda())

# Main training and evaluation loop
for image_name, image_tensor in images.items():
    for model_name, model_class in models.items():
        # Prepare input grids for each model
        if model_name != 'SIREN':
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
        else:
            visual_input = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, steps=256),
                torch.linspace(-1, 1, steps=256),
                indexing = "ij"
            ), dim=-1).reshape(-1, 2).cuda()
            model_input = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, steps=image_tensor.shape[1]),
                torch.linspace(-1, 1, steps=image_tensor.shape[2]),
                indexing = "ij"
            ), dim=-1).reshape(-1, 2).cuda()
        
        kx, ky = torch.meshgrid(
            torch.fft.fftfreq(image_tensor.shape[1]),
            torch.fft.fftfreq(image_tensor.shape[2]),
            indexing='ij'
        )
        kx, ky = torch.fft.fftshift(kx.cuda()), torch.fft.fftshift(ky.cuda())
        psnr_dict = {}
        time_dict = {}
        ssim_dict = {}
        model_dir = os.path.join(output_base_dir, f"{model_name}_{image_name}")
        os.makedirs(model_dir, exist_ok=True)

        for hyperparameter, loss_function in hyperparameters.items():
            # Loop over all loss functions
            psnr_dict[hyperparameter] = []
            time_dict[hyperparameter] = []
            ssim_dict[hyperparameter] = []
            init_dir = os.path.join(model_dir, f"{hyperparameter}")
            os.makedirs(init_dir, exist_ok=True)
            
            model_input = model_input.detach().requires_grad_(True)
            ground_truth = image_tensor.squeeze(0).cuda().detach()
            
            # Prepare ground truth gradients/fourier for some losses
            if hyperparameter == 'auto_grad_loss' and model_name == 'SIREN':
                gt_grad_x = torch.gradient(ground_truth, dim=0)[0]
                gt_grad_y = torch.gradient(ground_truth, dim=1)[0]
            elif hyperparameter == 'grad_loss' or hyperparameter == 'grad_loss_normalized':
                gt_grad_x = torch.gradient(ground_truth, dim=0)[0]
                gt_grad_y = torch.gradient(ground_truth, dim=1)[0]
            elif hyperparameter == 'fourier_grad_loss':
                fourier_gt = torch.fft.fftshift(torch.fft.fft2(ground_truth))
                fourier_gt_grad_x = fourier_gt * grad(kx)
                fourier_gt_grad_y = fourier_gt * grad(ky)
            elif hyperparameter == 'fourier_grad_loss_inverse':
                fourier_gt = torch.fft.fftshift(torch.fft.fft2(ground_truth))
                fourier_gt_grad_x = fourier_gt * grad(kx)
                fourier_gt_grad_y = fourier_gt * grad(ky)
            elif hyperparameter == 'fourier_amplitude_grad_loss':
                fourier_gt = torch.fft.fftshift(torch.fft.fft2(ground_truth))
                fourier_amplitude_gt = fourier_gt * amplitude_grad(kx, ky)
            elif hyperparameter == 'fourier_amplitude_grad_loss_inverse':
                fourier_gt = torch.fft.fftshift(torch.fft.fft2(ground_truth))
                fourier_amplitude_gt = fourier_gt * amplitude_grad(kx, ky)
            elif hyperparameter == 'fourier_weighted_frequency_loss':
                fourier_gt = torch.fft.fftshift(torch.fft.fft2(ground_truth))
                fourier_gt_low = fourier_gt * low_pass_frequency(kx, ky, PERCENTAGE_LOW)
                fourier_gt_high = fourier_gt * high_pass_frequency(kx, ky, PERCENTAGE_LOW)
            elif hyperparameter == 'fourier_weighted_frequency_loss_inverse':
                fourier_gt = torch.fft.fftshift(torch.fft.fft2(ground_truth))
                fourier_gt_low = fourier_gt * low_pass_frequency(kx, ky, PERCENTAGE_LOW)
                fourier_gt_high = fourier_gt * high_pass_frequency(kx, ky, PERCENTAGE_LOW)
                fourier_gt_low_inverse = torch.fft.ifft2(torch.fft.ifftshift(fourier_gt_low)).real
                fourier_gt_high_inverse = torch.fft.ifft2(torch.fft.ifftshift(fourier_gt_high)).real
            elif hyperparameter == 'sobel_grad_loss':
                sobel_gt_grad_x = torch.nn.functional.conv2d(ground_truth.unsqueeze(0).unsqueeze(0), SOBEL_X, padding=1)
                sobel_gt_grad_y = torch.nn.functional.conv2d(ground_truth.unsqueeze(0).unsqueeze(0), SOBEL_Y, padding=1)
            elif hyperparameter == 'kernel_grad_loss':
                kernel_gt_grad_x = torch.nn.functional.conv2d(ground_truth.unsqueeze(0).unsqueeze(0), KERNEL_X, padding=(0, 1))
                kernel_gt_grad_y = torch.nn.functional.conv2d(ground_truth.unsqueeze(0).unsqueeze(0), KERNEL_Y, padding=(1, 0))
            elif hyperparameter == 'normal_loss':
                pass
            else:
                continue
            for i in range(n_runs):
                # Training loop for each run
                model = model_class().cuda()
                optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
                psnr_values = []
                ssim_values = []
                time_values = []
                model.eval()
                if i == 0 and activation_start:
                    # Optionally record activations before training
                    with capture_activations(model, record_activation,activations, activations_gradients):
                        visual_input = visual_input.detach().requires_grad_(True)
                        visual_output = model(visual_input).squeeze(1)
                        # Compute loss for activations
                        if hyperparameter == 'fourier_grad_loss':
                            loss = loss_function(visual_output, visual_kx, visual_ky, visual_gt, visual_gt, visual_gt, grad_fn=grad, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                        elif hyperparameter == 'fourier_grad_loss_inverse':
                            loss = loss_function(visual_output, visual_gt, visual_kx, visual_ky, visual_gt, visual_gt, grad_fn=grad, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                        elif hyperparameter == 'fourier_amplitude_grad_loss':
                            loss = loss_function(visual_output, visual_kx, visual_ky, visual_gt, visual_gt, grad_fn=amplitude_grad, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                        elif hyperparameter == 'fourier_amplitude_grad_loss_inverse':
                            loss = loss_function(visual_output, visual_gt, visual_kx, visual_ky, visual_gt, grad_fn=amplitude_grad, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                        elif hyperparameter == 'fourier_weighted_frequency_loss':
                            loss = loss_function(visual_output, visual_kx, visual_ky, visual_gt, visual_gt, grad_fn_low=low_pass_frequency, grad_fn_high=high_pass_frequency, LAMBDA_LOW=LAMBDA_LOW, PERCENTAGE_LOW=PERCENTAGE_LOW, step=1, print_loss_interval=print_loss_interval)
                        elif hyperparameter == 'fourier_weighted_frequency_loss_inverse':
                            loss = loss_function(visual_output, visual_kx, visual_ky, visual_gt, visual_gt, grad_fn_low=low_pass_frequency, grad_fn_high=high_pass_frequency, LAMBDA_LOW=LAMBDA_LOW, PERCENTAGE_LOW=PERCENTAGE_LOW, step=1, print_loss_interval=print_loss_interval)
                        elif hyperparameter == 'normal_loss':
                            loss = loss_function(visual_output, visual_gt, step=1, print_loss_interval=print_loss_interval)
                        elif hyperparameter == 'sobel_grad_loss':
                            loss = loss_function(visual_output, visual_gt, visual_gt, visual_gt, SOBEL_X=SOBEL_X, SOBEL_Y=SOBEL_Y, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                        elif hyperparameter == 'kernel_grad_loss':
                            loss = loss_function(visual_output, visual_gt, visual_gt, visual_gt, KERNEL_X=KERNEL_X, KERNEL_Y=KERNEL_Y, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                        elif hyperparameter == 'auto_grad_loss':
                            loss = loss_function(visual_output, visual_gt, visual_input, visual_gt, visual_gt, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                        elif hyperparameter == 'grad_loss':
                            loss = loss_function(visual_output, visual_gt, visual_gt, visual_gt, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                        else:
                            loss = loss_function(visual_output, visual_gt, step=1, print_loss_interval=print_loss_interval)
                        loss.backward()
                    activations_dir = os.path.join(init_dir, f'activations_step_before')
                    plot_activations_spectrum_and_gradients_weights_with_tikz(model_name, model, activations_dir, activations, activations_gradients)
                    activations.clear(), activations_gradients.clear()
                    del visual_output
                    # Detach visual_input to prevent graph reuse issues
                    visual_input = visual_input.detach()
                    model.train()
                for step in tqdm(range(0, total_steps + 1), desc=f"Training INR {model_name} on {image_name} with hyperparameter: {hyperparameter}"):
                    # Main training loop
                    start_time = time.time()
                    optim.zero_grad()
                    # Detach model_input to prevent graph reuse
                    model_input = model_input.detach().requires_grad_(True)
                    model_output = model(model_input)
                    # Compute loss for current step
                    if hyperparameter == 'fourier_grad_loss':
                        loss = loss_function(model_output, kx, ky, fourier_gt_grad_x, fourier_gt_grad_y, fourier_gt, grad_fn=grad, LAMBDA=LAMBDA, step=step, print_loss_interval=print_loss_interval)
                    elif hyperparameter == 'fourier_grad_loss_inverse':
                        loss = loss_function(model_output, ground_truth, kx, ky, fourier_gt_grad_x, fourier_gt_grad_y, grad_fn=grad, LAMBDA=LAMBDA, step=step, print_loss_interval=print_loss_interval)
                    elif hyperparameter == 'fourier_amplitude_grad_loss':
                        loss = loss_function(model_output, kx, ky, fourier_amplitude_gt, fourier_gt, grad_fn=amplitude_grad, LAMBDA=LAMBDA, step=step, print_loss_interval=print_loss_interval)
                    elif hyperparameter == 'fourier_amplitude_grad_loss_inverse':
                        loss = loss_function(model_output, ground_truth, kx, ky, fourier_amplitude_gt, grad_fn=amplitude_grad, LAMBDA=LAMBDA, step=step, print_loss_interval=print_loss_interval)
                    elif hyperparameter == 'fourier_weighted_frequency_loss':
                        loss = loss_function(model_output, kx, ky, fourier_gt_low, fourier_gt_high, grad_fn_low=low_pass_frequency, grad_fn_high=high_pass_frequency, LAMBDA_LOW=LAMBDA_LOW, PERCENTAGE_LOW=PERCENTAGE_LOW, step=step, print_loss_interval=print_loss_interval)
                    elif hyperparameter == 'fourier_weighted_frequency_loss_inverse':
                        loss = loss_function(model_output, kx, ky, fourier_gt_low_inverse, fourier_gt_high_inverse, grad_fn_low=low_pass_frequency, grad_fn_high=high_pass_frequency, LAMBDA_LOW=LAMBDA_LOW, PERCENTAGE_LOW=PERCENTAGE_LOW, step=step, print_loss_interval=print_loss_interval)
                    elif hyperparameter == 'normal_loss':
                        loss = loss_function(model_output, ground_truth, step=step, print_loss_interval=print_loss_interval)
                    elif hyperparameter == 'sobel_grad_loss':
                        loss = loss_function(model_output, ground_truth, sobel_gt_grad_x, sobel_gt_grad_y, SOBEL_X=SOBEL_X, SOBEL_Y=SOBEL_Y, LAMBDA=LAMBDA, step=step, print_loss_interval=print_loss_interval)
                    elif hyperparameter == 'kernel_grad_loss':
                        loss = loss_function(model_output, ground_truth, kernel_gt_grad_x, kernel_gt_grad_y, KERNEL_X=KERNEL_X, KERNEL_Y=KERNEL_Y, LAMBDA=LAMBDA, step=step, print_loss_interval=print_loss_interval)
                    elif hyperparameter == 'auto_grad_loss':
                        loss = loss_function(model_output, ground_truth, model_input, gt_grad_x, gt_grad_y, LAMBDA=LAMBDA, step=step, print_loss_interval=print_loss_interval)
                    elif hyperparameter == 'grad_loss':
                        loss = loss_function(model_output, ground_truth, gt_grad_x, gt_grad_y, LAMBDA=LAMBDA, step=step, print_loss_interval=print_loss_interval)
                    else:
                        loss = loss_function(model_output, ground_truth, step=step, print_loss_interval=print_loss_interval)
                    loss.backward()
                    optim.step()
                    end_time = time.time()
                    time_values.append(end_time - start_time)
                    if step % activation_interval == 0 and step != 0 and i == 0:
                        # Optionally record activations during training
                        model.eval()
                        with capture_activations(model, record_activation,activations, activations_gradients):
                            visual_input = visual_input.detach().requires_grad_(True)
                            visual_output = model(visual_input)
                            # Compute loss for activations
                            if hyperparameter == 'fourier_grad_loss':
                                loss = loss_function(visual_output, visual_kx, visual_ky, visual_gt, visual_gt, visual_gt, grad_fn=grad, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                            elif hyperparameter == 'fourier_grad_loss_inverse':
                                loss = loss_function(visual_output, visual_gt, visual_kx, visual_ky, visual_gt, visual_gt, grad_fn=grad, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                            elif hyperparameter == 'fourier_amplitude_grad_loss':
                                loss = loss_function(visual_output, visual_kx, visual_ky, visual_gt, visual_gt, grad_fn=amplitude_grad, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                            elif hyperparameter == 'fourier_amplitude_grad_loss_inverse':
                                loss = loss_function(visual_output, visual_gt, visual_kx, visual_ky, visual_gt, grad_fn=amplitude_grad, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                            elif hyperparameter == 'fourier_weighted_frequency_loss':
                                loss = loss_function(visual_output, visual_kx, visual_ky, visual_gt, visual_gt, grad_fn_low=low_pass_frequency, grad_fn_high=high_pass_frequency, LAMBDA_LOW=LAMBDA_LOW, PERCENTAGE_LOW=PERCENTAGE_LOW, step=1, print_loss_interval=print_loss_interval)
                            elif hyperparameter == 'fourier_weighted_frequency_loss_inverse':
                                loss = loss_function(visual_output, visual_kx, visual_ky, visual_gt, visual_gt, grad_fn_low=low_pass_frequency, grad_fn_high=high_pass_frequency, LAMBDA_LOW=LAMBDA_LOW, PERCENTAGE_LOW=PERCENTAGE_LOW, step=1, print_loss_interval=print_loss_interval)
                            elif hyperparameter == 'normal_loss':
                                loss = loss_function(visual_output, visual_gt, step=1, print_loss_interval=print_loss_interval)
                            elif hyperparameter == 'sobel_grad_loss':
                                loss = loss_function(visual_output, visual_gt, visual_gt, visual_gt, SOBEL_X=SOBEL_X, SOBEL_Y=SOBEL_Y, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                            elif hyperparameter == 'kernel_grad_loss':
                                loss = loss_function(visual_output, visual_gt, visual_gt, visual_gt, KERNEL_X=KERNEL_X, KERNEL_Y=KERNEL_Y, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                            elif hyperparameter == 'auto_grad_loss':
                                loss = loss_function(visual_output, visual_gt, visual_input, visual_gt, visual_gt, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                            elif hyperparameter == 'grad_loss':
                                loss = loss_function(visual_output, visual_gt, visual_gt, visual_gt, LAMBDA=LAMBDA, step=1, print_loss_interval=print_loss_interval)
                            else:
                                loss = loss_function(visual_output, visual_gt, step=1, print_loss_interval=print_loss_interval)
                            loss.backward()
                        activations_dir = os.path.join(init_dir, f'activations_step_{step}')
                        plot_activations_spectrum_and_gradients_weights_with_tikz(model_name, model, activations_dir, activations, activations_gradients)
                        activations.clear(), activations_gradients.clear()
                        del visual_output
                        visual_input = visual_input.detach()
                        model.train()
                        
                    if step % psnr_interval == 0:
                        # Record PSNR and SSIM
                        psnr_value = psnr_metric(model_output.detach().squeeze(), ground_truth.detach().flatten())
                        psnr_values.append(psnr_value.item())
                        reshaped_model_output = model_output.detach().reshape(image_tensor.shape[1], image_tensor.shape[2]) 
                        reshaped_ground_truth = ground_truth.detach().reshape(image_tensor.shape[1], image_tensor.shape[2])
                        ssim_value = ssim_metric(reshaped_model_output.unsqueeze(0).unsqueeze(0), reshaped_ground_truth.unsqueeze(0).unsqueeze(0))
                        ssim_values.append(ssim_value.item())
                        
                    if step % image_interval == 0 and i == 0:
                        # Save output images and visualizations
                        img = ((model_output.detach().reshape(image_tensor.shape[1], image_tensor.shape[2]).clamp(-1,1)+1)/2*255).round().to(torch.uint8).cpu().numpy()
                        pil_img = Image.fromarray(img)
                        resolutions = [image_tensor.shape[1], image_tensor.shape[2]]
                        model_output_detached = model_output.detach()
                        
                        if hyperparameter == 'grad_loss':
                            visualize_grad_loss(model_output_detached, resolutions, init_dir, step)
                        elif hyperparameter == 'auto_grad_loss':
                            model.eval()
                            fresh_input = model_input.detach().requires_grad_(True)
                            fresh_output = model(fresh_input)
                            visualize_auto_grad_loss(fresh_output, resolutions, fresh_input, init_dir, step)
                            model.train()
                        elif hyperparameter == 'sobel_grad_loss':
                            visualize_sobel_grad_loss(model_output_detached, resolutions=resolutions, sobel_x=SOBEL_X, sobel_y=SOBEL_Y, output_dir=init_dir, step=step)
                        elif hyperparameter == 'kernel_grad_loss':
                            visualize_kernel_grad_loss(model_output_detached, resolutions=resolutions, kernel_x=KERNEL_X, kernel_y=KERNEL_Y, output_dir=init_dir, step=step)
                        elif hyperparameter == 'fourier_amplitude_grad_loss':
                            visualize_fourier_amplitude_grad_loss(model_output_detached, resolutions=resolutions, grad_fn=amplitude_grad, output_dir=init_dir, step=step)
                        elif hyperparameter == 'fourier_amplitude_grad_loss_inverse':
                            visualize_fourier_amplitude_grad_loss_inverse(model_output_detached, resolutions=resolutions, grad_fn=amplitude_grad, output_dir=init_dir, step=step)
                        elif hyperparameter == 'fourier_weighted_frequency_loss':
                            visualize_fourier_weighted_frequency_loss(model_output_detached, resolutions=resolutions, grad_fn_low=low_pass_frequency, grad_fn_high=high_pass_frequency, PERCENTAGE_LOW=PERCENTAGE_LOW, output_dir=init_dir, step=step)
                        elif hyperparameter == 'fourier_weighted_frequency_loss_inverse':
                            visualize_fourier_weighted_frequency_loss_inverse(model_output_detached, resolutions=resolutions, grad_fn_low=low_pass_frequency, grad_fn_high=high_pass_frequency, PERCENTAGE_LOW=PERCENTAGE_LOW, output_dir=init_dir, step=step)
                        elif hyperparameter == 'fourier_grad_loss':
                            visualize_fourier_grad_loss(model_output_detached, resolutions=resolutions, grad_fn=grad, output_dir=init_dir, step=step)
                        elif hyperparameter == 'fourier_grad_loss_inverse':
                            visualize_fourier_grad_loss_inverse(model_output_detached, resolutions=resolutions, grad_fn=grad, output_dir=init_dir, step=step)
                        if image_name == "tokyo":
                            downsample_factor = 2
                            new_size = (pil_img.width // downsample_factor, pil_img.height // downsample_factor)
                            pil_img = pil_img.resize(new_size, resample=Image.LANCZOS)
                            pil_img.save(
                                os.path.join(init_dir, f"{model_name}_downsampled_step_{step:04d}.jpg"),
                                format="JPEG",
                                quality=95
                            )
                        else:
                            pil_img.save(
                                os.path.join(init_dir, f"{model_name}_step_{step:04d}.jpg"),
                                format="JPEG",
                                quality=95
                            )
                psnr_dict[hyperparameter].append(psnr_values)
                ssim_dict[hyperparameter].append(ssim_values)
                time_dict[hyperparameter].append(time_values)

        # Compute and save metrics, plots for all losses
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
       
