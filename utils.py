import math
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from contextlib import contextmanager
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch import nn
from typing import Callable
import tikzplotlib
import matplotlib
import csv

def save_tikz_and_png(fig, filename, directory):
    tex_dir = os.path.join(directory, "tex")
    os.makedirs(tex_dir, exist_ok=True)
    png_dir = os.path.join(directory, "png")
    os.makedirs(png_dir, exist_ok=True)
    png_path = os.path.join(png_dir, f"{filename}.png")
    tikz_path = os.path.join(tex_dir, f"{filename}.tex")
    fig.savefig(png_path)
    tikzplotlib.save(tikz_path)
    plt.close(fig)

def plot_activations_spectrum_and_gradients_weights(model_name, model, model_dir, activations, activations_gradients):
    act_dir = os.path.join(model_dir, "activations")
    os.makedirs(act_dir, exist_ok=True)
    grad_dir = os.path.join(model_dir, "gradients")
    os.makedirs(grad_dir, exist_ok=True)
    weight_dir = os.path.join(model_dir, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.figure(figsize=(6, 4))
            weights = param.detach().cpu().numpy().flatten()
            layer = name.split('.')[1]
            plt.hist(weights, bins=100, alpha=0.6)
            if ('0' in name and model_name == 'HashSIREN') or model_name == 'SIREN':
                mean_weight = np.mean(weights)
                std_weight = np.std(weights)
                plt.title(f"{model_name} weight distribution layer {layer} - mean: {mean_weight:.2f}, std: {std_weight:.2f}")
            else:
                plt.title(f"{model_name} weight distribution layer {layer}")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(weight_dir, f"{name}_weight_distribution.png"))
            plt.close()
        if 'param' in name:
            plt.figure(figsize=(6, 4))
            params = param.detach().cpu().numpy().flatten()
            plt.hist(params, bins=100, alpha=0.6)
            plt.title(f"{model_name} encoding param distribution layer")
            plt.xlabel("Param Value")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(weight_dir, f"{name}_param_distribution.png"))
            plt.close()
    
    for name, data in activations.items():
        if name == '':
            layer = data['input']
            printname = "input"
            title = f"{model_name} input activation distribution"
        elif name.split('.')[-1] == 'encoding':
            layer = data['output']
            printname = "encoding"
            title = f"{model_name} encoding activation distribution"
        elif name == 'mlp':
            continue
        else:
            layer = data['output']
            layer_number = str(int(name.split('.')[1]) + 1)
            if name.split('.')[-1] == 'linear':
                mean_weight = np.mean(layer)
                std_weight = np.std(layer)
                printname = f"layer_{layer_number}_dotproduct"
                title = f"{model_name} layer {layer_number} dotproduct activation distribution - mean: {mean_weight:.2f}, std: {std_weight:.2f}"
            else:
                printname = f"layer_{layer_number}_activation_function"
                title = f"{model_name} layer {layer_number} activation function activation distribution"
        plot = layer.flatten().cpu().numpy()
        plt.figure(figsize=(6, 4))
        plt.hist(plot, bins=100, alpha=0.6)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(act_dir, f"{printname}_activation_distribution.png"))
        plt.close()

        if name in activations_gradients:
            plt.figure(figsize=(6, 4))
            grad = activations_gradients[name]
            plt.hist(grad.flatten().cpu().numpy(), bins=100, alpha=0.6)
            plt.title(f"{model_name} gradient distribution layer {layer_number}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(grad_dir, f"{printname}_gradient_distribution.png"))
            plt.close()
            
def plot_all_activations_spectrum_and_gradients_weights(model, model_dir, activations, activations_gradients):
    act_dir = os.path.join(model_dir, "activations")
    os.makedirs(act_dir, exist_ok=True)
    grad_dir = os.path.join(model_dir, "gradients")
    os.makedirs(grad_dir, exist_ok=True)
    weight_dir = os.path.join(model_dir, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.figure(figsize=(6, 4))
            weights = param.detach().cpu().numpy().flatten()
            plt.hist(weights, bins=100, alpha=0.6, label=f"{name} weights")
            plt.title(f"Weight Distribution - {name}")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(weight_dir, f"{name}_weight_distribution.png"))
            plt.close()
        if 'param' in name:
            plt.figure(figsize=(6, 4))
            params = param.detach().cpu().numpy().flatten()
            plt.hist(params, bins=100, alpha=0.6, label=f"{name} params")
            plt.title(f"Param Distribution - {name}")
            plt.xlabel("Param Value")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(weight_dir, f"{name}_param_distribution.png"))
            plt.close()
    
    for name, data in activations.items():
        if name == '':
            layer = data['input']
            printname = "input"
        elif name.split('.')[-1] == 'encoding':
            layer = data['output']
            printname = "encoding"
        elif name == 'mlp':
            printname = "mlp"
        else:
            layer = data['output']
            layer_number = str(int(name.split('.')[1]) + 1)
            if name.split('.')[-1] == 'linear':
                printname = f"layer_{layer_number}_dotproduct"
            else:
                printname = f"layer_{layer_number}_activation_function"
        input_layer = data['input']
        output_layer = data['output']
        
        input_plot = input_layer.flatten().cpu().numpy()
        plt.figure(figsize=(6, 4))
        plt.hist(input_plot, bins=100, alpha=0.6, label=f"{printname} input")
        plt.title(f"Input Activation Distribution")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(act_dir, f"{printname}_input_activation_distribution.png"))
        plt.close()
        
        output_plot = output_layer.flatten().cpu().numpy()
        plt.figure(figsize=(6, 4))
        plt.hist(output_plot, bins=100, alpha=0.6, label=f"{printname} output")
        plt.title(f"Output Activation Distribution")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(act_dir, f"{printname}_output_activation_distribution.png"))
        plt.close()
        
        if name in activations_gradients:
            plt.figure(figsize=(6, 4))
            grad = activations_gradients[name]
            plt.hist(grad.flatten().cpu().numpy(), bins=100, alpha=0.6, label=printname)
            plt.title(f"Gradient Distribution")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(grad_dir, f"{printname}_gradient_distribution.png"))
            plt.close()

def plot_activations_spectrum_and_gradients_weights_with_tikz(model_name, model, model_dir, activations, activations_gradients):
    act_dir = os.path.join(model_dir, "activations")
    os.makedirs(act_dir, exist_ok=True)
    grad_dir = os.path.join(model_dir, "gradients")
    os.makedirs(grad_dir, exist_ok=True)
    weight_dir = os.path.join(model_dir, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.figure(figsize=(6, 4))
            weights = param.detach().cpu().numpy().flatten()
            layer = name.split('.')[1]
            plt.hist(weights, bins=100, alpha=0.6)
            if ('0' in name and model_name == 'HashSIREN') or model_name == 'SIREN':
                mean_weight = np.mean(weights)
                std_weight = np.std(weights)
                plt.title(f"{model_name} weight distribution layer {layer} - mean: {mean_weight:.2f}, std: {std_weight:.2f}")
            else:
                plt.title(f"{model_name} weight distribution layer {layer}")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.tight_layout()
            save_tikz_and_png(plt.gcf(), f"{name}_weight_distribution", weight_dir)
        if 'param' in name:
            plt.figure(figsize=(6, 4))
            params = param.detach().cpu().numpy().flatten()
            plt.hist(params, bins=100, alpha=0.6)
            plt.title(f"{model_name} encoding param distribution layer")
            plt.xlabel("Param Value")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.tight_layout()
            save_tikz_and_png(plt.gcf(), f"{name}_param_distribution", weight_dir)
    
    for name, data in activations.items():
        if name == '':
            layer = data['input']
            layer_number = "input"
            printname = "input"
            title = f"{model_name} input activation distribution"
        elif name.split('.')[-1] == 'encoding':
            layer = data['output']
            layer_number = "encoding"
            printname = "encoding"
            title = f"{model_name} encoding activation distribution"
        elif name == 'mlp':
            continue
        else:
            layer = data['output']
            layer_number = f"layer_{int(name.split('.')[1]) + 1}"
            if name.split('.')[-1] == 'linear':
                printname = f"layer_{layer_number}_dotproduct"
                title = f"{model_name} {layer_number} dotproduct activation distribution"
            else:
                printname = f"layer_{layer_number}_activation_function"
                title = f"{model_name} {layer_number} activation function activation distribution"
            
        plot = layer.flatten().cpu().numpy()
        plt.figure(figsize=(6, 4))
        plt.hist(plot, bins=100, alpha=0.6)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_tikz_and_png(plt.gcf(), f"{printname}_activation_distribution", act_dir)

        if name in activations_gradients:
            plt.figure(figsize=(6, 4))
            grad = activations_gradients[name]
            plt.hist(grad.flatten().cpu().numpy(), bins=100, alpha=0.6)
            plt.title(f"{model_name} {layer_number} gradient distribution")
            plt.xticks(rotation=45)
            plt.tight_layout()
            save_tikz_and_png(plt.gcf(), f"{printname}_gradient_distribution", grad_dir)

def plot_compare_steps(model_name, model_dir, psnr_mean_std, ssim_mean_std, psnr_interval, output_base_dir, image_name, test_name, n_runs):
    # Save TikZ plot for PSNR based on steps
        plt.figure(figsize=(10, 5))
        for hyperparameter, (mean_values, std_values) in psnr_mean_std.items():
            x = np.arange(len(mean_values)) * psnr_interval

            # Line plot of the mean
            plt.plot(
                x,
                mean_values,
                label=hyperparameter.replace("_", " "),
                linewidth=2,
            )

            # Shaded region for ±1 std
            plt.fill_between(
                x,
                np.array(mean_values) - np.array(std_values),
                np.array(mean_values) + np.array(std_values),
                alpha=0.2
            )

        plt.xlabel("Training Steps", fontsize=16)
        plt.ylabel("PSNR", fontsize=16)
        plt.legend(loc='lower right', fontsize=10)  # Legend in the bottom right and smaller
        plt.grid(True)
        plt.title(f"Average PSNR ± Std over {n_runs} runs", fontsize=14)
        psnr_tikz_path = os.path.join(model_dir, f'{output_base_dir}_{model_name}_{image_name}_{test_name}_psnr.tex')
        os.makedirs(os.path.dirname(psnr_tikz_path), exist_ok=True)
        tikzplotlib.save(psnr_tikz_path, axis_width='15cm', axis_height='8cm') 
        plt.savefig(os.path.join(model_dir, f'{output_base_dir}_{model_name}_{image_name}_{test_name}_psnr.png'), dpi=300)
        plt.close()

       

        # Save TikZ plot for SSIM based on steps
        plt.figure(figsize=(10, 5))
        for hyperparameter, (mean_values, std_values) in ssim_mean_std.items():
            x = np.arange(len(mean_values)) * psnr_interval

            # Line plot of the mean
            plt.plot(
                x,
                mean_values,
                label=hyperparameter.replace("_", " "),
                linewidth=2,
            )

            # Shaded region for ±1 std
            plt.fill_between(
                x,
                np.array(mean_values) - np.array(std_values),
                np.array(mean_values) + np.array(std_values),
                alpha=0.2
            )

        plt.xlabel("Training Steps", fontsize=16)
        plt.ylabel("SSIM", fontsize=16)
        plt.legend(loc='lower right', fontsize=10)  # Legend in the bottom right and smaller
        plt.grid(True)
        plt.title(f"Average SSIM ± Std over {n_runs} runs", fontsize=14)
        ssim_tikz_path = os.path.join(model_dir, f'{output_base_dir}_{model_name}_{image_name}_{test_name}_ssim.tex')
        os.makedirs(os.path.dirname(ssim_tikz_path), exist_ok=True)
        tikzplotlib.save(ssim_tikz_path, axis_width='15cm', axis_height='8cm') 
        plt.savefig(os.path.join(model_dir, f'{output_base_dir}_{model_name}_{image_name}_{test_name}_ssim.png'), dpi=300)
        plt.close()
        
def plot_compare_time(model_name, model_dir, psnr_mean_std, ssim_mean_std, mean_time_ticks, psnr_interval, output_base_dir, image_name, test_name, n_runs):
         # Save TikZ plot for PSNR based on time
        plt.figure(figsize=(10, 5))
        for hyperparameter, (mean_values, std_values) in psnr_mean_std.items():
            plt.plot(
                mean_time_ticks[hyperparameter],
                mean_values,
                label=hyperparameter.replace("_", " "),
                linewidth=2,
            )

            # Shaded region for ±1 std
            plt.fill_between(
                mean_time_ticks[hyperparameter],
                np.array(mean_values) - np.array(std_values),
                np.array(mean_values) + np.array(std_values),
                alpha=0.2
            )

        plt.xlabel("Training Time (s)", fontsize=16)
        plt.ylabel("PSNR", fontsize=16)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True)
        plt.title(f"Average PSNR ± Std over {n_runs} runs (Time)", fontsize=14)
        psnr_time_tikz_path = os.path.join(model_dir, f'{output_base_dir}_{model_name}_{image_name}_{test_name}_psnr_time.tex')
        os.makedirs(os.path.dirname(psnr_time_tikz_path), exist_ok=True)
        tikzplotlib.save(psnr_time_tikz_path, axis_width='15cm', axis_height='8cm') 
        plt.savefig(os.path.join(model_dir, f'{output_base_dir}_{model_name}_{image_name}_{test_name}_psnr_time.png'), dpi=300)
        plt.close()

        # Save TikZ plot for SSIM based on time
        plt.figure(figsize=(10, 5))
        for hyperparameter, (mean_values, std_values) in ssim_mean_std.items():
            plt.plot(
                mean_time_ticks[hyperparameter],
                mean_values,
                label=hyperparameter.replace("_", " "),
                linewidth=2,
            )

            # Shaded region for ±1 std
            plt.fill_between(
                mean_time_ticks[hyperparameter],
                np.array(mean_values) - np.array(std_values),
                np.array(mean_values) + np.array(std_values),
                alpha=0.2
            )

        plt.xlabel("Training Time (s)", fontsize=16)
        plt.ylabel("SSIM", fontsize=16)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True)
        plt.title(f"Average SSIM ± Std over {n_runs} runs (Time)", fontsize=14)
        ssim_time_tikz_path = os.path.join(model_dir, f'{output_base_dir}_{model_name}_{image_name}_{test_name}_ssim_time.tex')
        os.makedirs(os.path.dirname(ssim_time_tikz_path), exist_ok=True)
        tikzplotlib.save(ssim_time_tikz_path, axis_width='15cm', axis_height='8cm') 
        plt.savefig(os.path.join(model_dir, f'{output_base_dir}_{model_name}_{image_name}_{test_name}_ssim_time.png'), dpi=300)
        plt.close()

def save_metrics_to_csv(psnr_mean_std, ssim_mean_std, mean_time_ticks, model_dir, image_name):
            csv_file = os.path.join(model_dir, f"{image_name}_metrics.csv")
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Hyperparameter", "PSNR Mean", "PSNR Std", "SSIM Mean", "SSIM Std", "Time Ticks"])
                for hyperparameter in psnr_mean_std.keys():
                    psnr_mean, psnr_std = psnr_mean_std[hyperparameter]
                    ssim_mean, ssim_std = ssim_mean_std[hyperparameter]
                    time_ticks = mean_time_ticks[hyperparameter]
                    writer.writerow([hyperparameter, psnr_mean.tolist(), psnr_std.tolist(), ssim_mean.tolist(), ssim_std.tolist(), time_ticks.tolist()])


def plot_all_activations_spectrum_and_gradients_weights_with_tikz(model, model_dir, activations, activations_gradients):
    act_dir = os.path.join(model_dir, "activations")
    os.makedirs(act_dir, exist_ok=True)
    grad_dir = os.path.join(model_dir, "gradients")
    os.makedirs(grad_dir, exist_ok=True)
    weight_dir = os.path.join(model_dir, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.figure(figsize=(6, 4))
            weights = param.detach().cpu().numpy().flatten()
            plt.hist(weights, bins=100, alpha=0.6, label=f"{name} weights")
            plt.title(f"Weight Distribution - {name}")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            save_tikz_and_png(plt.gcf(), f"{name}_weight_distribution", weight_dir)
        if 'param' in name:
            plt.figure(figsize=(6, 4))
            params = param.detach().cpu().numpy().flatten()
            plt.hist(params, bins=100, alpha=0.6, label=f"{name} params")
            plt.title(f"Param Distribution - {name}")
            plt.xlabel("Param Value")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            save_tikz_and_png(plt.gcf(), f"{name}_param_distribution", weight_dir)
    
    for name, data in activations.items():
        if name == '':
            layer = data['input']
            printname = "input"
        elif name.split('.')[-1] == 'encoding':
            layer = data['output']
            printname = "encoding"
        elif name == 'mlp':
            printname = "mlp"
        else:
            layer = data['output']
            layer_number = str(int(name.split('.')[1]) + 1)
            if name.split('.')[-1] == 'linear':
                printname = f"layer_{layer_number}_dotproduct"
            else:
                printname = f"layer_{layer_number}_activation_function"
        input_layer = data['input']
        output_layer = data['output']
        
        input_plot = input_layer.flatten().cpu().numpy()
        plt.figure(figsize=(6, 4))
        plt.hist(input_plot, bins=100, alpha=0.6, label=f"{printname} input")
        plt.title(f"Input Activation Distribution")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        save_tikz_and_png(plt.gcf(), f"{printname}_input_activation_distribution", act_dir)
        
        output_plot = output_layer.flatten().cpu().numpy()
        plt.figure(figsize=(6, 4))
        plt.hist(output_plot, bins=100, alpha=0.6, label=f"{printname} output")
        plt.title(f"Output Activation Distribution")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        save_tikz_and_png(plt.gcf(), f"{printname}_output_activation_distribution", act_dir)
        
        if name in activations_gradients:
            plt.figure(figsize=(6, 4))
            grad = activations_gradients[name]
            plt.hist(grad.flatten().cpu().numpy(), bins=100, alpha=0.6, label=printname)
            plt.title(f"Gradient Distribution")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            save_tikz_and_png(plt.gcf(), f"{printname}_gradient_distribution", grad_dir)

@contextmanager
def capture_activations(model: nn.Module, hook_fn: Callable, activations: dict, activations_gradients: dict):
    handles = []
    for name, module in model.named_modules():
        h = module.register_forward_hook(hook_fn(activations, activations_gradients, name))
        handles.append(h)
    try:
        yield
    finally:
        for h in handles:
            h.remove()

def record_activation(activations: dict, activations_gradients: dict, name: str):
    def hook(module, input, output):
        activations[name] = {
            "input": input[0].detach(),  
            "output": output.detach()
        }
        # Register backward hook to capture gradients
        output.requires_grad_(True) 
        def grad_hook(grad):
            activations_gradients[name] = grad.detach()
        output.register_hook(grad_hook)
    return hook
    

def transform_normalized(image):
    transform = Compose([
        Resize((image.height, image.width)),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    return transform(image)


grad = lambda k: 1j * k * 2 * np.pi
amplitude_grad = lambda kx, ky: 1j * torch.sqrt(ky**2 + kx**2) * 2 * np.pi
low_pass_frequency = lambda kx, ky, PERCENTAGE_LOW: torch.where(torch.sqrt(ky**2 + kx**2) > PERCENTAGE_LOW, 0, 1)
high_pass_frequency = lambda kx, ky, PERCENTAGE_LOW: torch.where(torch.sqrt(ky**2 + kx**2) < PERCENTAGE_LOW, 0, 1)
KERNEL_X = torch.tensor([[-1, 0, 1]], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
KERNEL_Y = torch.tensor([[-1], [0], [1]], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
SOBEL_X = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
SOBEL_Y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
   
    
# Visualization functions

def visualize_grad_loss(model_output, resolutions, output_dir="gradient_visualizations", step=0):
    """Visualize gradients computed using torch.gradient method"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_output_2d = model_output.reshape(resolutions[0], resolutions[1])
    grad_x = torch.gradient(model_output_2d, dim=0)[0]
    grad_y = torch.gradient(model_output_2d, dim=1)[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    im1 = axes[0].imshow(model_output_2d.detach().cpu().numpy(), cmap='gray')
    axes[0].set_title('Original Output')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # X gradient
    im2 = axes[1].imshow(grad_x.detach().cpu().numpy(), cmap='gray')
    axes[1].set_title('X Gradient (torch.gradient)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Y gradient
    im3 = axes[2].imshow(grad_y.detach().cpu().numpy(), cmap='gray')
    axes[2].set_title('Y Gradient (torch.gradient)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, f'grad_loss_visualization_step_{step}.png'), dpi=300, bbox_inches='tight')
    
    # Save as TikZ
    tikzplotlib.save(os.path.join(output_dir, f'grad_loss_visualization_step_{step}.tex'))
    
    plt.close()

def visualize_auto_grad_loss(model_output, resolutions, model_input, output_dir="gradient_visualizations", step=0):
    """Visualize gradients computed using autograd method"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute gradients using torch.autograd.grad
    grad_outputs = torch.ones_like(model_output)
    gradients = torch.autograd.grad(
        outputs=model_output,
        inputs=model_input,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]
    
    grad_x = gradients[:, 0].reshape(resolutions[0], resolutions[1])
    grad_y = gradients[:, 1].reshape(resolutions[0], resolutions[1])
    model_output_2d = model_output.reshape(resolutions[0], resolutions[1])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    im1 = axes[0].imshow(model_output_2d.detach().cpu().numpy(), cmap='gray')
    axes[0].set_title('Original Output')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # X gradient
    im2 = axes[1].imshow(grad_x.detach().cpu().numpy(), cmap='gray')
    axes[1].set_title('X Gradient (autograd)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Y gradient
    im3 = axes[2].imshow(grad_y.detach().cpu().numpy(), cmap='gray')
    axes[2].set_title('Y Gradient (autograd)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, f'auto_grad_loss_visualization_step_{step}.png'), dpi=300, bbox_inches='tight')
    
    # Save as TikZ
    tikzplotlib.save(os.path.join(output_dir, f'auto_grad_loss_visualization_step_{step}.tex'))
    
    plt.close()

def visualize_sobel_grad_loss(model_output, resolutions, sobel_x=SOBEL_X, sobel_y=SOBEL_Y, output_dir="gradient_visualizations", step=0):
    """Visualize gradients computed using Sobel filters"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_output_2d = model_output.reshape(resolutions[0], resolutions[1])
    
    
    grad_x = torch.nn.functional.conv2d(model_output_2d.unsqueeze(0).unsqueeze(0), sobel_x, padding=1).squeeze()
    grad_y = torch.nn.functional.conv2d(model_output_2d.unsqueeze(0).unsqueeze(0), sobel_y, padding=1).squeeze()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    im1 = axes[0].imshow(model_output_2d.detach().cpu().numpy(), cmap='gray')
    axes[0].set_title('Original Output')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    
    # X gradient result
    im2 = axes[1].imshow(grad_x.detach().cpu().numpy(), cmap='gray')
    axes[1].set_title('X Gradient (Sobel)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Y gradient result
    im3 = axes[2].imshow(grad_y.detach().cpu().numpy(), cmap='gray')
    axes[2].set_title('Y Gradient (Sobel)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, f'sobel_grad_loss_visualization_step_{step}.png'), dpi=300, bbox_inches='tight')
    
    # Save as TikZ
    tikzplotlib.save(os.path.join(output_dir, f'sobel_grad_loss_visualization_step_{step}.tex'))
    
    plt.close()

def visualize_kernel_grad_loss(model_output, resolutions, kernel_x=KERNEL_X, kernel_y=KERNEL_Y, output_dir="gradient_visualizations", step=0):
    """Visualize gradients computed using simple kernel filters"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_output_2d = model_output.reshape(resolutions[0], resolutions[1])

    
    grad_x = torch.nn.functional.conv2d(model_output_2d.unsqueeze(0).unsqueeze(0), kernel_x, padding=(0, 1)).squeeze()
    grad_y = torch.nn.functional.conv2d(model_output_2d.unsqueeze(0).unsqueeze(0), kernel_y, padding=(1, 0)).squeeze()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    im1 = axes[0].imshow(model_output_2d.detach().cpu().numpy(), cmap='gray')
    axes[0].set_title('Original Output')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # X gradient result
    im2 = axes[1].imshow(grad_x.detach().cpu().numpy(), cmap='gray')
    axes[1].set_title('X Gradient (Kernel)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Y gradient result
    im3 = axes[2].imshow(grad_y.detach().cpu().numpy(), cmap='gray')
    axes[2].set_title('Y Gradient (Kernel)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, f'kernel_grad_loss_visualization_step_{step}.png'), dpi=300, bbox_inches='tight')
    
    # Save as TikZ
    tikzplotlib.save(os.path.join(output_dir, f'kernel_grad_loss_visualization_step_{step}.tex'))
    
    plt.close()

def visualize_fourier_amplitude_grad_loss(model_output, resolutions, grad_fn=amplitude_grad, output_dir="gradient_visualizations", step=0):
    """Visualize gradients computed using Fourier amplitude gradient method"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_output_2d = model_output.reshape(resolutions[0], resolutions[1])
    
    # Create frequency grids
    kx, ky = torch.meshgrid(
        torch.fft.fftfreq(resolutions[0]),
        torch.fft.fftfreq(resolutions[1]),
        indexing='ij'
    )
    kx, ky = torch.fft.fftshift(kx.cuda()), torch.fft.fftshift(ky.cuda())
    
    # Compute FFT
    fourier_image = torch.fft.fftshift(torch.fft.fft2(model_output_2d))
    
    # Apply amplitude gradient filter
    amplitude_filter = grad_fn(kx, ky)
    fourier_filtered = fourier_image * amplitude_filter
    
    # Convert back to spatial domain
    spatial_filtered = torch.fft.ifft2(torch.fft.ifftshift(fourier_filtered)).real
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    im1 = axes[0, 0].imshow(model_output_2d.detach().cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Original Output')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # FFT magnitude
    im2 = axes[0, 1].imshow(torch.log(torch.abs(fourier_image) + 1e-8).detach().cpu().numpy(), cmap='hot')
    axes[0, 1].set_title('FFT Magnitude (log)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Amplitude gradient filter
    im3 = axes[1, 0].imshow(torch.abs(amplitude_filter).detach().cpu().numpy(), cmap='viridis')
    axes[1, 0].set_title('Amplitude Gradient Filter')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Filtered result
    im4 = axes[1, 1].imshow(spatial_filtered.detach().cpu().numpy(), cmap='gray')
    axes[1, 1].set_title('Amplitude Gradient Result')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, f'fourier_amplitude_grad_loss_visualization_step_{step}.png'), dpi=300, bbox_inches='tight')
    
    # Save as TikZ
    tikzplotlib.save(os.path.join(output_dir, f'fourier_amplitude_grad_loss_visualization_step_{step}.tex'))
    
    plt.close()

def visualize_fourier_amplitude_grad_loss_inverse(model_output, resolutions, grad_fn=amplitude_grad, output_dir="gradient_visualizations", step=0):
    """Visualize gradients computed using Fourier amplitude gradient inverse method"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_output_2d = model_output.reshape(resolutions[0], resolutions[1])
    
    # Create frequency grids
    kx, ky = torch.meshgrid(
        torch.fft.fftfreq(resolutions[0]),
        torch.fft.fftfreq(resolutions[1]),
        indexing='ij'
    )
    kx, ky = torch.fft.fftshift(kx.cuda()), torch.fft.fftshift(ky.cuda())
    
    # Compute FFT
    fourier_image = torch.fft.fftshift(torch.fft.fft2(model_output_2d))
    
    # Apply amplitude gradient filter
    amplitude_filter = grad_fn(kx, ky)
    fourier_filtered = fourier_image * amplitude_filter
    
    # Convert back to spatial domain
    spatial_filtered = torch.fft.ifft2(torch.fft.ifftshift(fourier_filtered)).real
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    im1 = axes[0, 0].imshow(model_output_2d.detach().cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Original Output')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # FFT magnitude
    im2 = axes[0, 1].imshow(torch.log(torch.abs(fourier_image) + 1e-8).detach().cpu().numpy(), cmap='hot')
    axes[0, 1].set_title('FFT Magnitude (log)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Amplitude gradient filter
    im3 = axes[1, 0].imshow(torch.abs(amplitude_filter).detach().cpu().numpy(), cmap='viridis')
    axes[1, 0].set_title('Amplitude Gradient Filter')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Filtered result
    im4 = axes[1, 1].imshow(spatial_filtered.detach().cpu().numpy(), cmap='gray')
    axes[1, 1].set_title('Amplitude Gradient Result')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, f'fourier_amplitude_grad_loss_visualization_step_{step}.png'), dpi=300, bbox_inches='tight')
    
    # Save as TikZ
    tikzplotlib.save(os.path.join(output_dir, f'fourier_amplitude_grad_loss_visualization_step_{step}.tex'))
    
    plt.close()

def visualize_fourier_weighted_frequency_loss(model_output, resolutions, grad_fn_low=low_pass_frequency, grad_fn_high=high_pass_frequency, PERCENTAGE_LOW=0, output_dir="gradient_visualizations", step=0):
    """Visualize gradients computed using Fourier weighted frequency method"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_output_2d = model_output.reshape(resolutions[0], resolutions[1])
    
    # Create frequency grids
    kx, ky = torch.meshgrid(
        torch.fft.fftfreq(resolutions[0]),
        torch.fft.fftfreq(resolutions[1]),
        indexing='ij'
    )
    kx, ky = torch.fft.fftshift(kx.cuda()), torch.fft.fftshift(ky.cuda())
    
    # Compute FFT
    fourier_image = torch.fft.fftshift(torch.fft.fft2(model_output_2d))
    
    # Apply frequency filters
    low_pass_filter = low_pass_frequency(kx, ky, PERCENTAGE_LOW)
    high_pass_filter = high_pass_frequency(kx, ky, PERCENTAGE_LOW)
    
    fourier_low = fourier_image * low_pass_filter
    fourier_high = fourier_image * high_pass_filter
    

    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    im1 = axes[0, 0].imshow(model_output_2d.detach().cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Original Output')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Low pass filter
    im2 = axes[0, 1].imshow(low_pass_filter.detach().cpu().numpy(), cmap='gray')
    axes[0, 1].set_title('Low Pass Filter')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # High pass filter
    im3 = axes[0, 2].imshow(high_pass_filter.detach().cpu().numpy(), cmap='gray')
    axes[0, 2].set_title('High Pass Filter')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Low frequency component (spatial)
    im4 = axes[1, 0].imshow(torch.log(torch.abs(fourier_low) + 1e-8).detach().cpu().numpy(), cmap='hot')
    axes[1, 0].set_title('Low Frequency Component')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # High frequency component (spatial)
    im5 = axes[1, 1].imshow(torch.log(torch.abs(fourier_high) + 1e-8).detach().cpu().numpy(), cmap='hot')
    axes[1, 1].set_title('High Frequency Component')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Fourier magnitude
    im6 = axes[1, 2].imshow(torch.log(torch.abs(fourier_image) + 1e-8).detach().cpu().numpy(), cmap='hot')
    axes[1, 2].set_title('FFT Magnitude (log)')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, f'fourier_weighted_frequency_loss_visualization_step_{step}.png'), dpi=300, bbox_inches='tight')
    
    # Save as TikZ
    tikzplotlib.save(os.path.join(output_dir, f'fourier_weighted_frequency_loss_visualization_step_{step}.tex'))
    
    plt.close()

def visualize_fourier_weighted_frequency_loss_inverse(model_output, resolutions, grad_fn_low=low_pass_frequency, grad_fn_high=high_pass_frequency, PERCENTAGE_LOW=0, output_dir="gradient_visualizations", step=0):
    """Visualize gradients computed using Fourier weighted frequency inverse method"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_output_2d = model_output.reshape(resolutions[0], resolutions[1])
    
    # Create frequency grids
    kx, ky = torch.meshgrid(
        torch.fft.fftfreq(resolutions[0]),
        torch.fft.fftfreq(resolutions[1]),
        indexing='ij'
    )
    kx, ky = torch.fft.fftshift(kx.cuda()), torch.fft.fftshift(ky.cuda())
    
    # Compute FFT
    fourier_image = torch.fft.fftshift(torch.fft.fft2(model_output_2d))
    
    # Apply frequency filters
    low_pass_filter = low_pass_frequency(kx, ky, PERCENTAGE_LOW)
    high_pass_filter = high_pass_frequency(kx, ky, PERCENTAGE_LOW)
    
    fourier_low = fourier_image * low_pass_filter
    fourier_high = fourier_image * high_pass_filter
    
    # Convert to spatial domain (inverse operation)
    spatial_low_inverse = torch.real(torch.fft.ifft2(torch.fft.ifftshift(fourier_low)))
    spatial_high_inverse = torch.real(torch.fft.ifft2(torch.fft.ifftshift(fourier_high)))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    im1 = axes[0, 0].imshow(model_output_2d.detach().cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Original Output')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Frequency domain representation
    im2 = axes[0, 1].imshow(torch.log(torch.abs(fourier_low) + 1e-8).detach().cpu().numpy(), cmap='hot')
    axes[0, 1].set_title('Low Freq. Fourier (log)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(torch.log(torch.abs(fourier_high) + 1e-8).detach().cpu().numpy(), cmap='hot')
    axes[0, 2].set_title('High Freq. Fourier (log)')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Inverse spatial domain results
    im4 = axes[1, 0].imshow(spatial_low_inverse.detach().cpu().numpy(), cmap='gray')
    axes[1, 0].set_title('Low Freq. Inverse')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(spatial_high_inverse.detach().cpu().numpy(), cmap='gray')
    axes[1, 1].set_title('High Freq. Inverse')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Reconstructed image (low + high)
    reconstructed = spatial_low_inverse + spatial_high_inverse
    im6 = axes[1, 2].imshow(reconstructed.detach().cpu().numpy(), cmap='gray')
    axes[1, 2].set_title('Reconstructed (Low+High)')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, f'fourier_weighted_frequency_loss_inverse_visualization_step_{step}.png'), dpi=300, bbox_inches='tight')
    
    # Save as TikZ
    tikzplotlib.save(os.path.join(output_dir, f'fourier_weighted_frequency_loss_inverse_visualization_step_{step}.tex'))
    
    plt.close()

def visualize_fourier_grad_loss(model_output, resolutions, grad_fn=grad, output_dir="gradient_visualizations", step=0):
    """Visualize gradients computed using Fourier domain method for grad loss"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_output_2d = model_output.reshape(resolutions[0], resolutions[1])
    
    # Create frequency grids
    kx, ky = torch.meshgrid(
        torch.fft.fftfreq(resolutions[0]),
        torch.fft.fftfreq(resolutions[1]),
        indexing='ij'
    )
    kx, ky = torch.fft.fftshift(kx.cuda()), torch.fft.fftshift(ky.cuda())
    
    # Compute FFT
    fourier_image = torch.fft.fftshift(torch.fft.fft2(model_output_2d))
    
    # Apply gradient operators in frequency domain
    image_x = fourier_image * grad_fn(kx)
    image_y = fourier_image * grad_fn(ky)
    
    # Convert back to spatial domain
    grad_x = torch.real(torch.fft.ifft2(torch.fft.ifftshift(image_x)))
    grad_y = torch.real(torch.fft.ifft2(torch.fft.ifftshift(image_y)))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    im1 = axes[0, 0].imshow(model_output_2d.detach().cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Original Output')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # FFT magnitude
    im2 = axes[0, 1].imshow(torch.log(torch.abs(fourier_image) + 1e-8).detach().cpu().numpy(), cmap='hot')
    axes[0, 1].set_title('FFT Magnitude (log)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])

    # Fourier X gradient
    im3 = axes[0, 2].imshow(torch.log(torch.abs(image_x) + 1e-8).detach().cpu().numpy(), cmap='hot')
    axes[0, 2].set_title('Fourier X Gradient (log)')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])

    # Fourier Y gradient
    im4 = axes[1, 0].imshow(torch.log(torch.abs(image_y) + 1e-8).detach().cpu().numpy(), cmap='hot')
    axes[1, 0].set_title('Fourier Y Gradient (log)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0])

    # Inverse X gradient
    im5 = axes[1, 1].imshow(grad_x.detach().cpu().numpy(), cmap='gray')
    axes[1, 1].set_title('Inverse X Gradient')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Inverse Y gradient
    im6 = axes[1, 2].imshow(grad_y.detach().cpu().numpy(), cmap='gray')
    axes[1, 2].set_title('Inverse Y Gradient')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, f'fourier_grad_loss_visualization_step_{step}.png'), dpi=300, bbox_inches='tight')
    
    # Save as TikZ
    tikzplotlib.save(os.path.join(output_dir, f'fourier_grad_loss_visualization_step_{step}.tex'))
    
    plt.close()

def visualize_fourier_grad_loss_inverse(model_output, resolutions, grad_fn=grad, output_dir="gradient_visualizations", step=0):
    """Visualize gradients computed using Fourier domain method for grad loss inverse"""
    os.makedirs(output_dir, exist_ok=True)
    print(resolutions)
    model_output_2d = model_output.reshape(resolutions[0], resolutions[1])
    # Create frequency grids
    kx, ky = torch.meshgrid(
        torch.fft.fftfreq(resolutions[0]),
        torch.fft.fftfreq(resolutions[1]),
        indexing='ij'
    )
    kx, ky = torch.fft.fftshift(kx.cuda()), torch.fft.fftshift(ky.cuda())
    
    # Compute FFT
    fourier_image = torch.fft.fftshift(torch.fft.fft2(model_output_2d))
    
    # Apply gradient operators in frequency domain
    image_x = fourier_image * grad_fn(kx)
    image_y = fourier_image * grad_fn(ky)
    
    # Convert back to spatial domain
    grad_x = torch.real(torch.fft.ifft2(torch.fft.ifftshift(image_x)))
    grad_y = torch.real(torch.fft.ifft2(torch.fft.ifftshift(image_y)))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    im1 = axes[0, 0].imshow(model_output_2d.detach().cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Original Output')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # FFT magnitude
    im2 = axes[0, 1].imshow(torch.log(torch.abs(fourier_image) + 1e-8).detach().cpu().numpy(), cmap='hot')
    axes[0, 1].set_title('FFT Magnitude (log)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])

    # Fourier X gradient
    im3 = axes[0, 2].imshow(torch.log(torch.abs(image_x) + 1e-8).detach().cpu().numpy(), cmap='hot')
    axes[0, 2].set_title('Fourier X Gradient (log)')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])

    # Fourier Y gradient
    im4 = axes[1, 0].imshow(torch.log(torch.abs(image_y) + 1e-8).detach().cpu().numpy(), cmap='hot')
    axes[1, 0].set_title('Fourier Y Gradient (log)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0])

    # Inverse X gradient
    im5 = axes[1, 1].imshow(grad_x.detach().cpu().numpy(), cmap='gray')
    axes[1, 1].set_title('Inverse X Gradient')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Inverse Y gradient
    im6 = axes[1, 2].imshow(grad_y.detach().cpu().numpy(), cmap='gray')
    axes[1, 2].set_title('Inverse Y Gradient')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, f'fourier_grad_loss_inverse_visualization_step_{step}.png'), dpi=300, bbox_inches='tight')
    
    # Save as TikZ
    tikzplotlib.save(os.path.join(output_dir, f'fourier_grad_loss_inverse_visualization_step_{step}.tex'))
    
    plt.close()




#Loss functions

def fourier_grad_loss(model_output, kx, ky, fourier_gt_grad_x, fourier_gt_grad_y, fourier_gt, grad_fn=grad, LAMBDA=0, step=1, print_loss_interval=1000, constant=1/100000):
    fourier_image = torch.fft.fftshift(torch.fft.fft2(model_output.reshape(fourier_gt.shape[0], fourier_gt.shape[1])))
    
    image_x = fourier_image * grad_fn(kx)
    image_y = fourier_image * grad_fn(ky)
    
    difference_x = image_x - fourier_gt_grad_x
    difference_y = image_y - fourier_gt_grad_y
    difference = fourier_image - fourier_gt
    
    x_loss = ((difference_x.real)**2).mean()*constant + ((difference_x.imag)**2).mean()*constant
    y_loss = ((difference_y.real)**2).mean()*constant + ((difference_y.imag)**2).mean()*constant
    fourier_loss = ((difference.real)**2).mean()*constant + ((difference.imag)**2).mean()*constant
    
    total_loss = (1-LAMBDA) * x_loss + (1-LAMBDA) * y_loss + LAMBDA * fourier_loss
    
    if step % print_loss_interval == 0:
        print(f"Step {step}: x_loss = {x_loss.item()}, y_loss = {y_loss.item()}, fourier_loss = {fourier_loss.item()}, total_loss = {total_loss.item()}")
    
    return total_loss * constant

def fourier_grad_loss_inverse(model_output, image_tensor, kx, ky, fourier_gt_grad_x, fourier_gt_grad_y, grad_fn=grad, LAMBDA=0, step=1, print_loss_interval=1000, constant=100):
    fourier_image = torch.fft.fftshift(torch.fft.fft2(model_output.reshape(image_tensor.shape[0], image_tensor.shape[1])))
    image_x = fourier_image * grad_fn(kx)
    image_y = fourier_image * grad_fn(ky)
    difference_x = image_x - fourier_gt_grad_x
    difference_y = image_y - fourier_gt_grad_y
    inverse_difference_x = torch.real(torch.fft.ifft2(torch.fft.ifftshift(difference_x)))
    inverse_difference_y = torch.real(torch.fft.ifft2(torch.fft.ifftshift(difference_y)))   
    loss_x = ((inverse_difference_x)**2).mean()*constant
    loss_y = ((inverse_difference_y)**2).mean()*constant
    loss_normal = ((model_output.flatten() - image_tensor.flatten())**2).mean()
    total_loss = (1-LAMBDA) * loss_x + (1-LAMBDA) * loss_y + LAMBDA * loss_normal
    
    if step % print_loss_interval == 0:
        print(f"Step {step}: loss_x = {loss_x.item()}, loss_y = {loss_y.item()}, loss_normal = {loss_normal.item()}, total_loss = {total_loss.item()}")
    
    return total_loss

def normal_loss(model_output, image_tensor, step=1, print_loss_interval=1000, constant=1):
    loss = ((model_output.flatten() - image_tensor.flatten())**2).mean()*constant
    if step % print_loss_interval == 0:
        print(f"Step {step}: loss = {loss.item()}")
    return loss

def fourier_amplitude_grad_loss(model_output, kx, ky, fourier_amplitude_gt, fourier_gt, grad_fn=amplitude_grad, LAMBDA=0, step=1, print_loss_interval=1000, constant=1/100000):
    fourier_image = torch.fft.fftshift(torch.fft.fft2(model_output.reshape(fourier_gt.shape[0], fourier_gt.shape[1])))
    fourier_filtered = fourier_image * grad_fn(kx, ky)
    grad_difference = fourier_filtered - fourier_amplitude_gt
    fourier_difference = fourier_image - fourier_gt
    loss_grad_real = ((grad_difference.real)**2).mean()*constant
    loss_grad_imag = ((grad_difference.imag)**2).mean()*constant
    loss_fourier_real = ((fourier_difference.real)**2).mean()*constant
    loss_fourier_imag = ((fourier_difference.imag)**2).mean()*constant
    total_loss = (1-LAMBDA) * (loss_grad_real + loss_grad_imag) + LAMBDA * (loss_fourier_real + loss_fourier_imag)
    
    if step % print_loss_interval == 0:
        print(f"Step {step}: loss_grad_real = {loss_grad_real.item()}, loss_grad_imag = {loss_grad_imag.item()}, loss_fourier_real = {loss_fourier_real.item()}, loss_fourier_imag = {loss_fourier_imag.item()}, total_loss = {total_loss.item()}")
    
    return total_loss

def fourier_amplitude_grad_loss_inverse(model_output, image_tensor, kx, ky, fourier_amplitude_gt, grad_fn=grad, LAMBDA=0, step=1, print_loss_interval=1000, constant=10000000000000):
    fourier_image = torch.fft.fftshift(torch.fft.fft2(model_output.reshape(image_tensor.shape[0], image_tensor.shape[1])))
    image_filtered = fourier_image * grad_fn(kx, ky)
    difference = image_filtered - fourier_amplitude_gt
    inverse_difference = torch.real(torch.fft.ifft2(torch.fft.ifftshift(difference)))
    loss_inverse = ((inverse_difference)**2).mean()*constant
    loss_normal = ((model_output.flatten() - image_tensor.flatten())**2).mean()
    total_loss = (1-LAMBDA) * loss_inverse + LAMBDA * loss_normal
    
    if step % print_loss_interval == 0:
        print(f"Step {step}: loss_inverse = {loss_inverse.item()}, loss_normal = {loss_normal.item()}, total_loss = {total_loss.item()}")
    
    return total_loss

def fourier_weighted_frequency_loss(model_output, kx, ky, fourier_gt_low, fourier_gt_high, grad_fn_low=low_pass_frequency, grad_fn_high=high_pass_frequency, LAMBDA_LOW=0, PERCENTAGE_LOW=0, step=1, print_loss_interval=1000, constant=1/30000):
    
    fourier_image = torch.fft.fftshift(torch.fft.fft2(model_output.reshape(fourier_gt_low.shape[0], fourier_gt_low.shape[1])))
        
    f_low = fourier_image * grad_fn_low(kx, ky, PERCENTAGE_LOW)
    f_high = fourier_image * grad_fn_high(kx, ky, PERCENTAGE_LOW)
    difference_high = f_high - fourier_gt_high
    difference_low = f_low - fourier_gt_low
    high_loss_real = ((difference_high.real)**2).mean()*constant
    high_loss_imag = ((difference_high.imag)**2).mean()*constant
    low_loss_real = ((difference_low.real)**2).mean()*constant
    low_loss_imag = ((difference_low.imag)**2).mean()*constant
    
    total_loss = (1-LAMBDA_LOW) * (high_loss_real + high_loss_imag) + LAMBDA_LOW * (low_loss_real + low_loss_imag)
    
    if step % print_loss_interval == 0:
        print(f"Step {step}: high_loss_real = {high_loss_real.item()}, high_loss_imag = {high_loss_imag.item()}, low_loss_real = {low_loss_real.item()}, low_loss_imag = {low_loss_imag.item()}, total_loss = {total_loss.item()}")
    
    return total_loss

def fourier_weighted_frequency_loss_inverse(model_output, kx, ky, fourier_gt_low_inverse, fourier_gt_high_inverse, grad_fn_low=low_pass_frequency, grad_fn_high=high_pass_frequency, LAMBDA_LOW=0, PERCENTAGE_LOW=0, step=1, print_loss_interval=1000, constant=100):
    
    fourier_image = torch.fft.fftshift(torch.fft.fft2(model_output.reshape(fourier_gt_low_inverse.shape[0], fourier_gt_low_inverse.shape[1])))
    
    f_low = fourier_image * grad_fn_low(kx, ky, PERCENTAGE_LOW)
    f_high = fourier_image * grad_fn_high(kx, ky, PERCENTAGE_LOW)
    
    f_low_inverse = torch.real(torch.fft.ifft2(torch.fft.ifftshift(f_low)))
    f_high_inverse = torch.real(torch.fft.ifft2(torch.fft.ifftshift(f_high)))
    
    difference_low = f_low_inverse - fourier_gt_low_inverse
    difference_high = f_high_inverse - fourier_gt_high_inverse
    
    loss_low = ((difference_low)**2).mean()
    loss_high = ((difference_high)**2).mean()
    
    total_loss = LAMBDA_LOW * loss_low + (1-LAMBDA_LOW) * loss_high
    
    if step % print_loss_interval == 0:
        print(f"Step {step}: loss_low = {loss_low.item()}, loss_high = {loss_high.item()}, total_loss = {total_loss.item()}")
    
    return total_loss

def sobel_grad_loss(model_output, image_tensor, sobel_gt_grad_x, sobel_gt_grad_y, LAMBDA=0, SOBEL_X=SOBEL_X, SOBEL_Y=SOBEL_Y, step=1, print_loss_interval=1000, constant = 1):
    # Reshape once and reuse
    model_output_2d = model_output.reshape(image_tensor.shape[0], image_tensor.shape[1])
    model_output_4d = model_output_2d.unsqueeze(0).unsqueeze(0)
    
    # Use pre-computed Sobel kernels
    model_output_x = torch.nn.functional.conv2d(model_output_4d, SOBEL_X, padding=1)
    model_output_y = torch.nn.functional.conv2d(model_output_4d, SOBEL_Y, padding=1)
    
    # Compute differences and loss efficiently
    diff_x = model_output_x - sobel_gt_grad_x
    diff_y = model_output_y - sobel_gt_grad_y
    diff_main = model_output.flatten() - image_tensor.flatten()
    
    # Use fused operations
    loss_x = diff_x.pow(2).mean()*constant
    loss_y = diff_y.pow(2).mean()*constant
    loss_main = diff_main.pow(2).mean()
    total_loss = (1-LAMBDA) * (loss_x + loss_y) + LAMBDA * loss_main
    
    if step % print_loss_interval == 0:
        print(f"Step {step}: loss_x = {loss_x.item()}, loss_y = {loss_y.item()}, loss_main = {loss_main.item()}, total_loss = {total_loss.item()}")
    
    return total_loss

def grad_loss(model_output, image_tensor, gt_grad_x, gt_grad_y, LAMBDA=0, step=1, print_loss_interval=1000, constant=100):
    model_output = model_output.reshape(image_tensor.shape[0], image_tensor.shape[1])
    grad_x = torch.gradient(model_output, dim=0)[0]
    grad_y = torch.gradient(model_output, dim=1)[0]
    difference_x = grad_x - gt_grad_x
    difference_y = grad_y - gt_grad_y
    loss_x = ((difference_x)**2).mean()*constant
    loss_y = ((difference_y)**2).mean()*constant
    loss_normal = ((model_output.flatten() - image_tensor.flatten())**2).mean()
    total_loss = (1-LAMBDA) * loss_x + (1-LAMBDA) * loss_y + LAMBDA * loss_normal
    
    if step % print_loss_interval == 0:
        print(f"Step {step}: loss_x = {loss_x.item()}, loss_y = {loss_y.item()}, loss_normal = {loss_normal.item()}, total_loss = {total_loss.item()}")
    
    return total_loss

def kernel_grad_loss(model_output, image_tensor, kernel_gt_grad_x, kernel_gt_grad_y, LAMBDA=0, KERNEL_X=KERNEL_X, KERNEL_Y=KERNEL_Y, step=1, print_loss_interval=1000, constant=100):
    # Reshape once and reuse
    model_output_2d = model_output.reshape(image_tensor.shape[0], image_tensor.shape[1])
    model_output_4d = model_output_2d.unsqueeze(0).unsqueeze(0)
    
    # Use pre-computed kernels
    model_output_x = torch.nn.functional.conv2d(model_output_4d, KERNEL_X, padding=(0, 1))
    model_output_y = torch.nn.functional.conv2d(model_output_4d, KERNEL_Y, padding=(1, 0))
    
    # Compute differences and loss in one go
    diff_x = model_output_x - kernel_gt_grad_x
    diff_y = model_output_y - kernel_gt_grad_y
    diff_main = model_output.flatten() - image_tensor.flatten()
    
    # Use fused operations
    loss_x = diff_x.pow(2).mean()*constant
    loss_y = diff_y.pow(2).mean()*constant
    loss_main = diff_main.pow(2).mean()
    total_loss = (1-LAMBDA) * (loss_x + loss_y) + LAMBDA * loss_main
    
    if step % print_loss_interval == 0:
        print(f"Step {step}: loss_x = {loss_x.item()}, loss_y = {loss_y.item()}, loss_main = {loss_main.item()}, total_loss = {total_loss.item()}")
    
    return total_loss

def auto_grad_loss(model_output, image_tensor, model_input, gt_grad_x, gt_grad_y, LAMBDA=0, step=1, print_loss_interval=1000, constant=10):
    # Ensure model_input requires gradients (avoid redundant calls)
    if not model_input.requires_grad:
        model_input.requires_grad_(True)
    
    # Compute gradients using torch.autograd.grad with vector outputs
    gradients = torch.autograd.grad(
        outputs=model_output,
        inputs=model_input,
        grad_outputs=torch.ones_like(model_output),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Extract x and y gradients more efficiently
    grad_x = gradients[:, 0].reshape(image_tensor.shape[0], image_tensor.shape[1])
    grad_y = gradients[:, 1].reshape(image_tensor.shape[0], image_tensor.shape[1])
    
    # Compute differences
    diff_x = grad_x - gt_grad_x
    diff_y = grad_y - gt_grad_y
    diff_main = model_output.flatten() - image_tensor.flatten()
    
    # Use fused operations
    loss_x = (diff_x**2).mean()*constant
    loss_y = (diff_y**2).mean()*constant
    loss_main = (diff_main**2).mean()
    total_loss = (1-LAMBDA) * loss_x + (1-LAMBDA) * loss_y + LAMBDA * loss_main
    
    if step % print_loss_interval == 0:
        print(f"Step {step}: loss_x = {loss_x.item()}, loss_y = {loss_y.item()}, loss_main = {loss_main.item()}, total_loss = {total_loss.item()}")
    
    return total_loss

