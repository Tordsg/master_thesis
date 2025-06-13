from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tinycudann as tcnn
except OSError:
    from typing import TYPE_CHECKING
  
  
def initialize(weights, init_type='default', omega_0=1, min=None, max=None, mean=None, std=None):
    with torch.no_grad():
        if init_type == 'scaled':
                nn.init.uniform_(weights, -1, 1)
        elif init_type == 'default':
            pass
        else:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weights)
            if init_type == 'siren':
                min = -math.sqrt(6 / fan_in) / omega_0
                max = math.sqrt(6 / fan_in) / omega_0
                nn.init.uniform_(weights, min, max)
            elif init_type == 'uniform':
                nn.init.uniform_(weights, min, max)
            elif init_type == 'normal':
                nn.init.normal_(weights, mean=mean, std=std)
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(weights, nonlinearity='relu')
            elif init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(weights, nonlinearity='relu')
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(weights)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(weights)

class HashRelu(nn.Module):
    def __init__(self, in_features = 2, out_features = 1, hidden_layers = 5, hidden_features = 48, n_levels = 16,          
                        n_features_per_level = 2,
                        log2_hashmap_size = 20,                        
                        base_resolution = 16,                          
                        per_level_scale = 2.0,
                        interpolation = "Linear",
                        outermost_linear=True, encoding_init_type='default', init_type='kaiming_normal'):
        super().__init__()
        self.encoding = tcnn.Encoding(
            n_input_dims=in_features,
            encoding_config={
                        "otype": "Grid",           
                        "type": "Hash",            
                        "n_levels": n_levels,        
                        "n_features_per_level": n_features_per_level,
                        "log2_hashmap_size": log2_hashmap_size,                        
                        "base_resolution": base_resolution,                          
                        "per_level_scale": per_level_scale,
                        "interpolation": interpolation,
                    }
        )
        initialize(next(self.encoding.parameters()), init_type=encoding_init_type)
           
        self.mlp = []
        
        layer = ReluLayer(n_levels * n_features_per_level, hidden_features)
        initialize(layer.linear.weight, init_type=init_type)
        self.mlp.append(layer)
        
        for _ in range(hidden_layers - 1):
            layer = ReluLayer(hidden_features, hidden_features)
            initialize(layer.linear.weight, init_type=init_type)
            self.mlp.append(layer)
            
        last_layer = nn.Linear(hidden_features, out_features)
        initialize(last_layer.weight, init_type=init_type)
        
        self.mlp.append(last_layer)
        self.mlp = nn.Sequential(*self.mlp)
        
    def forward(self, input):
        temp = self.encoding(input).float()
        output = self.mlp(temp)
        return output


class HashSIREN(nn.Module):
    def __init__(self, in_features = 2, out_features = 1, hidden_layers = 3, hidden_features = 16, n_levels = 12,          
                        n_features_per_level = 2,
                        log2_hashmap_size = 20,                        
                        base_resolution = 16,                          
                        per_level_scale = 2.0,
                        interpolation = "Linear",
                        first_omega_0 = 300,
                        hidden_omega_0 = 1,
                        first_init_omega_0 = 30,
                        outermost_linear=True, init_type='siren', encoding_init_type='default', first_init_type='normal', first_mean=0, first_std=0.1, first_bound=0.1):
        super().__init__()
        self.encoding = tcnn.Encoding(
            n_input_dims=in_features,
            encoding_config={
                        "otype": "Grid",           
                        "type": "Hash",            
                        "n_levels": n_levels,        
                        "n_features_per_level": n_features_per_level,
                        "log2_hashmap_size": log2_hashmap_size,                        
                        "base_resolution": base_resolution,                          
                        "per_level_scale": per_level_scale,
                        "interpolation": interpolation,
                    },
            dtype=torch.float32
        )
        initialize(next(self.encoding.parameters()), init_type=encoding_init_type)
           
        self.mlp = []
        first_layer = SineLayer(n_levels * n_features_per_level, hidden_features, is_first=True, omega_0=first_omega_0)
        initialize(first_layer.linear.weight, init_type=first_init_type, omega_0=first_init_omega_0, mean=first_mean, std=first_std, min=-first_bound, max=first_bound)
        self.mlp.append(first_layer)

        for _ in range(hidden_layers-1):
            layer = SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0)
            initialize(layer.linear.weight, init_type=init_type, omega_0=hidden_omega_0)
            self.mlp.append(layer)


        last_layer = nn.Linear(hidden_features, out_features)
        initialize(last_layer.weight, init_type=init_type)
        self.mlp.append(last_layer)

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, input):
        encoded = self.encoding(input).float()
        output = self.mlp(encoded)
        return output

class Siren(nn.Module):
    def __init__(self, in_features=2, out_features=1, hidden_features=32, hidden_layers=4, outermost_linear = True, 
                 first_omega_0=30, hidden_omega_0=1, init_type='siren', init_omega_0=30):
        super().__init__()
        
        self.mlp = []
        first_layer = SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0)
        initialize(first_layer.linear.weight, init_type=init_type, omega_0=init_omega_0)
        self.mlp.append(first_layer)

        for i in range(hidden_layers):
            layer = SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0)
            initialize(layer.linear.weight, init_type=init_type, omega_0=hidden_omega_0)
            self.mlp.append(layer)

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            initialize(final_linear.weight, init_type=init_type) 
            self.mlp.append(final_linear)
        else:
            last_layer = SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0)
            initialize(last_layer.linear.weight, init_type=init_type, omega_0=hidden_omega_0)
            self.mlp.append(last_layer)
        
        self.mlp = nn.Sequential(*self.mlp)
    
    def forward(self, coords):
        output = self.mlp(coords)
        return output     

class SineLayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=1):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class ReluLayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
 
    def forward(self, input):
        return F.relu(self.linear(input))

    