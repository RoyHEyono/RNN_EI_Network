import torch
import torch.nn as nn
import torch.nn.functional as F

class layer_norm_linear_ste(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 1. The 'Linear' path
        linear_out = x
        
        # 2. The 'LayerNorm' path
        ln_out = F.layer_norm(x)
        
        # 3. The Hijack:
        # Forward is linear_out, Backward is ln_out's gradient
        return ln_out + (linear_out - ln_out).detach()