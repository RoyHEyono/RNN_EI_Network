import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class INormLayer(nn.Module):
    def __init__(self, in_features, out_features, inh_ratio=0.1, eps=1e-5):
        super().__init__()
        self.eps = eps
        # The inhibitory population size is typically 10% of the excitatory size
        n_inh = int(out_features * inh_ratio)
        
        # Excitatory to Excitatory weights
        self.W_EE = nn.Parameter(torch.randn(out_features, in_features))
        
        # Subtractive Inhibitory Pathway
        self.W_IE = nn.Parameter(torch.randn(n_inh, in_features))  # E to I (sub)
        self.W_EI = nn.Parameter(torch.randn(out_features, n_inh)) # I (sub) to E
        
        # Divisive Inhibitory Pathway
        self.U_IE = nn.Parameter(torch.randn(n_inh, in_features))  # E to I (div)
        self.U_EI = nn.Parameter(torch.randn(out_features, n_inh)) # I (div) to E

        self.bias = nn.Parameter(torch.zeros(1, out_features))

        # Initialize weights to be positive
        for p in self.parameters():
            nn.init.kaiming_uniform_(p, a=1)
            p.data.abs_()

    def forward(self, h_prev):
        # Enforce Dale's Principle: keep weights non-negative
        with torch.no_grad():
            for p in self.parameters():
                p.clamp_(min=0)

        # 1. Calculate Inhibitory Activity (Feedforward)
        h_I = F.linear(h_prev, self.W_IE) # Subtractive population
        h_D = F.linear(h_prev, self.U_IE)**2 # Divisive population

        # 2. Direct Excitatory Drive
        e_drive = F.linear(h_prev, self.W_EE) + self.bias

        # 3. Subtractive Inhibition
        sub_inh = F.linear(h_I, self.W_EI)
        
        # 4. Divisive Inhibition
        div_inh = F.linear(h_D, self.U_EI)

        # 5. Combined Normalization (Equation 1 in paper)
        z = (e_drive - sub_inh) / torch.sqrt(div_inh + self.eps)
        
        return z