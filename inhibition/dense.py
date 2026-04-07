import torch
import torch.nn as nn
import torch.nn.functional as F
from inhibition.normalization import layer_norm_linear_ste as grad_norm
from inhibition import init

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
        self.U_IE = nn.Parameter(torch.randn(out_features, in_features))  # E to I (div)
        self.U_EI = nn.Parameter(torch.randn(out_features, out_features)) # I (div) to E

        self.bias = nn.Parameter(torch.zeros(1, out_features))
        self.bias.clamp = True

        self.grad_norm = grad_norm(out_features, elementwise_affine=False)
        self.ln_norm = torch.nn.LayerNorm(out_features, elementwise_affine=False)
        self.local_criterion = nn.MSELoss()

        init.excitatory_weight(self.W_EE)
        init.subtractive_excitatory_inhibitory_weight(self.W_IE, self.W_EE)
        init.subtractive_inhibitory_excitatory_weight(self.W_EE, self.W_EI)
        init.divisive_excitatory_inhibitory_weight(self.W_EI, self.W_EE, self.W_IE, self.U_IE)
        init.divisive_inhibitory_excitatory_weight(self.W_EE, self.U_EI)

    def forward(self, h_prev):
        # Enforce Dale's Principle: keep weights non-negative
        with torch.no_grad():
            for p in self.parameters():
                if p.clamp:
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
        z = (e_drive - sub_inh.detach()) / torch.sqrt(div_inh.detach() + self.eps)
        z = self.grad_norm(z) # Straight through estimator for gradient (V. important)

        return z

    def local_loss(self, h_prev):

        h = h_prev.detach()
        h_I = F.linear(h, self.W_IE)
        h_D = F.linear(h, self.U_IE) ** 2
        e_drive = F.linear(h, self.W_EE) + self.bias
        sub_inh = F.linear(h_I, self.W_EI)
        div_inh = F.linear(h_D, self.U_EI)
        z = (e_drive.detach() - sub_inh) / torch.sqrt(div_inh + self.eps)

        mean = torch.mean(z, dim=1, keepdim=True)
        var = z.var(dim=-1, unbiased=False)
        ln_ground_truth_loss = self.local_criterion(z, self.ln_norm(e_drive))
        
        var_term = (var-1) ** 2
        mean_term = mean ** 2
        
        return ((mean_term + var_term).mean()), (ln_ground_truth_loss).item()


class EiDenseLayer(nn.Module):
    """Subtractive E–I layer aligned with :class:`INormLayer`, without divisive ``h_D``.

    Uses only the inhibitory population activity ``h_I`` (``W_IE`` → ``W_EI``). There is
    no ``local_loss`` and no gradient stop on inhibition: task loss backprops through
    ``W_IE``, ``W_EI``, and ``W_EE`` like ``W_EE``. Output is ``e_drive - sub_inh`` then
    ``grad_norm`` (same STE LayerNorm wrapper as ``INormLayer``) in place of divisive
    normalization.
    """

    def __init__(self, in_features, out_features, inh_ratio=0.1, eps=1e-5):
        super().__init__()
        self.eps = eps
        n_inh = int(out_features * inh_ratio)

        self.W_EE = nn.Parameter(torch.randn(out_features, in_features))
        self.W_IE = nn.Parameter(torch.randn(n_inh, in_features))
        self.W_EI = nn.Parameter(torch.randn(out_features, n_inh))

        self.bias = nn.Parameter(torch.zeros(1, out_features))
        self.bias.clamp = True

        init.excitatory_weight(self.W_EE)
        init.subtractive_excitatory_inhibitory_weight(self.W_IE, self.W_EE)
        init.subtractive_inhibitory_excitatory_weight(self.W_EE, self.W_EI)

    def forward(self, h_prev):
        with torch.no_grad():
            for p in self.parameters():
                if p.clamp:
                    p.clamp_(min=0)

        h_I = F.linear(h_prev, self.W_IE)
        e_drive = F.linear(h_prev, self.W_EE) + self.bias
        sub_inh = F.linear(h_I, self.W_EI)

        z = e_drive - sub_inh

        return z