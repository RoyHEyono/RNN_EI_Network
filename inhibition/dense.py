import numpy as np
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
    """
    Class modeling a subtractive feed-forward inhibition layer
    """
    def __init__(self, n_input, ne, ni=0.1, nonlinearity=None,use_bias=True, split_bias=False,
                 init_weights_kwargs={"numerator":2, "ex_distribution":"lognormal", "k":1}):
        """
        ne : number of exciatatory outputs
        ni : number (argument is an int) or proportion (float (0,1)) of inhibtitory units
        """
        super().__init__()
        self.n_input = n_input
        self.n_output = ne
        self.nonlinearity = nonlinearity
        self.split_bias = split_bias
        self.use_bias = use_bias
        self.ne = ne
        if isinstance(ni, float): self.ni = int(ne*ni)
        elif isinstance(ni, int): self.ni = ni
        self.ln_norm = torch.nn.LayerNorm(ne, elementwise_affine=False)
        self.gradient_alignment_val = 0
        self.output_alignment_val = 0
        self.relu = nn.ReLU()

        # to-from notation - W_post_pre and the shape is n_output x n_input
        self.Wex = nn.Parameter(torch.empty(self.ne,self.n_input))
        self.Wix = nn.Parameter(torch.empty(self.ni,self.n_input))
        self.Wei = nn.Parameter(torch.empty(self.ne,self.ni))
        
        # init and define bias as 0, split into pos, neg if using eg
        if self.use_bias:
            if self.split_bias: 
                self.bias_pos = nn.Parameter(torch.ones(self.n_output,1)) 
                self.bias_neg = nn.Parameter(torch.ones(self.n_output,1)*-1)
            else:
                self.bias = nn.Parameter(torch.zeros(self.n_output, 1))
        else:
            self.register_parameter('bias', None)
            self.split_bias = False
        
        try:
            self.init_weights(**init_weights_kwargs)
        except:
            pass
            #print("Warning: Error initialising weights with default init!")

    @property
    def W(self):
        return self.Wex - torch.matmul(self.Wei, self.Wix)

    @property
    def b(self):
        if self.split_bias: 
            return self.bias_pos + self.bias_neg
        else: 
            return self.bias
    
    def init_weights(self, numerator=2, ex_distribution="lognormal", k=1):
        """
        Initialises inhibitory weights to perform the centering operation of Layer Norm:
            Wex ~ lognormal or exponential dist
            Rows of Wix are copies of the mean row of Wex
            Rows of Wei sum to 1, squashed after being drawn from same dist as Wex.  
            k : the mean of the lognormal is k*std (as in the exponential dist)
        """
        def calc_ln_mu_sigma(mean, var):
            """
            Helper function: given a desired mean and var of a lognormal dist 
            (the func arguments) calculates and returns the underlying mu and sigma
            for the normal distribution that underlies the desired log normal dist.
            """
            mu_ln = np.log(mean**2 / np.sqrt(mean**2 + var))
            sigma_ln = np.sqrt(np.log(1 + (var /mean**2)))
            return mu_ln, sigma_ln

        target_std_wex = np.sqrt(numerator*self.ne/(self.n_input*(self.ne-1)))
        # He initialistion standard deviation derived from var(\hat{z}) = d * ne-1/ne * var(wex)E[x^2] 
        # where Wix is set to mean row of Wex and rows of Wei sum to 1.

        if ex_distribution =="exponential":
            exp_scale = target_std_wex # The scale parameter, \beta = 1/\lambda = std
            Wex_np = np.random.exponential(scale=exp_scale, size=(self.ne, self.n_input))
            Wei_np = np.random.exponential(scale=exp_scale, size=(self.ne, self.ni))
        
        elif ex_distribution =="lognormal":
            # here is where we decide how to skew the distribution
            mu, sigma = calc_ln_mu_sigma(target_std_wex*k,target_std_wex**2)
            Wex_np = np.random.lognormal(mu, sigma, size=(self.ne, self.n_input))
            Wei_np = np.random.lognormal(mu, sigma, size=(self.ne, self.ni))
        
        Wei_np /= Wei_np.sum(axis=1, keepdims=True)
        Wix_np = np.ones(shape=(self.ni,1))*Wex_np.mean(axis=0,keepdims=True)
        self.Wex.data = torch.from_numpy(Wex_np).float()
        self.Wix.data = torch.from_numpy(Wix_np).float()
        self.Wei.data = torch.from_numpy(Wei_np).float()

    def gradient_alignment(self, z):

        loss_ln_sum = self.relu(self.ln_norm(z)).sum()
        loss_z_sum = self.relu(z).sum()

        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'Wex' in name:
                    grad_true = torch.autograd.grad(loss_ln_sum, param, retain_graph=True)[0]
                    grad_homeo = torch.autograd.grad(loss_z_sum, param, retain_graph=True)[0]
                    cos_sim = F.cosine_similarity(grad_true.view(-1).unsqueeze(0), grad_homeo.view(-1).unsqueeze(0))

        return cos_sim

    def output_alignment(self, z):

        ln_output = self.relu(self.ln_norm(z))
        z_output = self.relu(z)
        cos_sim = F.cosine_similarity(ln_output.view(-1).unsqueeze(0), z_output.view(-1).unsqueeze(0))

        return cos_sim

    def forward(self, x):
        """
        x is batch_dim x input_dim, 
        therefore x.T as W is ne x input_dim ??? Why I got error?
        """
        self.z = torch.matmul(x, self.W.T)
        # if self.b: self.z = self.z + self.b.T
        if self.use_bias: self.z = self.z + self.b.T
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z

        if torch.is_grad_enabled():
            self.gradient_alignment_val = self.gradient_alignment(self.h).item()
            self.output_alignment_val = self.output_alignment(self.h).item()
        
        return self.h