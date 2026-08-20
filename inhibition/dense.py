import torch
import torch.nn as nn
import torch.nn.functional as F

from inhibition.normalization import LayerNormFeedbackAblation
from inhibition import init


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two tensors flattened to a single vector."""
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).squeeze()


class INormLayer(nn.Module):
    """E/I layer whose inhibition learns to layer-normalize the excitatory drive.

    Implements Eq. 1 of the paper::

        z = (W_EE h - W_EI W_IE h) / sqrt(U_EI (U_IE h)^2)

    with both inhibitory terms detached from the task loss, so ``W_EE``/``bias``
    are trained by the task and the inhibitory weights only by
    :meth:`local_loss`. ``ln_feedback``/``gradient_norm`` select which part (if
    any) of the LayerNorm Jacobian is imposed on the back-propagated error
    (the paper's GradNorm); ``shunting=False`` drops the divisive pathway,
    giving the subtractive-only I-Norm condition.
    """

    def __init__(
        self,
        in_features,
        out_features,
        inh_ratio=0.1,
        eps=1e-5,
        ln_feedback="full",
        gradient_norm=True,
        shunting=True,
        track_alignment=False,
        freeze_ei=False,
    ):
        super().__init__()
        self.eps = eps
        self.ln_feedback = ln_feedback
        self.shunting = shunting
        self.track_alignment = track_alignment
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

        # Fixed, random, positive synapses summing to 1: the lateral-inhibition
        # pool of Theorem 1. Used only by the ``fa_center`` feedback rule.
        nu = torch.rand(out_features)
        self.register_buffer("fa_weights", nu / nu.sum())

        self.grad_norm = LayerNormFeedbackAblation(
            ln_feedback=ln_feedback, no_backward=not gradient_norm
        )
        # Reference backward for ``fa_center`` alignment: exact mean centering.
        self.mean_norm = LayerNormFeedbackAblation(
            ln_feedback="center", no_backward=not gradient_norm
        )
        self.ln_norm = torch.nn.LayerNorm(out_features, elementwise_affine=False)
        self.local_criterion = nn.MSELoss()

        self.gradient_alignment_val = float("nan")
        self.output_alignment_val = float("nan")

        init.excitatory_weight(self.W_EE)
        init.subtractive_excitatory_inhibitory_weight(self.W_IE, self.W_EE)
        init.subtractive_inhibitory_excitatory_weight(self.W_EE, self.W_EI)
        init.divisive_excitatory_inhibitory_weight(self.W_EI, self.W_EE, self.W_IE, self.U_IE)
        init.divisive_inhibitory_excitatory_weight(self.W_EE, self.U_EI)

        if freeze_ei:
            # The *_EI matrices are the fixed averaging step of the theory
            # (row-normalized W_EI, uniform U_EI); the reference implementation
            # keeps them frozen and trains only W_EE, W_IE and U_IE.
            self.W_EI.requires_grad_(False)
            self.U_EI.requires_grad_(False)

    def _clamp_weights(self):
        """Enforce Dale's principle (``U_IE`` is exempt: SVD init makes it signed)."""
        with torch.no_grad():
            for p in self.parameters():
                if getattr(p, "clamp", False):
                    p.clamp_(min=0)

    def _divisor(self, h_prev, detach=True):
        if not self.shunting:
            return torch.ones((), device=h_prev.device, dtype=h_prev.dtype)
        h = h_prev.detach() if detach else h_prev
        h_D = F.linear(h, self.U_IE) ** 2
        div_inh = F.linear(h_D, self.U_EI)
        z_d = torch.sqrt(div_inh + self.eps)
        return z_d.detach() if detach else z_d

    def output_alignment(self, z):
        """Cosine similarity of this layer's output with true LayerNorm's."""
        ln_out = F.relu(self.ln_norm(z))
        z_out = F.relu(self.grad_norm(z, weights=self.fa_weights))
        return _cosine(ln_out, z_out)

    def gradient_alignment(self, z):
        """Cosine similarity of dL/dW_EE under the reference vs applied backward.

        The reference is true LayerNorm, except for ``fa_center`` where it is
        exact mean centering -- the operation the random inhibitory pool is
        meant to approximate.
        """
        if self.ln_feedback == "fa_center":
            ref = self.mean_norm(z, weights=self.fa_weights)
        else:
            ref = self.ln_norm(z)
        applied = self.grad_norm(z, weights=self.fa_weights)
        g_ref = torch.autograd.grad(F.relu(ref).sum(), self.W_EE, retain_graph=True)[0]
        g_app = torch.autograd.grad(F.relu(applied).sum(), self.W_EE, retain_graph=True)[0]
        return _cosine(g_ref, g_app)

    def forward(self, h_prev):
        # Enforce Dale's Principle: keep weights non-negative
        self._clamp_weights()

        # 1. Calculate Inhibitory Activity (Feedforward)
        h_I = F.linear(h_prev, self.W_IE) # Subtractive population

        # 2. Direct Excitatory Drive
        e_drive = F.linear(h_prev, self.W_EE) + self.bias

        # 3. Subtractive Inhibition
        sub_inh = F.linear(h_I, self.W_EI)

        # 4. Divisive Inhibition
        z_d = self._divisor(h_prev, detach=True)

        # 5. Combined Normalization (Equation 1 in paper)
        z = (e_drive - sub_inh.detach()) / z_d

        if self.track_alignment and torch.is_grad_enabled():
            self.gradient_alignment_val = self.gradient_alignment(z).item()
            self.output_alignment_val = self.output_alignment(z).item()

        # Straight through estimator for gradient (V. important)
        z = self.grad_norm(z, weights=self.fa_weights)

        return z

    def local_loss(self, h_prev):
        """I-Norm loss: drive the normalized drive to mean 0, variance 1.

        Returns ``(moments_term, ln_mse)``; the second value is a diagnostic MSE
        against true LayerNorm and is not optimized.
        """

        h = h_prev.detach()
        h_I = F.linear(h, self.W_IE)
        e_drive = F.linear(h, self.W_EE) + self.bias
        sub_inh = F.linear(h_I, self.W_EI)
        z_d = self._divisor(h, detach=False)
        z = (e_drive.detach() - sub_inh) / z_d

        mean = torch.mean(z, dim=1, keepdim=True)
        var = z.var(dim=-1, unbiased=False)
        ln_ground_truth_loss = self.local_criterion(z, self.ln_norm(e_drive))

        var_term = (var-1) ** 2
        mean_term = mean ** 2

        return ((mean_term + var_term).mean()), (ln_ground_truth_loss).item()


class EiDenseLayer(nn.Module):
    """Standard EI layer: ``z = W_EE h - W_EI W_IE h + b``.

    Unlike :class:`INormLayer` there is no divisive pathway and no gradient stop
    on inhibition -- the task loss trains ``W_IE``, ``W_EI`` and ``W_EE`` alike.
    Normalization, when used, is applied by the enclosing network.
    """

    def __init__(self, in_features, out_features, inh_ratio=0.1, eps=1e-5,
                 track_alignment=False):
        super().__init__()
        self.eps = eps
        self.track_alignment = track_alignment
        n_inh = int(out_features * inh_ratio)

        self.W_EE = nn.Parameter(torch.randn(out_features, in_features))
        self.W_IE = nn.Parameter(torch.randn(n_inh, in_features))
        self.W_EI = nn.Parameter(torch.randn(out_features, n_inh))

        self.bias = nn.Parameter(torch.zeros(1, out_features))
        self.bias.clamp = True

        self.ln_norm = torch.nn.LayerNorm(out_features, elementwise_affine=False)
        self.gradient_alignment_val = float("nan")
        self.output_alignment_val = float("nan")

        init.excitatory_weight(self.W_EE)
        init.subtractive_excitatory_inhibitory_weight(self.W_IE, self.W_EE)
        init.subtractive_inhibitory_excitatory_weight(self.W_EE, self.W_EI)

    def _clamp_weights(self):
        with torch.no_grad():
            for p in self.parameters():
                if getattr(p, "clamp", False):
                    p.clamp_(min=0)

    def output_alignment(self, z):
        return _cosine(F.relu(self.ln_norm(z)), F.relu(z))

    def gradient_alignment(self, z):
        g_ref = torch.autograd.grad(
            F.relu(self.ln_norm(z)).sum(), self.W_EE, retain_graph=True
        )[0]
        g_app = torch.autograd.grad(F.relu(z).sum(), self.W_EE, retain_graph=True)[0]
        return _cosine(g_ref, g_app)

    def forward(self, h_prev):
        self._clamp_weights()

        h_I = F.linear(h_prev, self.W_IE)
        e_drive = F.linear(h_prev, self.W_EE) + self.bias
        sub_inh = F.linear(h_I, self.W_EI)

        z = e_drive - sub_inh

        if self.track_alignment and torch.is_grad_enabled():
            self.gradient_alignment_val = self.gradient_alignment(z).item()
            self.output_alignment_val = self.output_alignment(z).item()

        return z


class EDenseLayer(nn.Module):
    """Pure excitatory dense layer without inhibitory pathways."""

    def __init__(self, in_features, out_features, eps=1e-5, track_alignment=False):
        super().__init__()
        self.eps = eps
        self.track_alignment = track_alignment
        self.W_EE = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(1, out_features))
        self.bias.clamp = True

        self.ln_norm = torch.nn.LayerNorm(out_features, elementwise_affine=False)
        self.gradient_alignment_val = float("nan")
        self.output_alignment_val = float("nan")

        init.excitatory_weight(self.W_EE)

    def _clamp_weights(self):
        with torch.no_grad():
            for p in self.parameters():
                if getattr(p, "clamp", False):
                    p.clamp_(min=0)

    def output_alignment(self, z):
        return _cosine(F.relu(self.ln_norm(z)), F.relu(z))

    def gradient_alignment(self, z):
        g_ref = torch.autograd.grad(
            F.relu(self.ln_norm(z)).sum(), self.W_EE, retain_graph=True
        )[0]
        g_app = torch.autograd.grad(F.relu(z).sum(), self.W_EE, retain_graph=True)[0]
        return _cosine(g_ref, g_app)

    def forward(self, h_prev):
        self._clamp_weights()

        z = F.linear(h_prev, self.W_EE) + self.bias

        if self.track_alignment and torch.is_grad_enabled():
            self.gradient_alignment_val = self.gradient_alignment(z).item()
            self.output_alignment_val = self.output_alignment(z).item()

        return z
