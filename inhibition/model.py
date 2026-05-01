import torch
import torch.nn as nn
import torch.nn.functional as F

from inhibition.dense import INormLayer, EiDenseLayer
from inhibition.rnn import SimpleEERNN

MNIST_FLAT = 28 * 28
MNIST_SIDE = 28


def inorm_param_groups(model, lr_exc, lr_ie, lr_ei):
    """Excitatory and split inhibitory (*_IE vs *_EI) groups for INormLayer and EiDenseLayer."""
    exc_params, ie_params, ei_params = [], [], []
    for m in model.modules():
        if isinstance(m, INormLayer):
            exc_params.extend([m.W_EE, m.bias])
            ie_params.extend([m.W_IE, m.U_IE])
            ei_params.extend([m.W_EI, m.U_EI])
        elif isinstance(m, EiDenseLayer):
            exc_params.extend([m.W_EE, m.bias])
            ie_params.append(m.W_IE)
            ei_params.append(m.W_EI)
        elif isinstance(m, SimpleEERNN):
            exc_params.extend([m.W_XE, m.W_EE, m.bias])
    return [
        {"params": exc_params, "lr": lr_exc},
        {"params": ie_params, "lr": lr_ie},
        {"params": ei_params, "lr": lr_ei},
    ]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = INormLayer(MNIST_FLAT, 128)
        self.fc2 = INormLayer(128, 10)

    def forward(self, x, return_layer_inputs=False):
        h0 = torch.flatten(x, 1)
        z1 = self.fc1(h0)
        h1 = F.relu(z1)
        z2 = self.fc2(h1)
        output = z2  # logits for CrossEntropyLoss
        if return_layer_inputs:
            return output, (h0, h1)
        return output

    def inorm_layers(self):
        return [self.fc1, self.fc2]


class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.fc1 = INormLayer(MNIST_FLAT, 468)
        self.fc2 = INormLayer(468, 468)
        self.fc3 = INormLayer(468, 468)
        self.fc4 = EiDenseLayer(468, 10)

    def forward(self, x, return_layer_inputs=False):
        h0 = torch.flatten(x, 1)
        z1 = self.fc1(h0)
        h1 = F.relu(z1)
        z2 = self.fc2(h1)
        h2 = F.relu(z2)
        z3 = self.fc3(h2)
        h3 = F.relu(z3)
        z4 = self.fc4(h3)
        output = z4  # logits for CrossEntropyLoss
        if return_layer_inputs:
            return output, (h0, h1, h2, h3)
        return output

    def inorm_layers(self):
        return [self.fc1, self.fc2, self.fc3]


class RNNNet(nn.Module):
    """Simple classifier head over :class:`SimpleEERNN` for MNIST-like images.

    Expects the same dataloader tensor shape used elsewhere in this repo: ``(B, 1, 28, 28)``.
    Internally reshapes to a sequence ``(B, 28, 28)`` (rows as timesteps).
    """

    def __init__(self, hidden_size=128, nonlinearity="tanh", num_classes=10):
        super().__init__()
        self.rnn = SimpleEERNN(
            input_size=MNIST_SIDE,
            hidden_size=hidden_size,
            nonlinearity=nonlinearity,
            batch_first=True,
        )
        self.head = EiDenseLayer(hidden_size, num_classes)

    def forward(self, x, return_layer_inputs=False):
        if x.dim() != 4:
            raise ValueError(f"Expected input shape (B, 1, 28, 28), got {tuple(x.shape)}")
        if x.shape[1:] != (1, MNIST_SIDE, MNIST_SIDE):
            raise ValueError(
                f"Expected trailing dims (1, {MNIST_SIDE}, {MNIST_SIDE}), got {tuple(x.shape[1:])}"
            )

        seq = x.squeeze(1)  # (B, 28, 28)
        rnn_out, h_n = self.rnn(seq)
        logits = self.head(h_n)
        if return_layer_inputs:
            return logits, (seq, rnn_out, h_n)
        return logits

    def inorm_layers(self):
        # RNN path has no per-layer local loss terms in the current training objective.
        return []


class NeurogymRNNNet(nn.Module):
    """``SimpleEERNN`` + ``EiDenseLayer`` readout for NeuroGym-style vector observations.

    Same stack as :class:`RNNNet`, but each timestep is an ``ob_size`` vector (not an image row).

    Args:
        ob_size: observation dimension per timestep.
        hidden_size: recurrent hidden units.
        n_actions: number of discrete actions (logits per step).
        nonlinearity: passed to :class:`~inhibition.rnn.SimpleEERNN`.
    """

    def __init__(
        self,
        ob_size: int,
        hidden_size: int = 64,
        n_actions: int = 3,
        nonlinearity: str = "relu",
    ):
        super().__init__()
        self.ob_size = ob_size
        self.n_actions = n_actions
        self.rnn = SimpleEERNN(
            input_size=ob_size,
            hidden_size=hidden_size,
            nonlinearity=nonlinearity,
            batch_first=True,
        )
        self.head = EiDenseLayer(hidden_size, n_actions)

    def forward(self, x: torch.Tensor, return_layer_inputs: bool = False):
        if x.dim() != 3:
            raise ValueError(
                f"Expected input (batch, seq, ob_size), got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.ob_size:
            raise ValueError(
                f"Expected trailing dim ob_size={self.ob_size}, got {x.shape[-1]}"
            )
        rnn_out, _ = self.rnn(x)
        b, s, h = rnn_out.shape
        logits = self.head(rnn_out.reshape(b * s, h)).reshape(b, s, self.n_actions)
        if return_layer_inputs:
            return logits, ()
        return logits

    def inorm_layers(self):
        return []


class NeurogymVanillaRNNNet(nn.Module):
    """Control model: PyTorch :class:`torch.nn.RNN` + MLP readout for NeuroGym-style observations.

    Same I/O contract as :class:`NeurogymRNNNet` — ``(batch, seq, ob_size)`` → logits
    ``(batch, seq, n_actions)`` — but uses a standard RNN and a small feedforward head
    instead of :class:`~inhibition.rnn.SimpleEERNN` / :class:`~inhibition.dense.EiDenseLayer`.

    When ``use_layer_norm`` is True (default), applies :class:`~torch.nn.LayerNorm` on the
    hidden dimension of the RNN outputs (``elementwise_affine=False``, same style as
    :class:`~inhibition.rnn.SimpleEERNN`), before the MLP head. ``nn.RNN`` still applies
    its nonlinearity internally; this normalizes the emitted hidden states feeding the head.
    """

    def __init__(
        self,
        ob_size: int,
        hidden_size: int = 64,
        n_actions: int = 3,
        nonlinearity: str = "relu",
        num_layers: int = 1,
        ffn_hidden: int | None = None,
        use_layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        if nonlinearity not in {"tanh", "relu"}:
            raise ValueError("nonlinearity must be 'tanh' or 'relu' for nn.RNN")
        self.ob_size = ob_size
        self.n_actions = n_actions
        self.use_layer_norm = use_layer_norm
        self.rnn = nn.RNN(
            input_size=ob_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
        )
        self.layer_norm = (
            nn.LayerNorm(hidden_size, eps=layer_norm_eps, elementwise_affine=False)
            if use_layer_norm
            else None
        )
        ffn_dim = ffn_hidden if ffn_hidden is not None else hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor, return_layer_inputs: bool = False):
        if x.dim() != 3:
            raise ValueError(
                f"Expected input (batch, seq, ob_size), got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.ob_size:
            raise ValueError(
                f"Expected trailing dim ob_size={self.ob_size}, got {x.shape[-1]}"
            )
        rnn_out, _ = self.rnn(x)
        if self.layer_norm is not None:
            rnn_out = self.layer_norm(rnn_out)
        b, s, h = rnn_out.shape
        logits = self.head(rnn_out.reshape(b * s, h)).reshape(b, s, self.n_actions)
        if return_layer_inputs:
            return logits, ()
        return logits

    def inorm_layers(self):
        return []
