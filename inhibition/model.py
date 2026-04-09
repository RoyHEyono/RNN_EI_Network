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
