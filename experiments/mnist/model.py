import torch
import torch.nn as nn
import torch.nn.functional as F

from inhibition.dense import INormLayer

MNIST_FLAT = 28 * 28


def inorm_adadelta_param_groups(model, lr_exc, lr_ie, lr_ei):
    """Excitatory and split inhibitory (*_IE vs *_EI) groups for INormLayer."""
    exc_params, ie_params, ei_params = [], [], []
    for m in model.modules():
        if isinstance(m, INormLayer):
            exc_params.extend([m.W_EE, m.bias])
            ie_params.extend([m.W_IE, m.U_IE])
            ei_params.extend([m.W_EI, m.U_EI])
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
        output = F.log_softmax(z2, dim=1)
        if return_layer_inputs:
            return output, (h0, h1)
        return output

    def inorm_layers(self):
        return [self.fc1, self.fc2]
