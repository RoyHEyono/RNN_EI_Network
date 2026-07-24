import torch
import torch.nn as nn

from inhibition import init
from inhibition.normalization import ParametrizedLayerNorm


class SimpleEERNN(nn.Module):
    """Minimal RNN with excitatory input/recurrent weights and bias.

    API intentionally mirrors a subset of ``torch.nn.RNN``:
    - inputs are 3D tensors with shape ``(seq, batch, feat)`` by default
    - ``batch_first=True`` accepts ``(batch, seq, feat)``
    - optional ``hx`` initial state with shape ``(batch, hidden_size)``

    The update is:
    ``h_t = nonlinearity(x_t @ W_XE^T + h_{t-1} @ W_EE^T + bias)``
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nonlinearity: str | None = "tanh",
        batch_first: bool = False,
        use_layer_norm: bool = True,
        use_parametrized_layer_norm: bool = False,
        use_bias: bool = True,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        if nonlinearity not in {"tanh", "relu", None}:
            raise ValueError("nonlinearity must be 'tanh', 'relu', or None")
        if use_parametrized_layer_norm and not use_layer_norm:
            raise ValueError(
                "use_parametrized_layer_norm requires use_layer_norm=True"
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity
        self.use_layer_norm = use_layer_norm
        self.use_parametrized_layer_norm = use_parametrized_layer_norm
        self.use_bias = use_bias

        self.W_XE = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_EE = nn.Parameter(torch.empty(hidden_size, hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.bias.clamp = True
        else:
            self.register_parameter("bias", None)
        if not use_layer_norm:
            self.layer_norm = None
        elif use_parametrized_layer_norm:
            self.layer_norm = ParametrizedLayerNorm(
                input_size, hidden_size, eps=layer_norm_eps
            )
        else:
            self.layer_norm = nn.LayerNorm(
                hidden_size, eps=layer_norm_eps, elementwise_affine=False
            )

        init.excitatory_weight(self.W_XE)
        init.excitatory_weight(self.W_EE)

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.nonlinearity is None:
            return x
        if self.nonlinearity == "relu":
            return torch.relu(x)
        return torch.tanh(x)

    def _clamp_weights(self) -> None:
        with torch.no_grad():
            if getattr(self.W_XE, "clamp", False):
                self.W_XE.clamp_(min=0)
            if getattr(self.W_EE, "clamp", False):
                self.W_EE.clamp_(min=0)
            if self.bias is not None and getattr(self.bias, "clamp", False):
                self.bias.clamp_(min=0)

    def linear_drive(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """One-step linear drive before activation."""
        self._clamp_weights()
        pre_act = torch.matmul(x_t, self.W_XE.T) + torch.matmul(h_prev, self.W_EE.T)
        if self.bias is not None:
            pre_act = pre_act + self.bias
        if self.layer_norm is not None:
            if self.use_parametrized_layer_norm:
                pre_act, aux_loss = self.layer_norm(pre_act, x_t, h_prev)
                return pre_act, aux_loss
            else:
                pre_act = self.layer_norm(pre_act)
        return pre_act

    def forward(self, input: torch.Tensor, hx: torch.Tensor | None = None):
        if input.dim() != 3:
            raise ValueError("input must be a 3D tensor")

        if self.batch_first:
            batch_size, seq_len, input_size = input.shape
            x = input.transpose(0, 1)  # (seq, batch, feat)
        else:
            seq_len, batch_size, input_size = input.shape
            x = input

        if input_size != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, got input_size={input_size}"
            )

        if hx is None:
            h_t = x.new_zeros(batch_size, self.hidden_size)
        else:
            if hx.shape != (batch_size, self.hidden_size):
                raise ValueError(
                    f"hx must have shape {(batch_size, self.hidden_size)}, got {tuple(hx.shape)}"
                )
            h_t = hx

        outputs = []
        aux_losses = []
        for t in range(seq_len):
            if self.use_parametrized_layer_norm:
                h_t, aux_loss = self.linear_drive(x[t], h_t)
                h_t = self._activation(h_t)
                aux_losses.append(aux_loss)
            else:
                h_t = self._activation(self.linear_drive(x[t], h_t))
            outputs.append(h_t)

        output = torch.stack(outputs, dim=0)  # (seq, batch, hidden)
        if self.batch_first:
            output = output.transpose(0, 1)  # (batch, seq, hidden)

        if self.use_parametrized_layer_norm:
            return outputs, h_t, aux_losses

        return output, h_t

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"nonlinearity={self.nonlinearity}, batch_first={self.batch_first}, "
            f"use_layer_norm={self.use_layer_norm}, "
            f"use_parametrized_layer_norm={self.use_parametrized_layer_norm}, "
            f"use_bias={self.use_bias}"
        )
