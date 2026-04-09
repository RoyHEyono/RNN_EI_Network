import torch
import torch.nn as nn

from inhibition import init


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
        nonlinearity: str = "tanh",
        batch_first: bool = False,
        use_layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        if nonlinearity not in {"tanh", "relu"}:
            raise ValueError("nonlinearity must be either 'tanh' or 'relu'")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity
        self.use_layer_norm = use_layer_norm

        self.W_XE = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_EE = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.bias.clamp = True
        self.layer_norm = (
            nn.LayerNorm(hidden_size, eps=layer_norm_eps, elementwise_affine=False)
            if use_layer_norm
            else None
        )

        init.excitatory_weight(self.W_XE)
        init.excitatory_weight(self.W_EE)

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.nonlinearity == "relu":
            return torch.relu(x)
        return torch.tanh(x)

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

        # Keep excitatory weight matrices non-negative, matching inhibition modules.
        with torch.no_grad():
            if getattr(self.W_XE, "clamp", False):
                self.W_XE.clamp_(min=0)
            if getattr(self.W_EE, "clamp", False):
                self.W_EE.clamp_(min=0)
            if getattr(self.bias, "clamp", False):
                self.bias.clamp_(min=0)

        outputs = []
        for t in range(seq_len):
            x_drive = torch.matmul(x[t], self.W_XE.T)
            h_drive = torch.matmul(h_t, self.W_EE.T)
            pre_act = x_drive + h_drive + self.bias
            if self.layer_norm is not None:
                pre_act = self.layer_norm(pre_act)
            h_t = self._activation(pre_act)
            outputs.append(h_t)

        output = torch.stack(outputs, dim=0)  # (seq, batch, hidden)
        if self.batch_first:
            output = output.transpose(0, 1)  # (batch, seq, hidden)

        return output, h_t

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"nonlinearity={self.nonlinearity}, batch_first={self.batch_first}, "
            f"use_layer_norm={self.use_layer_norm}"
        )
