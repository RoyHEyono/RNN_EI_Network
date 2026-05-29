import torch
import torch.nn as nn
import torch.nn.functional as F

from inhibition import init
from inhibition.normalization import layer_norm_linear_ste as grad_norm


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
        use_bias: bool = True,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        if nonlinearity not in {"tanh", "relu", None}:
            raise ValueError("nonlinearity must be 'tanh', 'relu', or None")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity
        self.use_layer_norm = use_layer_norm
        self.use_bias = use_bias

        self.W_XE = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_EE = nn.Parameter(torch.empty(hidden_size, hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.bias.clamp = True
        else:
            self.register_parameter("bias", None)
        self.layer_norm = (
            nn.LayerNorm(hidden_size, eps=layer_norm_eps, elementwise_affine=False)
            if use_layer_norm
            else None
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
        """One-step linear drive before activation (for composition in :class:`SimpleEIRNN`)."""
        self._clamp_weights()
        pre_act = torch.matmul(x_t, self.W_XE.T) + torch.matmul(h_prev, self.W_EE.T)
        if self.bias is not None:
            pre_act = pre_act + self.bias
        if self.layer_norm is not None:
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
        for t in range(seq_len):
            h_t = self._activation(self.linear_drive(x[t], h_t))
            outputs.append(h_t)

        output = torch.stack(outputs, dim=0)  # (seq, batch, hidden)
        if self.batch_first:
            output = output.transpose(0, 1)  # (batch, seq, hidden)

        return output, h_t

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"nonlinearity={self.nonlinearity}, batch_first={self.batch_first}, "
            f"use_layer_norm={self.use_layer_norm}, use_bias={self.use_bias}"
        )


class SimpleEIRNN(nn.Module):
    """Recurrent E/I cell composed from three :class:`SimpleEERNN` linear drives.

    At each step:
    ``z = (exc - W_EI @ sub) / sqrt(U_EI @ div^2 + eps)``,
    then optional grad-norm STE when ``use_layer_norm`` is True, then ``tanh``/``relu``.

    API matches :class:`SimpleEERNN` (seq-first or batch-first 3D input, optional ``hx``).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nonlinearity: str = "tanh",
        batch_first: bool = False,
        use_layer_norm: bool = True,
        inh_ratio: float = 0.1,
        eps: float = 1e-5,
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
        self.eps = eps

        rnn_kw = dict(
            nonlinearity=None,
            batch_first=batch_first,
            use_layer_norm=False,
        )
        self.exc = SimpleEERNN(
            input_size, hidden_size, use_bias=True, **rnn_kw
        )
        self.sub = SimpleEERNN(
            input_size, hidden_size, use_bias=False, **rnn_kw
        )
        self.div = SimpleEERNN(
            input_size, hidden_size, use_bias=False, **rnn_kw
        )

        self.W_EI = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.U_EI = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_EI.clamp = True
        self.U_EI.clamp = True

        self.grad_norm = (
            grad_norm(hidden_size, eps=layer_norm_eps, elementwise_affine=False)
            if use_layer_norm
            else None
        )

        init.subtractive_excitatory_inhibitory_weight(self.sub.W_EE, self.exc.W_EE)
        init.subtractive_excitatory_inhibitory_weight(self.sub.W_XE, self.exc.W_XE)
        init.subtractive_inhibitory_excitatory_weight(self.exc.W_EE, self.W_EI)
        init.divisive_excitatory_inhibitory_weight(
            self.W_EI, self.exc.W_EE, self.sub.W_EE, self.div.W_EE
        )
        init.excitatory_weight(self.div.W_XE)
        init.divisive_inhibitory_excitatory_weight(self.exc.W_EE, self.U_EI)

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.nonlinearity == "relu":
            return torch.relu(x)
        return torch.tanh(x)

    def _cell(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if getattr(self.W_EI, "clamp", False):
                self.W_EI.clamp_(min=0)
            if getattr(self.U_EI, "clamp", False):
                self.U_EI.clamp_(min=0)

        e_drive = self.exc.linear_drive(x_t, h_t)
        h_i = self.sub.linear_drive(x_t, h_t)
        div_pre = self.div.linear_drive(x_t, h_t)

        sub_inh = F.linear(h_i, self.W_EI)
        div_inh = F.linear(div_pre ** 2, self.U_EI)

        z = (e_drive - sub_inh.detach()) / torch.sqrt(div_inh.detach() + self.eps)
        if self.grad_norm is not None:
            z = self.grad_norm(z)
        return self._activation(z)

    def forward(self, input: torch.Tensor, hx: torch.Tensor | None = None):
        if input.dim() != 3:
            raise ValueError("input must be a 3D tensor")

        if self.batch_first:
            batch_size, seq_len, input_size = input.shape
            x = input.transpose(0, 1)
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
        for t in range(seq_len):
            h_t = self._cell(x[t], h_t)
            outputs.append(h_t)

        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)

        return output, h_t

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"nonlinearity={self.nonlinearity}, batch_first={self.batch_first}, "
            f"use_layer_norm={self.use_layer_norm}"
        )
