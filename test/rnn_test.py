import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from inhibition.rnn import SimpleEERNN, SimpleEIRNN


def _copy_exc_weights(ei: SimpleEIRNN, ee: SimpleEERNN) -> None:
    """Share excitatory parameters between EI.exc and EE+LN."""
    ee.W_XE.data.copy_(ei.exc.W_XE.data)
    ee.W_EE.data.copy_(ei.exc.W_EE.data)
    ee.bias.data.copy_(ei.exc.bias.data)


def _raw_exc_drive(exc: SimpleEERNN, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
    """Excitatory linear drive without LN."""
    return torch.matmul(x_t, exc.W_XE.T) + torch.matmul(h_prev, exc.W_EE.T) + exc.bias


def _ei_pre_activation(ei: SimpleEIRNN, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
    """E/I pre-activation z before grad_norm STE and activation."""
    e_drive = _raw_exc_drive(ei.exc, x_t, h_prev)
    h_i = ei.sub.linear_drive(x_t, h_prev)
    div_pre = ei.div.linear_drive(x_t, h_prev)
    sub_inh = F.linear(h_i, ei.W_EI)
    div_inh = F.linear(div_pre ** 2, ei.U_EI)
    return (e_drive - sub_inh.detach()) / torch.sqrt(div_inh.detach() + ei.eps)


class TestSimpleEERNN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_forward_shapes_seq_first(self):
        model = SimpleEERNN(input_size=8, hidden_size=10, batch_first=False)
        x = torch.randn(5, 3, 8)  # (seq, batch, feat)
        output, h_n = model(x)
        self.assertEqual(output.shape, (5, 3, 10))
        self.assertEqual(h_n.shape, (3, 10))

    def test_forward_shapes_batch_first(self):
        model = SimpleEERNN(input_size=6, hidden_size=4, batch_first=True)
        x = torch.randn(4, 7, 6)  # (batch, seq, feat)
        output, h_n = model(x)
        self.assertEqual(output.shape, (4, 7, 4))
        self.assertEqual(h_n.shape, (4, 4))

    def test_has_input_excitatory_matrix_and_bias(self):
        model = SimpleEERNN(input_size=4, hidden_size=8)
        self.assertEqual(model.W_XE.shape, (8, 4))
        self.assertEqual(model.bias.shape, (8,))

    def test_layer_norm_config(self):
        model_default = SimpleEERNN(input_size=4, hidden_size=8)
        self.assertIsNotNone(model_default.layer_norm)

        model_no_ln = SimpleEERNN(input_size=4, hidden_size=8, use_layer_norm=False)
        self.assertIsNone(model_no_ln.layer_norm)

    def test_clamps_recurrent_weights_to_non_negative(self):
        model = SimpleEERNN(input_size=5, hidden_size=5)
        with torch.no_grad():
            model.W_XE.fill_(-1.0)
            model.W_EE.fill_(-1.0)
            model.bias.fill_(-1.0)
        x = torch.randn(3, 2, 5)
        _ = model(x)
        self.assertTrue(torch.all(model.W_XE >= 0))
        self.assertTrue(torch.all(model.W_EE >= 0))
        self.assertTrue(torch.all(model.bias >= 0))

    def test_accepts_initial_hidden_state(self):
        model = SimpleEERNN(input_size=3, hidden_size=3)
        x = torch.randn(2, 4, 3)
        hx = torch.randn(4, 3)
        output, h_n = model(x, hx=hx)
        self.assertEqual(output.shape, (2, 4, 3))
        self.assertEqual(h_n.shape, (4, 3))


class TestSimpleEIRNN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

    def test_forward_shapes_seq_first(self):
        model = SimpleEIRNN(input_size=8, hidden_size=10, batch_first=False)
        x = torch.randn(5, 3, 8)
        output, h_n = model(x)
        self.assertEqual(output.shape, (5, 3, 10))
        self.assertEqual(h_n.shape, (3, 10))

    def test_forward_shapes_batch_first(self):
        model = SimpleEIRNN(input_size=6, hidden_size=8, batch_first=True)
        x = torch.randn(4, 7, 6)
        output, h_n = model(x)
        self.assertEqual(output.shape, (4, 7, 8))
        self.assertEqual(h_n.shape, (4, 8))

    def test_forward_matches_excitatory_ln_at_init(self):
        """At init, E/I dynamics should match a plain excitatory RNN with LayerNorm."""
        ei = SimpleEIRNN(input_size=6, hidden_size=8, batch_first=True, use_layer_norm=True)
        ee = SimpleEERNN(input_size=6, hidden_size=8, batch_first=True, use_layer_norm=True)
        _copy_exc_weights(ei, ee)

        x = torch.randn(4, 7, 6)
        out_ei, hn_ei = ei(x)
        out_ee, hn_ee = ee(x)

        self.assertTrue(
            torch.allclose(out_ei, out_ee, atol=1e-5),
            msg="E/I output should match excitatory RNN + LayerNorm at init",
        )
        self.assertTrue(
            torch.allclose(hn_ei, hn_ee, atol=1e-5),
            msg="Final hidden state should match excitatory RNN + LayerNorm at init",
        )

if __name__ == "__main__":
    unittest.main()
