import unittest

import torch

from inhibition.rnn import SimpleEERNN


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


if __name__ == "__main__":
    unittest.main()
