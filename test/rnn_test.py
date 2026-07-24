import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from inhibition.normalization import ParametrizedLayerNorm
from inhibition.rnn import SimpleEERNN


class TestParametrizedLayerNorm(unittest.TestCase):
    """Isolated checks for ``ParametrizedLayerNorm`` (shapes, STE, aux fit)."""

    def setUp(self):
        torch.manual_seed(0)
        self.batch_size = 32
        self.input_size = 8
        self.hidden_size = 16
        self.eps = 1e-5
        self.module = ParametrizedLayerNorm(
            self.input_size, self.hidden_size, eps=self.eps
        )
        self.x_t, self.h_prev, self.pre_act = self._exact_batch()

    def _exact_batch(self):
        """Build a batch whose LN stats are exactly representable by ``proj``.

        Mean is linear in ``(x_t, h_prev)``; variance is ``softplus`` of a
        linear map — the same functional form as ``_predict_stats``.
        """
        feat_dim = self.input_size + self.hidden_size
        w_true = torch.randn(2, feat_dim) * 0.5
        b_true = torch.randn(2) * 0.5
        x_t = torch.randn(self.batch_size, self.input_size)
        h_prev = torch.randn(self.batch_size, self.hidden_size)
        stats = torch.cat([x_t, h_prev], dim=-1) @ w_true.T + b_true
        mu = stats[..., :1]
        empiric_var = F.softplus(stats[..., 1:])
        z = torch.randn(self.batch_size, self.hidden_size)
        z = z - z.mean(dim=-1, keepdim=True)
        z = z / z.var(dim=-1, keepdim=True, unbiased=False).sqrt().clamp_min(
            self.eps
        )
        pre_act = mu + z * empiric_var.sqrt()
        return x_t, h_prev, pre_act

    def test_forward_shapes_and_aux_scalar(self):
        out, aux = self.module(self.pre_act, self.x_t, self.h_prev)
        self.assertEqual(out.shape, self.pre_act.shape)
        self.assertEqual(aux.shape, ())
        self.assertTrue(torch.isfinite(aux))

    def test_ste_forward_matches_predicted_norm(self):
        pred_mean, pred_var = self.module._predict_stats(self.x_t, self.h_prev)
        predicted = (self.pre_act - pred_mean) / torch.sqrt(pred_var)
        out, _ = self.module(self.pre_act, self.x_t, self.h_prev)
        self.assertTrue(
            torch.allclose(out, predicted, atol=1e-6, rtol=1e-5),
            msg="STE forward values should equal predicted-stats normalization",
        )

    def test_ste_backward_matches_layer_norm_grad(self):
        pre_ste = self.pre_act.detach().requires_grad_(True)
        out, _ = self.module(pre_ste, self.x_t, self.h_prev)
        out.sum().backward()

        pre_ln = self.pre_act.detach().requires_grad_(True)
        F.layer_norm(pre_ln, pre_ln.shape[-1:], eps=self.eps).sum().backward()

        self.assertTrue(
            torch.allclose(pre_ste.grad, pre_ln.grad, atol=1e-6, rtol=1e-5),
            msg="STE backward should match true LayerNorm Jacobian on pre_act",
        )

    def test_task_loss_does_not_train_proj(self):
        """With STE, a loss on the module output trains ``pre_act`` via LN, not ``proj``."""
        self.module.zero_grad(set_to_none=True)
        pre_act = self.pre_act.detach().requires_grad_(True)
        out, _ = self.module(pre_act, self.x_t, self.h_prev)
        out.pow(2).mean().backward()
        self.assertIsNotNone(pre_act.grad)
        self.assertGreater(pre_act.grad.abs().sum().item(), 0.0)
        self.assertIsNone(self.module.proj.weight.grad)
        self.assertIsNone(self.module.proj.bias.grad)

    def test_aux_loss_trains_proj(self):
        """Aux trains ``proj`` only; must not send grads into ``x_t`` / ``h_prev`` / ``pre_act``."""
        self.module.zero_grad(set_to_none=True)
        x_t = self.x_t.detach().requires_grad_(True)
        h_prev = self.h_prev.detach().requires_grad_(True)
        pre_act = self.pre_act.detach().requires_grad_(True)
        _, aux = self.module(pre_act, x_t, h_prev)
        aux.backward()
        self.assertIsNotNone(self.module.proj.weight.grad)
        self.assertIsNotNone(self.module.proj.bias.grad)
        self.assertGreater(self.module.proj.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.module.proj.bias.grad.abs().sum().item(), 0.0)
        self.assertIsNone(x_t.grad)
        self.assertIsNone(h_prev.grad)
        self.assertIsNone(pre_act.grad)

    def test_predict_stats_shapes_and_positivity(self):
        pred_mean, pred_var = self.module._predict_stats(self.x_t, self.h_prev)
        self.assertEqual(pred_mean.shape, (self.batch_size, 1))
        self.assertEqual(pred_var.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(pred_var > 0))

    def test_trains_to_convergence_on_batch(self):
        """Overfit one exactly-representable batch until predicted norm ≈ LN."""
        torch.manual_seed(1)
        module = ParametrizedLayerNorm(
            self.input_size, self.hidden_size, eps=self.eps
        )
        x_t, h_prev, pre_act = self._exact_batch()
        opt = torch.optim.Adam(module.parameters(), lr=1e-2)

        with torch.no_grad():
            _, init_aux = module(pre_act, x_t, h_prev)
            init_aux = init_aux.item()

        best_aux = float("inf")
        best_state = None
        for _ in range(4000):
            opt.zero_grad()
            _, aux = module(pre_act, x_t, h_prev)
            aux.backward()
            opt.step()
            if aux.item() < best_aux:
                best_aux = aux.item()
                best_state = {
                    k: v.detach().clone() for k, v in module.state_dict().items()
                }

        module.load_state_dict(best_state)
        with torch.no_grad():
            _, final_aux_t = module(pre_act, x_t, h_prev)
            final_aux = final_aux_t.item()
            pred_mean, pred_var = module._predict_stats(x_t, h_prev)
            predicted = (pre_act - pred_mean) / torch.sqrt(pred_var)
            target = F.layer_norm(pre_act, pre_act.shape[-1:], eps=self.eps)
            mse = F.mse_loss(predicted, target).item()
            max_abs = (predicted - target).abs().max().item()
            out, _ = module(pre_act, x_t, h_prev)

        self.assertLess(
            final_aux,
            5e-4,
            msg=f"aux did not converge: init={init_aux:.3e} final={final_aux:.3e}",
        )
        self.assertLess(
            final_aux,
            1e-3 * init_aux,
            msg=f"aux drop too small: init={init_aux:.3e} final={final_aux:.3e}",
        )
        self.assertLess(mse, 5e-4, msg=f"activation MSE too high: {mse:.3e}")
        self.assertLess(max_abs, 5e-2, msg=f"max |pred-LN| too high: {max_abs:.3e}")
        self.assertTrue(
            torch.allclose(out, target, atol=5e-2, rtol=1e-3),
            msg="STE forward should match LayerNorm after convergence",
        )


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

    def test_parametrized_layer_norm_proj_shape(self):
        input_size, hidden_size = 6, 8
        model = SimpleEERNN(
            input_size=input_size,
            hidden_size=hidden_size,
            use_layer_norm=True,
            use_parametrized_layer_norm=True,
        )
        self.assertIsInstance(model.layer_norm.proj, nn.Linear)
        self.assertEqual(model.layer_norm.proj.in_features, input_size + hidden_size)
        self.assertEqual(model.layer_norm.proj.out_features, 2)

    def test_parametrized_layer_norm_forward_shapes(self):
        model = SimpleEERNN(
            input_size=6,
            hidden_size=8,
            batch_first=True,
            use_layer_norm=True,
            use_parametrized_layer_norm=True,
        )
        x = torch.randn(4, 7, 6)
        output, h_n = model(x)
        self.assertEqual(output.shape, (4, 7, 8))
        self.assertEqual(h_n.shape, (4, 8))

    def test_default_layer_norm_is_nn_layer_norm(self):
        model = SimpleEERNN(input_size=4, hidden_size=8, use_layer_norm=True)
        self.assertIsInstance(model.layer_norm, nn.LayerNorm)
        self.assertFalse(model.use_parametrized_layer_norm)

    def test_parametrized_layer_norm_vs_nn_layer_norm(self):
        model_nn_ln = SimpleEERNN(input_size=8, hidden_size=10, batch_first=False, use_parametrized_layer_norm = False)
        model_param_ln = SimpleEERNN(input_size=8, hidden_size=10, batch_first=False, use_parametrized_layer_norm = True)
        model_param_ln.W_XE = torch.nn.Parameter(model_nn_ln.W_XE.clone())
        model_param_ln.W_EE = torch.nn.Parameter(model_nn_ln.W_EE.clone())
        model_param_ln.bias = torch.nn.Parameter(model_nn_ln.bias.clone())

        x = torch.randn(5, 3, 8)  # (seq, batch, feat)
        output1, h_n1 = model_nn_ln(x)
        output2, h_n2, aux_2 = model_param_ln(x)
        # aux_2 is the local loss value that needs to be trained
        print(output1)

        pass


if __name__ == "__main__":
    unittest.main()
