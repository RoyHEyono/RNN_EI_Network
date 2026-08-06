import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from inhibition.normalization import ParametrizedLayerNorm
from inhibition.rnn import SimpleEERNN

try:
    import neurogym as ngym

    _NEUROGYM_AVAILABLE = True
except ImportError:
    _NEUROGYM_AVAILABLE = False


class TestParametrizedLayerNorm(unittest.TestCase):
    """Isolated checks for ``ParametrizedLayerNorm`` (shapes, STE, aux, exact init)."""

    def setUp(self):
        torch.manual_seed(0)
        self.batch_size = 32
        self.input_size = 8
        self.hidden_size = 16
        self.eps = 1e-5
        # Build the module through a SimpleEERNN so its divisive stats predictors
        # are initialized from the RNN weights (exact LayerNorm at init).
        self.rnn = SimpleEERNN(
            self.input_size,
            self.hidden_size,
            batch_first=True,
            use_parametrized_layer_norm=True,
            layer_norm_eps=self.eps,
        )
        self.module = self.rnn.layer_norm
        self.x_t, self.h_prev, self.pre_act = self._exact_batch()

    def _exact_batch(self):
        """Build a batch whose pre-activation is the RNN's actual linear drive.

        Because the stats predictors are initialized from the RNN weights, the
        LayerNorm mean/variance of this drive are represented exactly.
        """
        x_t = torch.randn(self.batch_size, self.input_size)
        h_prev = torch.randn(self.batch_size, self.hidden_size)
        with torch.no_grad():
            pre_act = torch.matmul(x_t, self.rnn.W_XE.T) + torch.matmul(
                h_prev, self.rnn.W_EE.T
            )
            if self.rnn.bias is not None:
                pre_act = pre_act + self.rnn.bias
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

    def test_predicted_norm_matches_layer_norm_at_init(self):
        """The divisive init reproduces LayerNorm exactly (notebook cells 78-83)."""
        with torch.no_grad():
            pred_mean, pred_var = self.module._predict_stats(self.x_t, self.h_prev)
            predicted = (self.pre_act - pred_mean) / torch.sqrt(pred_var)
            target = F.layer_norm(self.pre_act, self.pre_act.shape[-1:], eps=self.eps)
            _, aux = self.module(self.pre_act, self.x_t, self.h_prev)
        max_abs = (predicted - target).abs().max().item()
        self.assertTrue(
            torch.allclose(predicted, target, atol=1e-3, rtol=1e-3),
            msg=f"predicted norm should match LayerNorm at init (max |diff|={max_abs:.2e})",
        )
        self.assertLess(
            aux.item(), 1e-4, msg=f"aux should be ~0 at exact init: {aux.item():.3e}"
        )

    def test_task_loss_does_not_train_proj(self):
        """With STE, a loss on the module output trains ``pre_act`` via LN, not the stats nets."""
        self.module.zero_grad(set_to_none=True)
        pre_act = self.pre_act.detach().requires_grad_(True)
        out, _ = self.module(pre_act, self.x_t, self.h_prev)
        out.pow(2).mean().backward()
        self.assertIsNotNone(pre_act.grad)
        self.assertGreater(pre_act.grad.abs().sum().item(), 0.0)
        for p in self.module.mean_net.parameters():
            self.assertIsNone(p.grad)
        for p in self.module.var_net.parameters():
            self.assertIsNone(p.grad)

    def test_aux_loss_trains_stats_nets(self):
        """Aux trains ``mean_net``/``var_net`` only; must not send grads into ``x_t`` / ``h_prev`` / ``pre_act``."""
        self.module.zero_grad(set_to_none=True)
        x_t = self.x_t.detach().requires_grad_(True)
        h_prev = self.h_prev.detach().requires_grad_(True)
        # A pre_act unrelated to the predicted stats makes aux (and its grads) nonzero.
        pre_act = torch.randn(
            self.batch_size, self.hidden_size, requires_grad=True
        )
        _, aux = self.module(pre_act, x_t, h_prev)
        aux.backward()
        for p in self.module.mean_net.parameters():
            self.assertIsNotNone(p.grad)
            self.assertGreater(p.grad.abs().sum().item(), 0.0)
        for p in self.module.var_net.parameters():
            self.assertIsNotNone(p.grad)
            self.assertGreater(p.grad.abs().sum().item(), 0.0)
        self.assertIsNone(x_t.grad)
        self.assertIsNone(h_prev.grad)
        self.assertIsNone(pre_act.grad)

    def test_predict_stats_shapes_and_positivity(self):
        pred_mean, pred_var = self.module._predict_stats(self.x_t, self.h_prev)
        self.assertEqual(pred_mean.shape, (self.batch_size, 1))
        self.assertEqual(pred_var.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(pred_var > 0))

    def test_aux_recovers_layer_norm_after_perturbation(self):
        """Perturb the stats predictors, then aux training drives them back to LN."""
        torch.manual_seed(1)
        rnn = SimpleEERNN(
            self.input_size,
            self.hidden_size,
            batch_first=True,
            use_parametrized_layer_norm=True,
            layer_norm_eps=self.eps,
        )
        module = rnn.layer_norm
        x_t = torch.randn(self.batch_size, self.input_size)
        h_prev = torch.randn(self.batch_size, self.hidden_size)
        with torch.no_grad():
            pre_act = torch.matmul(x_t, rnn.W_XE.T) + torch.matmul(h_prev, rnn.W_EE.T)
            if rnn.bias is not None:
                pre_act = pre_act + rnn.bias
            # Small perturbation away from the exact solution.
            for p in module.parameters():
                p.add_(torch.randn_like(p) * 0.05)
            _, init_aux = module(pre_act, x_t, h_prev)
            init_aux = init_aux.item()

        opt = torch.optim.Adam(module.parameters(), lr=3e-3)
        best_aux = float("inf")
        best_state = None
        for _ in range(2000):
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
            pred_mean, pred_var = module._predict_stats(x_t, h_prev)
            predicted = (pre_act - pred_mean) / torch.sqrt(pred_var)
            target = F.layer_norm(pre_act, pre_act.shape[-1:], eps=self.eps)
            mse = F.mse_loss(predicted, target).item()

        self.assertLess(
            best_aux,
            5e-4,
            msg=f"aux did not converge: init={init_aux:.3e} best={best_aux:.3e}",
        )
        self.assertLess(mse, 5e-3, msg=f"activation MSE too high: {mse:.3e}")


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

    def test_parametrized_layer_norm_stats_nets_shape(self):
        input_size, hidden_size = 6, 8
        model = SimpleEERNN(
            input_size=input_size,
            hidden_size=hidden_size,
            use_layer_norm=True,
            use_parametrized_layer_norm=True,
        )
        feat_dim = input_size + hidden_size
        ln = model.layer_norm
        # mean predictor: a single linear map (subtractive inhibition).
        self.assertIsInstance(ln.mean_net, nn.Linear)
        self.assertEqual(ln.mean_net.in_features, feat_dim)
        self.assertEqual(ln.mean_net.out_features, 1)
        # variance predictor: Linear -> square -> average (divisive inhibition).
        self.assertIsInstance(ln.var_net, nn.Sequential)
        self.assertEqual(ln.var_net[0].in_features, feat_dim)
        self.assertEqual(ln.var_net[0].out_features, hidden_size)
        self.assertEqual(ln.var_net[-1].out_features, 1)

    def test_parametrized_layer_norm_forward_shapes(self):
        model = SimpleEERNN(
            input_size=6,
            hidden_size=8,
            batch_first=True,
            use_layer_norm=True,
            use_parametrized_layer_norm=True,
        )
        x = torch.randn(4, 7, 6)
        output, h_n, aux_losses = model(x)
        self.assertEqual(output.shape, (4, 7, 8))
        self.assertEqual(h_n.shape, (4, 8))

    def test_default_layer_norm_is_nn_layer_norm(self):
        model = SimpleEERNN(input_size=4, hidden_size=8, use_layer_norm=True)
        self.assertIsInstance(model.layer_norm, nn.LayerNorm)
        self.assertFalse(model.use_parametrized_layer_norm)

    def test_parametrized_layer_norm_matches_nn_layer_norm_at_init(self):
        """With shared weights, the divisive init makes the rollout match nn.LayerNorm."""
        model_nn_ln = SimpleEERNN(
            input_size=8, hidden_size=10, batch_first=False,
            use_parametrized_layer_norm=False,
        )
        model_param_ln = SimpleEERNN(
            input_size=8, hidden_size=10, batch_first=False,
            use_parametrized_layer_norm=True,
        )
        with torch.no_grad():
            model_param_ln.W_XE.copy_(model_nn_ln.W_XE)
            model_param_ln.W_EE.copy_(model_nn_ln.W_EE)
            model_param_ln.bias.copy_(model_nn_ln.bias)
        # Re-init the stats predictors from the (now shared) weights.
        model_param_ln.layer_norm.init_from_rnn_weights(
            model_param_ln.W_XE, model_param_ln.W_EE, model_param_ln.bias
        )

        x = torch.randn(5, 3, 8)  # (seq, batch, feat)
        with torch.no_grad():
            output_nn, h_nn = model_nn_ln(x)
            output_param, h_param, _ = model_param_ln(x)

        self.assertTrue(
            torch.allclose(output_nn, output_param, atol=5e-3, rtol=1e-3),
            f"Outputs diverge: max diff = {(output_nn - output_param).abs().max().item():.6f}",
        )
        self.assertTrue(
            torch.allclose(h_nn, h_param, atol=5e-3, rtol=1e-3),
            f"Hidden states diverge: max diff = {(h_nn - h_param).abs().max().item():.6f}",
        )

    @unittest.skipUnless(_NEUROGYM_AVAILABLE, "neurogym not installed")
    def test_parametrized_layer_norm_matches_nn_layer_norm_neurogym_task(self):
        dataset = ngym.Dataset(
            "PerceptualDecisionMaking-v0",
            env_kwargs={"dt": 100},
            batch_size=3,
            seq_len=5,
        )
        # Dataset.__init__ pre-fills its cache using an unseeded env RNG, so reseed
        # and rebuild the cache to make the sampled batch reproducible.
        dataset.seed(0)
        dataset._cache()
        inputs_np, _ = dataset()
        x = torch.from_numpy(inputs_np).float()  # (seq, batch, ob_size) already
        ob_size = dataset.env.observation_space.shape[0]

        model_nn_ln = SimpleEERNN(
            input_size=ob_size, hidden_size=10, batch_first=False,
            use_parametrized_layer_norm=False,
        )
        model_param_ln = SimpleEERNN(
            input_size=ob_size, hidden_size=10, batch_first=False,
            use_parametrized_layer_norm=True,
        )
        with torch.no_grad():
            model_param_ln.W_XE.copy_(model_nn_ln.W_XE)
            model_param_ln.W_EE.copy_(model_nn_ln.W_EE)
            model_param_ln.bias.copy_(model_nn_ln.bias)
        model_param_ln.layer_norm.init_from_rnn_weights(
            model_param_ln.W_XE, model_param_ln.W_EE, model_param_ln.bias
        )

        with torch.no_grad():
            output_nn, h_nn = model_nn_ln(x)
            output_param, h_param, _ = model_param_ln(x)

        self.assertTrue(
            torch.allclose(output_nn, output_param, atol=5e-3, rtol=1e-3),
            f"Outputs diverge: max diff = {(output_nn - output_param).abs().max().item():.6f}",
        )
        self.assertTrue(
            torch.allclose(h_nn, h_param, atol=5e-3, rtol=1e-3),
            f"Hidden states diverge: max diff = {(h_nn - h_param).abs().max().item():.6f}",
        )


if __name__ == "__main__":
    unittest.main()
