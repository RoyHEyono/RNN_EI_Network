import unittest

import torch
import torch.nn.functional as F

from experiments.dense_fmnist.cli import build_parser
from experiments.dense_fmnist.optim import param_groups
from inhibition.densenet import (
    EDenseNet,
    EIDenseNet,
    INormDenseNet,
    LN_NORMALIZE,
    MEAN_NORMALIZE,
    NO_NORMALIZE,
    VAR_NORMALIZE,
    norm_type_from_flags,
)
from inhibition.normalization import (
    DivisiveNormalize,
    LayerNormalizeCustom,
    MeanNormalize,
)


class TestForwardNormalizations(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(6, 20, requires_grad=True)
        self.g = torch.randn(6, 20)

    def _grad(self, module):
        x = self.x.detach().clone().requires_grad_(True)
        module(x).backward(self.g)
        return x.grad

    def test_mean_normalize_forward_and_backward(self):
        out = MeanNormalize()(self.x)
        self.assertTrue(torch.allclose(out, self.x - self.x.mean(-1, keepdim=True)))
        expected = self.g - self.g.mean(-1, keepdim=True)
        self.assertTrue(torch.allclose(self._grad(MeanNormalize()), expected, atol=1e-6))

    def test_divisive_normalize_forward(self):
        sigma = torch.sqrt(self.x.var(-1, keepdim=True, unbiased=False) + 1e-5)
        self.assertTrue(torch.allclose(DivisiveNormalize()(self.x), self.x / sigma))

    def test_layer_normalize_matches_torch(self):
        ours = LayerNormalizeCustom()(self.x)
        theirs = F.layer_norm(self.x, self.x.shape[-1:], eps=1e-5)
        self.assertTrue(torch.allclose(ours, theirs, atol=1e-5))

    def test_layer_normalize_backward_matches_torch(self):
        ref = self.x.detach().clone().requires_grad_(True)
        F.layer_norm(ref, ref.shape[-1:], eps=1e-5).backward(self.g)
        self.assertTrue(torch.allclose(self._grad(LayerNormalizeCustom()), ref.grad, atol=1e-5))

    def test_no_backward_routes_gradient_around_the_norm(self):
        """LN- : normalization still shapes activity, but not the error signal."""
        grad = self._grad(LayerNormalizeCustom(no_backward=True))
        self.assertTrue(torch.allclose(grad, self.g))
        # The forward is unchanged by the detach.
        self.assertTrue(
            torch.allclose(
                LayerNormalizeCustom(no_backward=True)(self.x),
                LayerNormalizeCustom(no_backward=False)(self.x),
            )
        )


class TestNormTypeSelection(unittest.TestCase):
    def test_flag_precedence(self):
        self.assertEqual(norm_type_from_flags(0, 0, 0), NO_NORMALIZE)
        self.assertEqual(norm_type_from_flags(1, 0, 0), MEAN_NORMALIZE)
        self.assertEqual(norm_type_from_flags(0, 1, 0), VAR_NORMALIZE)
        self.assertEqual(norm_type_from_flags(0, 0, 1), LN_NORMALIZE)
        self.assertEqual(norm_type_from_flags(1, 1, 1), MEAN_NORMALIZE)


class TestDenseNets(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.rand(16, 1, 28, 28)

    def test_shapes_and_layer_count(self):
        for net in (EIDenseNet(hidden_size=64), EDenseNet(hidden_size=64),
                    INormDenseNet(hidden_size=64)):
            self.assertEqual(net.hidden_names, ["fc0", "fc1"])
            self.assertEqual(net(self.x).shape, (16, 10))

    def test_e_only_net_has_no_inhibitory_weights_in_hidden_layers(self):
        net = EDenseNet(hidden_size=64)
        for layer in net.hidden_layers():
            self.assertFalse(hasattr(layer, "W_IE"))

    def test_inorm_is_layer_normalized_at_init(self):
        """The SVD initialization should already produce mean 0, variance 1."""
        net = INormDenseNet(hidden_size=64)
        net(self.x)
        for h_prev, layer in zip(net._layer_inputs, net.hidden_layers()):
            z = layer(h_prev)
            self.assertTrue(torch.allclose(z.mean(-1), torch.zeros(16), atol=1e-4))
            self.assertTrue(torch.allclose(z.var(-1, unbiased=False), torch.ones(16), atol=1e-3))

    def test_inorm_local_loss_near_zero_at_init(self):
        net = INormDenseNet(hidden_size=64)
        net(self.x)
        moments, diagnostics = net.local_loss()
        self.assertLess(moments.item(), 1e-4)
        for value in diagnostics.values():
            self.assertLess(value, 1e-4)

    def test_local_loss_requires_a_forward_pass(self):
        with self.assertRaises(RuntimeError):
            INormDenseNet(hidden_size=64).local_loss()

    def test_freeze_ei_keeps_the_averaging_weights_fixed(self):
        net = INormDenseNet(hidden_size=64, freeze_ei=True)
        net(self.x)
        moments, _ = net.local_loss()
        moments.backward()
        for layer in net.hidden_layers():
            self.assertFalse(layer.W_EI.requires_grad)
            self.assertFalse(layer.U_EI.requires_grad)
            self.assertIsNone(layer.W_EI.grad)
            self.assertIsNotNone(layer.W_IE.grad)
            self.assertIsNotNone(layer.U_IE.grad)

    def test_subtractive_only_drops_the_divisive_pathway(self):
        net = INormDenseNet(hidden_size=64, shunting=False)
        layer = net.fc0
        h = torch.flatten(self.x, 1)
        expected = (
            F.linear(h, layer.W_EE) + layer.bias - F.linear(F.linear(h, layer.W_IE), layer.W_EI)
        )
        self.assertTrue(torch.allclose(layer(h), expected, atol=1e-5))


class TestGradNormAndLateralInhibition(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.rand(16, 1, 28, 28)

    def test_fa_weights_are_a_positive_convex_combination(self):
        layer = INormDenseNet(hidden_size=64).fc0
        self.assertTrue((layer.fa_weights > 0).all())
        self.assertAlmostEqual(layer.fa_weights.sum().item(), 1.0, places=5)

    def test_uniform_fa_weights_reproduce_exact_centering(self):
        """Theorem 1's limiting case: uniform pooling weights == mean subtraction."""
        torch.manual_seed(1)
        z = torch.randn(4, 12, requires_grad=True)
        g = torch.randn(4, 12)
        layer = INormDenseNet(hidden_size=12).fc0
        with torch.no_grad():
            layer.fa_weights.copy_(torch.full((12,), 1.0 / 12))
        layer.grad_norm.ln_feedback = "fa_center"
        layer.grad_norm(z, weights=layer.fa_weights).backward(g)
        self.assertTrue(torch.allclose(z.grad, g - g.mean(-1, keepdim=True), atol=1e-6))

    def test_full_gradnorm_aligns_perfectly_with_layernorm(self):
        net = INormDenseNet(hidden_size=64, ln_feedback="full",
                            gradient_norm=True, track_alignment=True)
        net(self.x)
        self.assertAlmostEqual(net.fc0.gradient_alignment_val, 1.0, places=4)

    def test_detaching_the_norm_breaks_gradient_alignment(self):
        net = INormDenseNet(hidden_size=64, ln_feedback="full",
                            gradient_norm=False, track_alignment=True)
        net(self.x)
        self.assertLess(net.fc0.gradient_alignment_val, 0.9)

    def test_partial_feedback_rules_are_partially_aligned(self):
        for feedback in ("center", "scale", "decorrelate"):
            net = INormDenseNet(hidden_size=64, ln_feedback=feedback,
                                gradient_norm=True, track_alignment=True)
            net(self.x)
            value = net.fc0.gradient_alignment_val
            self.assertGreater(value, 0.0, msg=feedback)
            self.assertLess(value, 0.999, msg=feedback)

    def test_alignment_is_not_computed_when_tracking_is_off(self):
        net = INormDenseNet(hidden_size=64, track_alignment=False)
        net(self.x)
        self.assertNotEqual(net.fc0.gradient_alignment_val, net.fc0.gradient_alignment_val)


class TestParamGroups(unittest.TestCase):
    def test_inhibitory_projections_get_their_own_learning_rates(self):
        args = build_parser(homeostasis_defaults=True).parse_args(
            ["--opt.lr=0.01", "--opt.inhib_lrs.wei=0.002", "--opt.inhib_lrs.wix=0.5",
             "--model.freeze_ei=0"]
        )
        net = INormDenseNet(hidden_size=64, freeze_ei=False)
        groups = param_groups(net, args)
        self.assertEqual([g["lr"] for g in groups], [0.01, 0.5, 0.002])
        # fc0/fc1 contribute W_IE and U_IE to the wix group; the readout has W_IE only.
        self.assertEqual(len(groups[1]["params"]), 5)
        self.assertEqual(len(groups[2]["params"]), 5)

    def test_frozen_parameters_are_excluded(self):
        args = build_parser(homeostasis_defaults=True).parse_args([])
        net = INormDenseNet(hidden_size=64, freeze_ei=True)
        groups = param_groups(net, args)
        # Only the readout layer's W_EI remains trainable in the wei group.
        self.assertEqual(len(groups[2]["params"]), 1)

    def test_single_learning_rate_when_separate_lrs_are_disabled(self):
        args = build_parser(homeostasis_defaults=False).parse_args(
            ["--opt.lr=0.02", "--opt.use_sep_inhib_lrs=0"]
        )
        groups = param_groups(EIDenseNet(hidden_size=64), args)
        self.assertTrue(all(g["lr"] == 0.02 for g in groups))


if __name__ == "__main__":
    unittest.main()
