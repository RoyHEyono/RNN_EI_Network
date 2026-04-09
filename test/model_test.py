import unittest

import numpy as np
import torch
import torch.nn.functional as F

from inhibition.model import DeepNet, MNIST_FLAT, Net, RNNNet, inorm_param_groups


def _inorm_manual(layer, x):
    """Match INormLayer forward (subtractive/divisive paths, same as dense_test)."""
    hex_ = torch.matmul(x, layer.W_EE.T) + layer.bias
    hi = torch.matmul(x, layer.W_IE.T)
    hi = torch.matmul(hi, layer.W_EI.T)
    z_d_squared = torch.matmul(torch.matmul(x, layer.U_IE.T) ** 2, layer.U_EI.T)
    z_d = torch.sqrt(z_d_squared + layer.eps)
    return (hex_ - hi) / z_d


def _net_forward_manual(net: Net, x):
    h0 = torch.flatten(x, 1)
    z1 = _inorm_manual(net.fc1, h0)
    h1 = F.relu(z1)
    z2 = _inorm_manual(net.fc2, h1)
    return z2, h0, h1, z1, z2


class TestNetINormLayers(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.net = Net()
        self.x = torch.randn(self.batch_size, 1, 28, 28)

    def test_forward_output_numerically(self):
        manual_out, *_ = _net_forward_manual(self.net, self.x)
        net_out = self.net(self.x)
        self.assertTrue(
            torch.equal(manual_out, net_out),
            msg="Numerical test for full Net forward",
        )

    # def test_init_moments(self):
    #     _, _, _, z1, z2 = _net_forward_manual(self.net, self.x)
    #     for z, name in ((z1, "fc1"), (z2, "fc2")):
    #         var = z.var(dim=-1, unbiased=True)
    #         mu = z.mean(dim=-1)
    #         self.assertTrue(
    #             torch.allclose(var, torch.ones_like(var), atol=1e-6),
    #             msg=f"{name}: expected variance=1, got {var}",
    #         )
    #         self.assertTrue(
    #             torch.allclose(mu, torch.zeros_like(mu), atol=1e-6),
    #             msg=f"{name}: expected mu=0, got {mu}",
    #         )

    def test_forward_against_layernorm_at_init(self):
        h0 = torch.flatten(self.x, 1)
        h1 = F.relu(self.net.fc1(h0))

        ln1 = torch.nn.LayerNorm(self.net.fc1.W_EE.shape[0], elementwise_affine=False)
        hex1 = torch.matmul(h0, self.net.fc1.W_EE.T)
        self.assertTrue(
            torch.allclose(ln1(hex1), self.net.fc1(h0), atol=1e-5),
            msg="fc1 init doesn't match LayerNorm on excitatory output",
        )

        ln2 = torch.nn.LayerNorm(self.net.fc2.W_EE.shape[0], elementwise_affine=False)
        hex2 = torch.matmul(h1, self.net.fc2.W_EE.T)
        self.assertTrue(
            torch.allclose(ln2(hex2), self.net.fc2(h1), atol=1e-5),
            msg="fc2 init doesn't match LayerNorm on excitatory output",
        )

    def test_forward_output_shape(self):
        output = self.net.forward(self.x)
        self.assertEqual(output.shape, (self.batch_size, 10))

    def test_weights_initialized_correctly(self):
        fc1, fc2 = self.net.fc1, self.net.fc2
        self.assertEqual(fc1.W_EE.shape, (128, MNIST_FLAT))
        self.assertEqual(torch.matmul(fc1.W_EI, fc1.W_IE).shape, (128, MNIST_FLAT))
        self.assertEqual(fc2.W_EE.shape, (10, 128))
        self.assertEqual(torch.matmul(fc2.W_EI, fc2.W_IE).shape, (10, 128))

    def test_local_loss_tuple_near_zero(self):
        _, h0, h1, _, _ = _net_forward_manual(self.net, self.x)
        for layer, h_prev, label in (
            (self.net.fc1, h0, "fc1"),
            (self.net.fc2, h1, "fc2"),
        ):
            moments_loss, ln_loss = layer.local_loss(h_prev)
            m = moments_loss.item() if torch.is_tensor(moments_loss) else float(moments_loss)
            self.assertTrue(
                np.isclose(m, 0.0, atol=1e-5),
                msg=f"{label}: expected moments term near 0, got {m}",
            )
            self.assertTrue(
                np.isclose(ln_loss, 0.0, atol=1e-5),
                msg=f"{label}: expected LN ground-truth MSE near 0, got {ln_loss}",
            )

    def test_inhibitory_weights_gradient_updated_in_forward(self):
        self.net.zero_grad()
        _, h0, h1, _, _ = _net_forward_manual(self.net, self.x)
        m1, _ = self.net.fc1.local_loss(h0)
        m2, _ = self.net.fc2.local_loss(h1)
        (m1 + m2).backward()

        for name, layer in (("fc1", self.net.fc1), ("fc2", self.net.fc2)):
            self.assertIsNotNone(layer.W_EI.grad, msg=f"{name} W_EI")
            self.assertIsNotNone(layer.U_EI.grad, msg=f"{name} U_EI")
            self.assertIsNotNone(layer.W_IE.grad, msg=f"{name} W_IE")
            self.assertIsNotNone(layer.U_IE.grad, msg=f"{name} U_IE")
            self.assertIsNone(layer.W_EE.grad, msg=f"{name} W_EE")
            self.assertIsNone(layer.bias.grad, msg=f"{name} bias")

    def test_excitatory_weights_gradient_updated_in_backward(self):
        self.net.zero_grad()
        output = self.net(self.x)
        self.assertIsNone(self.net.fc1.W_EE.grad)
        self.assertIsNone(self.net.fc1.bias.grad)
        self.assertIsNone(self.net.fc2.W_EE.grad)
        self.assertIsNone(self.net.fc2.bias.grad)

        loss = output.sum()
        loss.backward()

        for name, layer in (("fc1", self.net.fc1), ("fc2", self.net.fc2)):
            self.assertIsNotNone(layer.W_EE.grad, msg=f"{name} W_EE")
            self.assertIsNotNone(layer.bias.grad, msg=f"{name} bias")
            self.assertIsNone(layer.W_EI.grad, msg=f"{name} W_EI")
            self.assertIsNone(layer.U_EI.grad, msg=f"{name} U_EI")
            self.assertIsNone(layer.W_IE.grad, msg=f"{name} W_IE")
            self.assertIsNone(layer.U_IE.grad, msg=f"{name} U_IE")

    def test_excitatory_and_inhibitory_gradients_from_separate_losses(self):
        """Task NLL path vs local INorm losses: combined backward should match two isolated runs."""
        net = self.net
        x = self.x

        net.zero_grad()
        out_task = net(x)
        loss_task = out_task.sum()
        loss_task.backward()
        exc = {
            "fc1_W_EE": net.fc1.W_EE.grad.clone(),
            "fc1_bias": net.fc1.bias.grad.clone(),
            "fc2_W_EE": net.fc2.W_EE.grad.clone(),
            "fc2_bias": net.fc2.bias.grad.clone(),
        }

        net.zero_grad()
        _, (h0, h1) = net(x, return_layer_inputs=True)
        m1, _ = net.fc1.local_loss(h0)
        m2, _ = net.fc2.local_loss(h1)
        loss_local = m1 + m2
        loss_local.backward()
        inh = {}
        for prefix, layer in (("fc1", net.fc1), ("fc2", net.fc2)):
            for pname in ("W_EI", "U_EI", "W_IE", "U_IE"):
                inh[f"{prefix}_{pname}"] = getattr(layer, pname).grad.clone()

        net.zero_grad()
        out_both, (h0b, h1b) = net(x, return_layer_inputs=True)
        (out_both.sum() + net.fc1.local_loss(h0b)[0] + net.fc2.local_loss(h1b)[0]).backward()

        for key, ref in exc.items():
            layer_name, param = key.split("_", 1)
            layer = net.fc1 if layer_name == "fc1" else net.fc2
            g = layer.W_EE.grad if param == "W_EE" else layer.bias.grad
            self.assertTrue(
                torch.allclose(g, ref, rtol=1e-6, atol=1e-8),
                msg=f"Excitatory grad for {key} should match task-loss-only backward",
            )

        for key, ref in inh.items():
            layer_name, pname = key.split("_", 1)
            layer = net.fc1 if layer_name == "fc1" else net.fc2
            g = getattr(layer, pname).grad
            self.assertTrue(
                torch.allclose(g, ref, rtol=1e-6, atol=1e-8),
                msg=f"Inhibitory grad for {key} should match local-loss-only backward",
            )


class TestInormParamGroups(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = Net()

    def test_returns_three_groups_with_learning_rates(self):
        groups = inorm_param_groups(self.net, lr_exc=1e-4, lr_ie=1e-2, lr_ei=1e-3)
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0]["lr"], 1e-4)
        self.assertEqual(groups[1]["lr"], 1e-2)
        self.assertEqual(groups[2]["lr"], 1e-3)
        for g in groups:
            self.assertIn("params", g)
            self.assertIsInstance(g["params"], list)

    def test_optimizer_accepts_param_groups(self):
        groups = inorm_param_groups(self.net, 0.01, 0.02, 0.03)
        opt = torch.optim.Adadelta(groups)
        self.assertEqual(len(opt.param_groups), 3)

    def test_deep_net_all_parameters_in_groups(self):
        model = DeepNet()
        groups = inorm_param_groups(model, 1e-3, 1e-3, 1e-3)
        seen = set()
        for g in groups:
            for p in g["params"]:
                self.assertNotIn(id(p), seen, msg="parameter appears in more than one group")
                seen.add(id(p))
        for p in model.parameters():
            self.assertIn(id(p), seen, msg="model parameter missing from inorm_param_groups")

    def test_sgd_step_uses_correct_lr_per_group_after_e_and_i_backward(self):
        """Backward through task + local losses (E and I), then unit grads so Δw = -lr per element."""
        torch.manual_seed(7)
        net = Net()
        x = torch.randn(2, 1, 28, 28)
        lr_exc, lr_ie, lr_ei = 0.1, 0.25, 0.4
        groups = inorm_param_groups(net, lr_exc, lr_ie, lr_ei)
        opt = torch.optim.SGD(groups, momentum=0.0)

        net.zero_grad()
        out, (h0, h1) = net(x, return_layer_inputs=True)
        (
            out.sum()
            + net.fc1.local_loss(h0)[0]
            + net.fc2.local_loss(h1)[0]
        ).backward()

        for p in net.parameters():
            p.grad = torch.ones_like(p)

        before = {id(p): p.data.clone() for p in net.parameters()}
        opt.step()

        exc_ids = {id(p) for p in groups[0]["params"]}
        ie_ids = {id(p) for p in groups[1]["params"]}
        ei_ids = {id(p) for p in groups[2]["params"]}

        for p in net.parameters():
            delta = p.data - before[id(p)]
            if id(p) in exc_ids:
                lr = lr_exc
            elif id(p) in ie_ids:
                lr = lr_ie
            elif id(p) in ei_ids:
                lr = lr_ei
            else:
                self.fail("parameter not in any inorm_param_groups list")
            want = -lr * torch.ones_like(delta)
            self.assertTrue(
                torch.allclose(delta, want),
                msg=f"expected update -lr*1 with lr={lr}, got max abs err {(delta - want).abs().max().item()}",
            )


class TestRNNNet(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.batch_size = 4
        self.model = RNNNet(hidden_size=32)
        self.x = torch.randn(self.batch_size, 1, 28, 28)

    def test_forward_output_shape(self):
        output = self.model(self.x)
        self.assertEqual(output.shape, (self.batch_size, 10))

    def test_uses_eidense_head(self):
        self.assertEqual(self.model.head.__class__.__name__, "EiDenseLayer")

    def test_forward_with_layer_inputs(self):
        logits, (seq, rnn_out, h_n) = self.model(self.x, return_layer_inputs=True)
        self.assertEqual(logits.shape, (self.batch_size, 10))
        self.assertEqual(seq.shape, (self.batch_size, 28, 28))
        self.assertEqual(rnn_out.shape, (self.batch_size, 28, 32))
        self.assertEqual(h_n.shape, (self.batch_size, 32))
