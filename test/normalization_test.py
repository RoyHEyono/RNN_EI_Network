import unittest

import torch

from inhibition.normalization import LayerNormFeedbackAblation


class TestLayerNormFeedbackAblation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(3, 5, requires_grad=True)
        self.grad_output = torch.randn(3, 5)
        self.weights = torch.softmax(torch.randn(3, 5), dim=-1)

    def _run_backward(self, mode, no_backward=False, weights=None):
        x = self.x.detach().clone().requires_grad_(True)
        module = LayerNormFeedbackAblation(ln_feedback=mode, no_backward=no_backward)
        out = module(x, weights=weights)
        out.backward(self.grad_output)
        return x.grad, module

    def test_no_backward_passthrough(self):
        grad, _ = self._run_backward(mode="full", no_backward=True)
        self.assertTrue(torch.allclose(grad, self.grad_output))

    def test_center_feedback(self):
        grad, _ = self._run_backward(mode="center")
        expected = self.grad_output - self.grad_output.mean(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(grad, expected, atol=1e-6, rtol=1e-6))

    def test_scale_feedback(self):
        grad, _ = self._run_backward(mode="scale")
        eps = 1e-5
        scale = torch.sqrt(self.x.detach().var(dim=-1, keepdim=True, unbiased=False) + eps)
        expected = self.grad_output / scale
        self.assertTrue(torch.allclose(grad, expected, atol=1e-6, rtol=1e-6))

    def test_decorrelate_feedback(self):
        grad, _ = self._run_backward(mode="decorrelate")
        d = self.x.shape[-1]
        centered = self.x - self.x.mean(dim=-1, keepdim=True)
        dot = (self.grad_output * centered.detach()).sum(dim=-1, keepdim=True)
        expected = self.grad_output - centered.detach() * dot / d
        self.assertTrue(torch.allclose(grad, expected, atol=1e-6, rtol=1e-6))

    def test_fa_center_feedback(self):
        grad, module = self._run_backward(mode="fa_center", weights=self.weights)
        expected = self.grad_output - (self.weights * self.grad_output).sum(
            dim=-1, keepdim=True
        )
        self.assertTrue(torch.allclose(grad, expected, atol=1e-6, rtol=1e-6))
        self.assertIsNotNone(module.grad_norm_delta)
        self.assertTrue(torch.allclose(module.grad_norm_delta, self.grad_output))

    def test_full_feedback(self):
        grad, _ = self._run_backward(mode="full")
        eps = 1e-5
        d = self.x.shape[-1]
        grad_mean = self.grad_output.mean(dim=-1, keepdim=True)
        x_centered = self.x.detach() - self.x.detach().mean(dim=-1, keepdim=True)
        actual_var = torch.sqrt(self.x.detach().var(dim=-1, keepdim=True, unbiased=False) + eps)
        x_hat = x_centered / actual_var
        dot = (self.grad_output * x_hat).mean(dim=-1, keepdim=True)
        expected = (self.grad_output - grad_mean - x_hat * dot) / actual_var
        self.assertTrue(torch.allclose(grad, expected, atol=1e-6, rtol=1e-6))

    def test_full_feedback_matches_torch_layernorm_backward(self):
        x_custom = self.x.detach().clone().requires_grad_(True)
        x_ln = self.x.detach().clone().requires_grad_(True)

        module = LayerNormFeedbackAblation(ln_feedback="full", no_backward=False)
        y_custom = module(x_custom)
        y_custom.backward(self.grad_output)
        grad_custom = x_custom.grad.clone()

        ln = torch.nn.LayerNorm(x_ln.shape[-1], elementwise_affine=False, eps=1e-5)
        y_ln = ln(x_ln)
        y_ln.backward(self.grad_output)
        grad_ln = x_ln.grad.clone()

        self.assertTrue(
            torch.allclose(grad_custom, grad_ln, atol=1e-6, rtol=1e-6),
            msg=(
                "full feedback gradient should match nn.LayerNorm backward; "
                f"max abs diff={(grad_custom - grad_ln).abs().max().item():.3e}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
