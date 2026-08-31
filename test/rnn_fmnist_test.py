import unittest

import torch

from experiments.rnn_fmnist.cli import build_train_arg_parser
from experiments.rnn_fmnist.training import evaluate
from inhibition.data import RandomAdjustBrightness, RandomAdjustContrast
from inhibition.model import RNNNet, inorm_param_groups, param_ln_param_groups


class TestRNNFashionMNISTModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.x = torch.rand(3, 1, 28, 28)

    def test_standard_layer_norm_forward(self):
        model = RNNNet(hidden_size=16)
        logits = model(self.x)
        self.assertEqual(logits.shape, (3, 10))
        self.assertIsNone(model.last_aux_loss)

    def test_parametrized_layer_norm_forward_and_backward(self):
        model = RNNNet(hidden_size=16, use_parametrized_layer_norm=True)
        logits = model(self.x)
        self.assertEqual(logits.shape, (3, 10))
        self.assertIsNotNone(model.last_aux_loss)
        self.assertTrue(torch.isfinite(model.last_aux_loss))

        (logits.square().mean() + model.last_aux_loss).backward()
        norm = model.rnn.layer_norm
        self.assertTrue(
            all(parameter.grad is not None for parameter in norm.parameters())
        )

    def test_main_and_norm_optimizer_groups_are_disjoint_and_complete(self):
        model = RNNNet(hidden_size=8, use_parametrized_layer_norm=True)
        main_groups = inorm_param_groups(model, 0.1, 0.2, 0.3)
        norm_groups = param_ln_param_groups(model.rnn.layer_norm, 0.4, 0.5, 0.6)
        main_ids = {id(p) for group in main_groups for p in group["params"]}
        norm_ids = {id(p) for group in norm_groups for p in group["params"]}

        self.assertFalse(main_ids & norm_ids)
        self.assertEqual(
            main_ids | norm_ids, {id(parameter) for parameter in model.parameters()}
        )


class TestRNNFashionMNISTConfiguration(unittest.TestCase):
    def test_sweep_aliases_and_task_selection(self):
        args = build_train_arg_parser().parse_args(
            [
                "--normalization=paramln",
                "--dataset=fashionmnist_contrast",
                "--epsilon=0.75",
                "--lr_ie=0.002",
                "--lr_ei=0.2",
                "--lr_norm_mean=0.00001",
            ]
        )
        self.assertEqual(args.normalization, "paramln")
        self.assertEqual(args.dataset, "fashionmnist_contrast")
        self.assertEqual(args.epsilon, 0.75)
        self.assertEqual(args.lr_ie, 0.002)
        self.assertEqual(args.lr_ei, 0.2)
        self.assertEqual(args.lr_norm_mean, 0.00001)

    def test_fixed_brightness_and_contrast_have_dense_task_semantics(self):
        x = torch.tensor([[[0.0, 0.5], [0.5, 1.0]]])
        bright = RandomAdjustBrightness(0.25, fixed=True)(x)
        self.assertTrue(
            torch.equal(
                bright, torch.tensor([[[0.25, 0.75], [0.75, 1.0]]])
            )
        )

        contrast = RandomAdjustContrast(0.5, fixed=True)(x)
        expected = torch.clamp((x - x.mean()) * 1.5 + x.mean(), 0.0, 1.0)
        self.assertTrue(torch.equal(contrast, expected))


class TestRNNFashionMNISTEvaluation(unittest.TestCase):
    def test_joint_loss_combines_held_out_task_and_auxiliary_losses(self):
        class ModelWithAuxLoss(torch.nn.Module):
            def forward(self, x):
                self.last_aux_loss = torch.tensor(0.25, device=x.device)
                return torch.tensor([[2.0, 0.0]], device=x.device).repeat(
                    x.shape[0], 1
                )

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.zeros(3, 1), torch.zeros(3, dtype=torch.long)
            ),
            batch_size=2,
        )
        metrics = evaluate(
            ModelWithAuxLoss(),
            torch.device("cpu"),
            loader,
            aux_loss_weight=2.0,
        )

        self.assertAlmostEqual(metrics["aux_loss"], 0.25)
        self.assertAlmostEqual(
            metrics["joint_loss"], metrics["task_loss"] + 0.5
        )


if __name__ == "__main__":
    unittest.main()
