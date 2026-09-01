# RNN Fashion-MNIST invariance experiments

This package compares standard LayerNorm (LN) with ParametrizedLayerNorm
(ParamLN) on the brightness and contrast Fashion-MNIST tasks from
`experiments/dense_fmnist`.

## Experiment matrix

The complete matrix has 16 independent sweep arms:

- normalization: `ln`, `paramln`
- task: `fashionmnist` (brightness), `fashionmnist_contrast` (contrast)
- epsilon: `0`, `0.25`, `0.5`, `0.75`

Images remain in the raw `[0, 1]` range. Brightness adds one image-wide value
sampled from `Uniform(-epsilon, +epsilon)`. Contrast scales deviations around
each image's mean by a value sampled from `Uniform(1-epsilon, 1+epsilon)`.

The model scans the 28 image rows as 28 recurrent timesteps, classifies the
final hidden state, and uses an EI dense readout. The recurrent core itself is
`SimpleEERNN`, which has nonnegative excitatory input and recurrent weights; it
does not contain an inhibitory recurrent population.

## Single runs

From the repository root:

```bash
python -m experiments.rnn_fmnist.main \
  --normalization=ln \
  --dataset=fashionmnist \
  --epsilon=0.75
```

ParamLN uses a separate optimizer for its three parameter groups:

```bash
python -m experiments.rnn_fmnist.main \
  --normalization=paramln \
  --dataset=fashionmnist_contrast \
  --epsilon=0.75 \
  --lr=0.01 --lr-ie=0.001 --lr-ei=0.1 \
  --lr-norm-mean=0.00001 \
  --lr-norm-var0=0.000001
```

Add `--dry-run --no-accel` for a one-batch smoke test. By default, test images
receive random jitter from the same distribution as training images. A
nonzero `--eval-epsilon` instead applies that fixed brightness shift or fixed
contrast change at evaluation.

## W&B sweeps

Both sweeps use Bayesian optimization and maximize `test/accuracy_pct`.
The ranges are copied exactly from `experiments/neurogym/sweep.yaml`:

- `lr`: 0.001 to 0.1
- `lr_ie`: 0.00001 to 0.01
- `lr_ei`: 0.01 to 1.0
- ParamLN only, `lr_norm_mean`: 0.000001 to 0.0001
- ParamLN only, `lr_norm_var0`: 0.00000001 to 0.0001

Generate the 16 concrete YAML files:

```bash
python -m experiments.rnn_fmnist.sweeps.generate_configs
```

Register all of them with W&B:

```bash
bash experiments/rnn_fmnist/batch_run/create_sweeps.sh
```

W&B prints one complete sweep path for each arm. Submit agents for each path:

```bash
mkdir -p experiments/rnn_fmnist/batch_run/{sbatch_out,sbatch_err}
sbatch \
  --array=0-49%10 \
  --export=ALL,SWEEP_ID=<entity/rnn-fmnist-invariance/sweep-id> \
  experiments/rnn_fmnist/batch_run/sweep_agent.sh
```

Each array element claims one trial. Override `--array` to choose the number
of trials and maximum concurrency for each independent arm.
