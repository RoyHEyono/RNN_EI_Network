# NeuroGym batch runs and W&B sweeps

These scripts launch NeuroGym experiments on SLURM and log every run to
[Weights & Biases](https://wandb.ai/).

Run the commands below from this directory unless a command explicitly changes
to the repository root.

## Setup

From the repository root, install the NeuroGym dependencies and authenticate
with W&B:

```bash
uv sync --extra neurogym
uv run wandb login
```

Then prepare the SLURM log directories:

```bash
cd experiments/neurogym/batch_run
mkdir -p sbatch_out sbatch_err
```

## Fixed batch run

The fixed batch run compares six model configurations at ten learning rates:

- EI RNN
- EI RNN with parametrized LayerNorm
- vanilla RNN with and without LayerNorm
- LSTM with and without LayerNorm

All 60 runs use seed 42. Training uses SGD with the default momentum of zero.

First generate `random_configs.json`:

```bash
uv run --extra neurogym python generate_configs.py
```

Then submit the 60-element SLURM array:

```bash
sbatch batch_run.sh
```

`DelayMatchSample-v0` is the default task. Override the task and W&B project at
submission time if needed:

```bash
sbatch \
  --export=ALL,TASK=PerceptualDecisionMaking-v0,WANDB_PROJECT=ng-perceptual-decision \
  batch_run.sh
```

The SLURM logs are written to `sbatch_out/` and `sbatch_err/`. Runs appear in
the W&B project named by `WANDB_PROJECT`; without an override, the project name
is derived from the task as `ng-<task-name>`.

## W&B learning-rate sweep

The native W&B sweep is configured in `../sweep.yaml`. It runs an EI RNN with
ParametrizedLayerNorm on `DelayMatchSample-v0` and independently tunes:

- `lr`, `lr_ie`, `lr_ei` — task network (excitatory RNN + EI readout), log-uniform:
  `lr` 0.001–0.1, `lr_ie` 1e-5–0.01, `lr_ei` 0.01–1.0
- `lr_norm_mean`, `lr_norm_var0`, `lr_norm_var2` — ParametrizedLayerNorm, log-uniform
  0.001–0.5 each

The optimizer is fixed to SGD, momentum is fixed to zero, and Bayesian search maximizes
`eval/trial_accuracy_auc`. Hyperband early termination (`early_terminate` in
`sweep.yaml`) can stop weak runs before the full 5000 steps; the training loop
honours that via `wandb.run.should_stop()`.

Create a new sweep from the repository root:

```bash
cd ../../..
uv run --extra neurogym wandb sweep experiments/neurogym/sweep.yaml
```

W&B prints a sweep path such as:

```text
royeyono/ng-delaymatchsample-v0/abcdefgh
```

Return to this directory and submit the sweep agents using that complete path:

```bash
cd experiments/neurogym/batch_run
sbatch \
  --export=ALL,SWEEP_ID=royeyono/ng-delaymatchsample-v0/abcdefgh \
  sweep_agent.sh
```

`sweep_agent.sh` submits 60 trials and permits at most five concurrent trials,
allowing the Bayesian optimizer to incorporate completed results. Each array
task requests one W&B trial.

The `wandb sweep` output also prints the online sweep URL. Its general form is:

```text
https://wandb.ai/<entity>/<project>/sweeps/<sweep-id>
```

## Monitoring

Check either SLURM array with:

```bash
squeue -j <slurm-job-id>
```

Cancel an array if necessary:

```bash
scancel <slurm-job-id>
```

Changing `sweep.yaml` does not alter an existing W&B sweep. Run `wandb sweep`
again after editing it, then submit agents with the newly printed sweep ID.
