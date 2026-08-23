# Dense Fashion-MNIST: reproducing the I-Norm paper figures

Reproduction path for **"Inhibitory normalization of error signals improves learning in
neural circuits"** ([arXiv:2603.17676](https://arxiv.org/abs/2603.17676)), rewritten from
[`HomeostaticDANN/models/dense_mnist_task`](https://github.com/RoyHEyono/HomeostaticDANN).

The task is Fashion-MNIST with a per-image luminance shift `Δ ~ Uniform(−ε, +ε)` on `[0, 1]`
pixels, clamped back into range. `ε ∈ {0, 0.25, 0.5, 0.75}` sets how much invariance the
network needs.

## Conditions

Networks have two hidden layers (`fc0`, `fc1`) plus an E/I readout, an inhibitory population
10% the size of the excitatory one, and Dale's principle enforced by clamping.

| Condition | Entry point | Flags |
|---|---|---|
| E/I, no norm | `train_ei_network` | `--model.excitation_training=0 --model.layer_norm=0` |
| E/I + LN⁺ | `train_ei_network` | `--model.excitation_training=0 --model.layer_norm=1 --model.normtype_detach=0` |
| E/I + LN⁻ (forward only) | `train_ei_network` | `... --model.normtype_detach=1` |
| E-only + LN⁺ | `train_ei_network` | `--model.excitation_training=1 --model.layer_norm=1` |
| I-Norm (subtractive only) | `train_inorm_network` | `--model.shunting=0 --model.normtype_detach=1` |
| I-Norm | `train_inorm_network` | `--model.shunting=1 --model.normtype_detach=1` |
| I-Norm + GradNorm | `train_inorm_network` | `--model.normtype_detach=0 --model.ln_feedback=full` |
| GradNorm components | `train_inorm_network` | `--model.ln_feedback={scale,center,decorrelate}` |
| Lateral inhibition | `train_inorm_network` | `--model.ln_feedback=fa_center` |

Two flags carry most of the meaning:

* **`--model.normtype_detach`** routes the error signal *around* the normalization. It is what
  separates LN⁺ from LN⁻, and plain I-Norm (Fig. 4) from I-Norm + GradNorm (Fig. 5).
* **`--model.ln_feedback`** picks which term of the LayerNorm Jacobian GradNorm imposes
  (Fig. 6); `fa_center` replaces exact mean subtraction with pooling through fixed, random,
  positive synapses (Fig. 7).

Excitatory weights are trained on cross-entropy; inhibitory weights on the local I-Norm loss
`E[μ²  + (σ²−1)²]`, weighted by `--opt.lambda_homeo_var`. The two are kept apart by
stop-gradients inside `INormLayer`, not by separate backward passes.

## Running

Single run:

```bash
python -m experiments.dense_fmnist.train_inorm_network \
  --data.brightness_factor=0.75 --model.normtype_detach=0 --model.ln_feedback=center \
  --train.epochs=50 --opt.lr=0.005 --opt.inhib_lrs.wix=0.1 --opt.inhib_lrs.wei=0.001
```

Add `--exp.dry_run` for a five-batch smoke test, and `--model.track_alignment=0` to skip the
cosine-similarity measurement (which roughly doubles step time and is only needed for the
alignment panels).

Full sweeps on SLURM. Each `batch_*` script is a grid over conditions; every grid point
`sbatch`es an inner array of 30 random hyperparameter configurations:

```bash
cd experiments/dense_fmnist/batch_scripts
python generate_random_params.py
export REPO_ROOT=$HOME/RNN_EI_Network WANDB_ENTITY=your_entity
sbatch batch_ei_non_homeostasis_runs.sh   # Figure 2
sbatch batch_inorm_runs.sh                # Figures 3-6
sbatch batch_inorm_subtractive_runs.sh    # Figure 3b's I-Norm (sub)
sbatch batch_inorm_fa_runs.sh             # Figure 7
sbatch batch_inorm_lambda_runs.sh         # lambda sensitivity
```

The inner scripts also run standalone, which is the quickest way to check a sweep arm before
committing the array:

```bash
REPO_ROOT=$PWD USE_WANDB=0 EPOCHS=1 BRIGHTNESS_FACTOR=0.75 NORMTYPE_DETACH=0 \
  LN_FEEDBACK=full SHUNTING=1 LAMBDA_HOMEOS=0.01 SLURM_ARRAY_TASK_ID=0 \
  bash experiments/dense_fmnist/batch_scripts/run_inorm_network.sh
```

Budget: ~2760 runs at roughly 20 min each on an rtx8000.

## Figures

Runs are read back directly from the wandb public API from within each notebook.

Figure generation is notebook-only. Use the per-figure notebooks in:

- `experiments/dense_fmnist/analysis/figure1b_stimulus.ipynb`
- `experiments/dense_fmnist/analysis/figure2.ipynb`
- `experiments/dense_fmnist/analysis/figure3.ipynb`
- `experiments/dense_fmnist/analysis/figure4.ipynb`
- `experiments/dense_fmnist/analysis/figure5.ipynb`
- `experiments/dense_fmnist/analysis/figure6.ipynb`
- `experiments/dense_fmnist/analysis/figure7.ipynb`

Each notebook exposes selection, pairing, and plotting steps, and writes outputs to
`experiments/dense_fmnist/figures_out_notebooks/` by default. Figures 1, and the schematic
panels of 3, 5, 6 and 7, are hand-drawn and not generated here.

If you want paper-like typography, set your preferred sans-serif font directly in each notebook's
styling cell; otherwise the default sans-serif is used.

## Differences from the original implementation

* Input pixels stay in `[0, 1]` (no mean/std standardization), so `ε` means what the paper says.
* Plain `CrossEntropyLoss`, no label smoothing; the train loader shuffles.
* `W_EI` and `U_EI` — the fixed averaging weights of the theory — are frozen by default
  (`--model.freeze_ei`), as in the reference; `--opt.inhib_lrs.wei` then only serves to label
  and pair runs.
* The I-Norm loss is added to the task loss and backpropagated once, rather than being
  backpropagated from inside `forward()`. Gradient isolation comes from the stop-gradients,
  so the two are equivalent up to the `λ` scaling.
* `train_loss` / `train_acc` are running averages over the training epoch rather than a
  second pass over the training set.
* numpy's global RNG is seeded from `--train.seed` (`inhibition/init.py` draws weights from
  it), so a run is reproducible from its seed.
