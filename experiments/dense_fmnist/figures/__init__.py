"""Figure generation for the inhibitory-normalization paper.

Run all panels with::

    python -m experiments.dense_fmnist.figures.make_figures --all \
        --entity YOUR_ENTITY --project Luminosity_LNHomeostasis

Run summaries are downloaded once from the wandb public API and cached on disk,
so re-plotting is offline and fast.
"""
