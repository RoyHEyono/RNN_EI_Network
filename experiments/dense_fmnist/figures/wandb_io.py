"""Download and cache run summaries from the wandb public API.

Every paper panel is a function of a run's *config* and its *summary* metrics,
so one pass over the project is enough. The result is cached as JSON; panels
then filter it in memory, which keeps re-plotting offline and fast.
"""

from __future__ import annotations

import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "_cache"

#: Summary metrics the panels read. Missing keys are simply absent from a record.
SUMMARY_KEYS = (
    "test_acc",
    "test_loss",
    "test_loss_auc",
    "train_acc",
    "train_loss",
    "train_fc0_mu",
    "train_fc1_mu",
    "train_fc0_var",
    "train_fc1_var",
    "output_alignment_fc0",
    "output_alignment_fc1",
    "gradient_alignment_fc0",
    "gradient_alignment_fc1",
    "eval_fc0_gradient_eigenvalue_max",
    "eval_fc1_gradient_eigenvalue_max",
)


def cache_path(entity: str, project: str) -> Path:
    return CACHE_DIR / f"{entity}__{project}.json"


def download_runs(entity: str, project: str) -> list[dict]:
    """Pull every finished run in the project as ``{"config": ..., "summary": ...}``."""
    import wandb

    api = wandb.Api(timeout=60)
    records = []
    for run in api.runs(f"{entity}/{project}"):
        summary = {}
        for key in SUMMARY_KEYS:
            value = run.summary.get(key)
            if value is not None:
                summary[key] = value
        records.append(
            {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "config": {k: v for k, v in run.config.items() if not k.startswith("_")},
                "summary": summary,
            }
        )
    return records


def load_runs(entity: str, project: str, *, refresh: bool = False) -> list[dict]:
    """Cached :func:`download_runs`. Pass ``refresh=True`` to re-query wandb."""
    path = cache_path(entity, project)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    records = download_runs(entity, project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records))
    print(f"cached {len(records)} runs to {path}")
    return records


def select(runs: list[dict], **conditions) -> list[dict]:
    """Runs whose config matches every ``key=value`` condition.

    ``None`` means "any value"; a tuple or list means "any of these".
    """
    out = []
    for run in runs:
        cfg = run["config"]
        if all(_matches(cfg.get(k), v) for k, v in conditions.items()):
            out.append(run)
    return out


def _matches(actual, expected) -> bool:
    if expected is None:
        return True
    if isinstance(expected, (tuple, list, set)):
        return any(_matches(actual, e) for e in expected)
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(actual) == bool(expected)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) < 1e-12
    return actual == expected


def top_k(runs: list[dict], k: int | None = None, metric: str = "test_acc") -> list[dict]:
    """Runs sorted by ``metric``, best first, optionally truncated to ``k``."""
    ranked = sorted(
        (r for r in runs if r["summary"].get(metric) is not None),
        key=lambda r: r["summary"][metric],
        reverse=True,
    )
    return ranked if k is None else ranked[:k]


#: Hyperparameters that identify "the same configuration" across conditions.
MATCH_KEYS = ("lr", "wd", "inhib_lrs", "momentum", "inhib_momentum")


def _match_signature(config: dict):
    parts = []
    for key in MATCH_KEYS:
        value = config.get(key)
        if isinstance(value, dict):
            value = tuple(sorted(value.items()))
        parts.append(value)
    return tuple(parts)


def pair_by_config(runs_x: list[dict], runs_y: list[dict], metric: str = "test_acc"):
    """Pair runs from two conditions that share a hyperparameter configuration.

    Returns ``[(x_value, y_value), ...]``. Used for the paper's diagonal scatter
    plots, where each point must compare like with like.
    """
    index = {}
    for run in runs_y:
        index.setdefault(_match_signature(run["config"]), run)
    pairs = []
    for run in runs_x:
        partner = index.get(_match_signature(run["config"]))
        if partner is None:
            continue
        xv, yv = run["summary"].get(metric), partner["summary"].get(metric)
        if xv is not None and yv is not None:
            pairs.append((xv, yv))
    return pairs


def layer_mean(run: dict, prefix: str, layers=("fc0", "fc1")) -> float | None:
    """Average a per-layer summary metric (e.g. ``gradient_alignment_``) over layers."""
    values = [run["summary"].get(f"{prefix}{layer}") for layer in layers]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None
