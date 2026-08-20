"""Named experimental conditions as wandb-config filters.

One place to answer "which runs are the I-Norm condition?", so the panel code
reads like the paper rather than like a query language.
"""

BASE = {"dataset": "fashionmnist", "use_testset": True}

#: The default weight of the I-Norm loss used everywhere except the lambda sweep.
DEFAULT_LAMBDA = 0.01


def _with(eps, **extra):
    out = dict(BASE)
    if eps is not None:
        out["brightness_factor"] = eps
    out.update(extra)
    return out


# --- Figure 2: hard-coded normalization, no homeostatic loss -----------------

def ei_no_ln(eps):
    """Standard E/I network, no normalization."""
    return _with(eps, homeostasis=0, excitation_training=0, layer_norm=0,
                 normtype=0, normtype_detach=0)


def ei_ln(eps):
    """E/I network with hard-coded LayerNorm in both passes (LN+)."""
    return _with(eps, homeostasis=0, excitation_training=0, layer_norm=1,
                 normtype=0, normtype_detach=0)


def ei_ln_minus(eps):
    """LN in the forward pass only; the error signal bypasses it (LN-)."""
    return _with(eps, homeostasis=0, excitation_training=0, layer_norm=1,
                 normtype=0, normtype_detach=1)


def e_only_ln(eps):
    """Excitatory-only network with LN+ -- the inhibition-ablated control."""
    return _with(eps, homeostasis=0, excitation_training=1, layer_norm=1,
                 normtype=0, normtype_detach=0)


# --- Figures 3-7: inhibition trained to normalize ----------------------------

def inorm(eps, lam=DEFAULT_LAMBDA):
    """Full I-Norm (subtractive + divisive), no GradNorm."""
    return _with(eps, homeostasis=1, shunting=1, normtype_detach=1,
                 ln_feedback="full", lambda_homeo_var=lam)


def inorm_subtractive(eps, lam=DEFAULT_LAMBDA):
    """I-Norm with the divisive pathway removed (Fig. 3b's "I-Norm (sub)")."""
    return _with(eps, homeostasis=1, shunting=0, normtype_detach=1,
                 lambda_homeo_var=lam)


def inorm_gradnorm(eps, ln_feedback="full", lam=DEFAULT_LAMBDA):
    """I-Norm whose backward pass is shaped by (part of) the LN gradient."""
    return _with(eps, homeostasis=1, shunting=1, normtype_detach=0,
                 ln_feedback=ln_feedback, lambda_homeo_var=lam)


def inorm_lateral_inhibition(eps, lam=DEFAULT_LAMBDA):
    """Gradient centering by a fixed random inhibitory pool (Fig. 7)."""
    return inorm_gradnorm(eps, ln_feedback="fa_center", lam=lam)
