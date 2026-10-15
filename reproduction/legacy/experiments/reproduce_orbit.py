"""Phase 0.2 -- reproduce ORBIT point estimates (PI / IPW / AIPW silhouettes).

Faithful reproduction of ``top-causal-effect-main/ORBIT/main.ipynb`` using the
``tcda_uq`` library. Because we regenerate orbits with numpy (not the paper's
torch ``data.pt``), curves are not bit-identical, but the qualitative result
reproduces: AIPW tracks the true causal-effect silhouette with the smallest bias.

Run (from repo root, in the venv):
    .venv/bin/python experiments/reproduce_orbit.py --n 300 --num-orbits 300 --iters 10
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tcda_uq.datasets import make_orbit_causal, gen_covariate, gen_trt_prob
from tcda_uq.estimators import cross_fit
from tcda_uq.utils import numerical_integration

MU1 = np.array([1, 0.6, -0.7, 2.2, -1.0])
MU2 = np.array([0.4, -0.4, -0.6, 3.3, 3.0])
BETA = np.array([-0.5, -0.1, 0.6, 0.1, 0.1])
SIGMA = np.eye(5) * 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of units")
    ap.add_argument("--num-orbits", type=int, default=300, help="orbits generated per label")
    ap.add_argument("--iters", type=int, default=10, help="treatment redraws")
    ap.add_argument("--n-basis", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/figures/orbit_reproduction.png")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- fixed potential-outcome silhouettes (the topological part) ------------
    print(f"Generating ORBIT potential outcomes (n={args.n}) ...")
    causal = make_orbit_causal(
        n=args.n, num_pts=1000, interval=(0, 0.2), r=3, resolution=100, rng=rng
    )
    potential_outcomes = causal["potential_outcomes"]   # [n, 2, hom, res]
    true_effect = causal["true_effect"]                 # [hom, res]
    tseq = causal["tseq"]
    n_hom = true_effect.shape[0]

    # --- estimation over treatment redraws ------------------------------------
    est = {k: [[] for _ in range(n_hom)] for k in ("pi", "ipw", "aipw")}
    for _ in range(args.iters):
        _, _, X = gen_covariate(MU1, MU2, SIGMA, args.n, rng=rng)
        _, A = gen_trt_prob(X, BETA, rng=rng)
        phi = potential_outcomes[np.arange(args.n), A]
        res = cross_fit((phi, A, X), tseq, n_basis=args.n_basis, n_splits=2, random_state=0)
        for d in range(n_hom):
            est["pi"][d].append(res.plugin[d])
            est["ipw"][d].append(res.ipw[d])
            est["aipw"][d].append(res.aipw[d])

    # --- report L1 error + estimator std --------------------------------------
    print(f"\nORBIT reproduction ({args.iters} reps, n={args.n})")
    print(f"{'dim':>4} {'method':>6} {'L1_err':>10} {'std':>10}")
    avg = {k: [None] * n_hom for k in est}
    for d in range(n_hom):
        for k in ("pi", "ipw", "aipw"):
            curves = np.array(est[k][d])
            avg[k][d] = curves.mean(0)
            l1 = numerical_integration(np.abs(true_effect[d] - avg[k][d]), tseq)
            sd = np.sqrt(np.cov(curves, rowvar=False).mean())
            print(f"{d:>4} {k.upper():>6} {l1:>10.5f} {sd:>10.5f}")

    # --- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, n_hom, figsize=(7 * n_hom, 5))
    axes = np.atleast_1d(axes)
    for d in range(n_hom):
        ax = axes[d]
        ax.plot(tseq, true_effect[d], label="True", lw=4)
        ax.plot(tseq, avg["pi"][d], label="PI", ls="-.", lw=2)
        ax.plot(tseq, avg["ipw"][d], label="IPW", ls=":", lw=2)
        ax.plot(tseq, avg["aipw"][d], label="AIPW", ls="--", lw=2)
        ax.set_title(f"Silhouette effect $H_{d}$")
        ax.legend()
    fig.tight_layout()

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"\nSaved figure -> {args.out}")


if __name__ == "__main__":
    main()
