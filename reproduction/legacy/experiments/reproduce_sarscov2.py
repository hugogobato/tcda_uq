"""Phase 0.3 -- reproduce the SARS-CoV-2 CT notebook (H0 cubical silhouettes).

Confirms the silhouette + nuisance pipeline on the real CT modality. The dataset
(~2.5k CT images) is pulled via kagglehub on first run and cached locally; heavy
real-data experiments are otherwise deferred to Phase 7.

Requires the ``[data]`` extra:  uv pip install -e ".[data]"
Run:  .venv/bin/python experiments/reproduce_sarscov2.py
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tcda_uq.datasets.sarscov2 import (
    download_ct,
    load_ct_images,
    images_to_silhouettes,
    make_sarscov2_causal,
)
from tcda_uq.datasets import gen_covariate, gen_trt_prob
from tcda_uq.estimators import cross_fit
from tcda_uq.utils import numerical_integration

MU1 = np.array([1, 0.6, -0.7, 2.2, -1.0])
MU2 = np.array([0.4, -0.4, -0.6, 3.3, 3.0])
BETA = np.array([-0.5, -0.1, 0.6, 0.1, 0.1])
SIGMA = np.eye(5) * 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--p", type=float, default=0.75)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cache", default="./experiments/data/sarscov2")
    ap.add_argument("--out", default="experiments/figures/sarscov2_reproduction.png")
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    print("Downloading / locating SARS-CoV-2 CT dataset (kagglehub) ...")
    root = download_ct(cache_dir=args.cache)
    infected, noninfected = load_ct_images(root)
    print(f"  {len(infected)} COVID, {len(noninfected)} non-COVID images")

    print("Computing H0 cubical silhouettes ...")
    phi_inf = images_to_silhouettes(infected, interval=(0, 1), r=0.1, resolution=100)
    phi_non = images_to_silhouettes(noninfected, interval=(0, 1), r=0.1, resolution=100)

    causal = make_sarscov2_causal(phi_inf, phi_non, n=args.n, p=args.p, rng=rng)
    po, true_effect = causal["potential_outcomes"], causal["true_effect"]
    tseq = np.linspace(0, 1, 100)

    aipw_reps = []
    for _ in range(args.iters):
        _, _, X = gen_covariate(MU1, MU2, SIGMA, args.n, rng=rng)
        _, A = gen_trt_prob(X, BETA, rng=rng)
        phi = po[np.arange(args.n), A]
        res = cross_fit((phi, A, X), tseq, n_basis=3, n_splits=2, random_state=0)
        aipw_reps.append(res.aipw[0])
    aipw = np.mean(aipw_reps, axis=0)
    print(f"AIPW H0 L1 error vs true effect: "
          f"{numerical_integration(np.abs(true_effect[0] - aipw), tseq):.5f}")

    import os

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tseq, true_effect[0], label="True", lw=4)
    ax.plot(tseq, aipw, label="AIPW", ls="--", lw=2)
    ax.set_title("SARS-CoV-2 CT: silhouette effect $H_0$")
    ax.legend()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Saved figure -> {args.out}")


if __name__ == "__main__":
    main()
