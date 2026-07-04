#!/usr/bin/env python
"""Build the executable tutorial notebooks shipped with ``tcda_uq``.

The notebooks are generated from compact Python cell sources so the JSON stays
reproducible and easy to refresh after API changes.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


BOOTSTRAP = r"""
from pathlib import Path
import sys

cwd = Path.cwd().resolve()
repo = cwd.parent if cwd.name == "notebooks" else cwd
if repo.name == "tcda_uq":
    sys.path.insert(0, str(repo.parent))
elif (repo / "tcda_uq").exists():
    sys.path.insert(0, str(repo))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
""".strip()


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write_notebook(path: Path, title: str, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        }
    )
    nb.cells = [md(f"# {title}"), code(BOOTSTRAP), *cells]
    nbf.write(nb, path)


def tate_notebook():
    return [
        md(
            """
            This notebook builds TATE confidence bands on the tri-oracle
            simulation harness and evaluates them against the known marginal
            truth. It uses small CPU settings for a tutorial smoke run.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.linear_model import LogisticRegression

            from tcda_uq import evaluate_coverage, tate_band
            from tcda_uq.datasets import TriOracleSimulation
            from tcda_uq.estimators import cross_fit
            from tcda_uq.metrics import interval_wise_error
            """
        ),
        code(
            """
            sim = TriOracleSimulation(n_hom_dim=1, resolution=28, n_basis=5, noise_scale=0.3, seed=0)
            sample = sim.sample(220, rng=1)
            fit = cross_fit(
                sample.observed,
                sim.tseq,
                n_basis=5,
                n_splits=2,
                random_state=0,
                propensity_estimator=LogisticRegression(max_iter=500),
            )
            truth = sim.true_tate()[0]
            """
        ),
        code(
            """
            specs = {
                "mbb": {"n_boot": 120, "rng": 2},
                "pv": {"n_boot": 120, "rng": 2},
                "lr": {"backend": "python"},
            }

            bands = {name: tate_band(fit, method=name, d=0, alpha=0.10, **kw) for name, kw in specs.items()}
            rows = []
            for name, band in bands.items():
                result = evaluate_coverage(band, truth)
                metric = interval_wise_error(band.lower, band.upper, truth) if name == "pv" else result.coverage
                rows.append(
                    {
                        "method": name,
                        "reported_metric": "interval-wise error" if name == "pv" else "simultaneous coverage",
                        "value": metric,
                        "mean_width": band.mean_width(),
                    }
                )
            pd.DataFrame(rows)
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(sim.tseq, truth, color="black", lw=2, label="true TATE")
            for name, band in bands.items():
                ax.plot(sim.tseq, band.center, lw=1, label=f"{name} center")
                ax.fill_between(sim.tseq, band.lower, band.upper, alpha=0.12)
            ax.set(xlabel="t", ylabel="silhouette effect", title="TATE confidence bands")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            """
        ),
    ]


def ctate_notebook():
    return [
        md(
            """
            CTATE bands target the conditional mean curve at a fixed covariate
            value. A prediction band at the same covariate value is wider because
            it covers an individual draw rather than the conditional mean.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import pandas as pd
            from sklearn.linear_model import LogisticRegression

            from tcda_uq import ctate_band, evaluate_coverage, itte_band
            from tcda_uq.datasets import TriOracleSimulation
            from tcda_uq.estimators import CTATEDRLearner
            from tcda_uq.uq.conformal import ITTEConformal, make_weight_fn
            """
        ),
        code(
            """
            sim = TriOracleSimulation(n_hom_dim=1, resolution=28, n_basis=5, noise_scale=0.3, seed=4)
            sample = sim.sample(260, rng=5)
            x = sim.EX + 0.25

            learner = CTATEDRLearner(n_basis=5).fit(
                sample.observed,
                sim.tseq,
                n_splits=2,
                random_state=6,
                propensity_estimator=LogisticRegression(max_iter=500),
            )
            conf = ctate_band(learner, x, d=0, alpha=0.10, n_boot=120, rng=7)
            truth = sim.true_ctate(x[None, :])[0, 0]

            predictor = ITTEConformal.fit(
                sample.observed,
                sim.tseq,
                n_basis=5,
                weight_fn=make_weight_fn("overlap"),
                random_state=8,
                propensity_estimator=LogisticRegression(max_iter=500),
            )
            pred = itte_band(predictor, x, d=0, alpha=0.10)
            """
        ),
        code(
            """
            pd.DataFrame(
                [
                    {
                        "object": "CTATE confidence band",
                        "kind": conf.kind,
                        "covers_mean_truth": evaluate_coverage(conf, truth).coverage,
                        "mean_width": conf.mean_width(),
                    },
                    {
                        "object": "ITTE prediction band at same x",
                        "kind": pred.kind,
                        "covers_mean_truth": evaluate_coverage(pred, truth).coverage,
                        "mean_width": pred.mean_width(),
                    },
                ]
            )
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(sim.tseq, truth, color="black", lw=2, label="true CTATE mean")
            ax.fill_between(sim.tseq, conf.lower, conf.upper, alpha=0.25, label="confidence")
            ax.fill_between(sim.tseq, pred.lower, pred.upper, alpha=0.18, label="prediction")
            ax.plot(sim.tseq, conf.center, lw=1.5, label="CTATE estimate")
            ax.set(xlabel="t", ylabel="effect", title="Confidence vs prediction at fixed x")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            """
        ),
    ]


def itte_notebook():
    return [
        md(
            """
            This notebook compares naive inverse-propensity conformal weighting
            with overlap-stabilized weighting under weak overlap.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.linear_model import LogisticRegression

            from tcda_uq import evaluate_coverage
            from tcda_uq.datasets import TriOracleSimulation
            from tcda_uq.uq.conformal import ITTEConformal, make_weight_fn
            """
        ),
        code(
            """
            sim = TriOracleSimulation(
                n_hom_dim=1,
                resolution=25,
                n_basis=5,
                noise_scale=0.3,
                prop_scale=3.0,
                seed=10,
            )
            train = sim.sample(360, rng=11)
            test = sim.sample(160, rng=12)

            models = {
                "naive": ITTEConformal.fit(
                    train.observed,
                    sim.tseq,
                    n_basis=5,
                    weight_fn=make_weight_fn("naive"),
                    random_state=13,
                    propensity_estimator=LogisticRegression(max_iter=500),
                ),
                "overlap": ITTEConformal.fit(
                    train.observed,
                    sim.tseq,
                    n_basis=5,
                    weight_fn=make_weight_fn("overlap"),
                    random_state=13,
                    propensity_estimator=LogisticRegression(max_iter=500),
                ),
            }
            """
        ),
        code(
            """
            rows = []
            target = test.oracle_itte[:, 0, :]
            for name, model in models.items():
                lo, hi, center = model.band_bounds(test.X, alpha=0.10, d=0)
                res = evaluate_coverage((lo, hi), target, level=0.90)
                finite = np.isfinite(hi - lo).all(axis=1)
                rows.append(
                    {
                        "weighting": name,
                        "coverage": res.coverage,
                        "finite_band_share": finite.mean(),
                        "mean_width": res.mean_width,
                    }
                )
            pd.DataFrame(rows)
            """
        ),
        code(
            """
            plot_idx = [0, 1, 2, 3]
            lo, hi, center = models["overlap"].band_bounds(test.X[plot_idx], alpha=0.10, d=0)
            true_itte = test.oracle_itte[plot_idx, 0, :]

            fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
            for ax, idx, lower, upper, ctr, truth in zip(
                axes.ravel(), plot_idx, lo, hi, center, true_itte
            ):
                ax.fill_between(sim.tseq, lower, upper, color="tab:blue", alpha=0.18, label="90% band")
                ax.plot(sim.tseq, ctr, color="tab:blue", lw=1.8, label="estimated ITTE")
                ax.plot(sim.tseq, truth, color="black", lw=1.5, linestyle="--", label="true ITTE")
                ax.set_title(f"held-out unit {idx}")
                ax.axhline(0.0, color="0.75", lw=0.8)

            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
            fig.supxlabel("t")
            fig.supylabel("ITTE curve")
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            """
        ),
    ]


def orbit_notebook():
    return [
        md(
            """
            This example runs the actual ORBIT point-cloud to persistence to
            silhouette pipeline, then treats the generated counterfactual
            silhouettes as potential topological outcomes for a small TATE band.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.linear_model import LogisticRegression

            from tcda_uq import evaluate_coverage, tate_band
            from tcda_uq.datasets import make_orbit_causal
            from tcda_uq.estimators import cross_fit
            """
        ),
        code(
            """
            rng = np.random.default_rng(20)
            orbit = make_orbit_causal(
                n=24,
                p_reverse=0.25,
                interval=(0.0, 0.25),
                resolution=22,
                homology_dims=(0, 1),
                num_pts=40,
                rng=21,
            )
            po = orbit["potential_outcomes"]
            n = po.shape[0]
            X = rng.normal(size=(n, 3))
            A = np.array([0, 1] * (n // 2))
            rng.shuffle(A)
            phi = po[np.arange(n), A]
            sample = (phi, A, X)
            """
        ),
        code(
            """
            fit = cross_fit(
                sample,
                orbit["tseq"],
                n_basis=4,
                n_splits=2,
                random_state=22,
                propensity_estimator=LogisticRegression(max_iter=500),
            )
            band = tate_band(fit, method="mbb", d=1, alpha=0.10, n_boot=80, rng=23)
            truth = orbit["true_effect"][1]
            pd.DataFrame(
                [
                    {
                        "dataset": "ORBIT synthetic point clouds",
                        "homology_dim": 1,
                        "coverage_against_sample_truth": evaluate_coverage(band, truth).coverage,
                        "mean_width": band.mean_width(),
                    }
                ]
            )
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(orbit["tseq"], truth, color="black", lw=2, label="sample oracle TATE")
            ax.fill_between(orbit["tseq"], band.lower, band.upper, alpha=0.25, label="mbb band")
            ax.plot(orbit["tseq"], band.center, lw=1.5, label="AIPW estimate")
            ax.set(xlabel="filtration value", ylabel="H1 silhouette effect", title="ORBIT TATE band")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            """
        ),
    ]


def sars_notebook():
    return [
        md(
            """
            The real SARS-CoV-2 CT loader needs the optional ``[data]`` extra and
            local dataset files. This executable tutorial uses ``TCDA_UQ_SARS_ROOT``
            when available, otherwise it falls back to tiny synthetic CT-like
            images so the image-to-silhouette and causal-UQ code path still runs.
            """
        ),
        code(
            """
            import os

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.linear_model import LogisticRegression

            from tcda_uq import evaluate_coverage, tate_band
            from tcda_uq.datasets.sarscov2 import images_to_silhouettes, load_ct_images, make_sarscov2_causal
            from tcda_uq.estimators import cross_fit
            """
        ),
        code(
            """
            root = os.environ.get("TCDA_UQ_SARS_ROOT")
            if root:
                infected, noninfected = load_ct_images(root)
                infected = infected[:12]
                noninfected = noninfected[:12]
                source = "local SARS-CoV-2 CT files"
            else:
                rng = np.random.default_rng(30)
                grid = np.linspace(-1.0, 1.0, 18)
                xx, yy = np.meshgrid(grid, grid)

                def blob(offset, scale):
                    base = np.exp(-((xx - offset) ** 2 + yy**2) / scale)
                    return np.clip(base + 0.08 * rng.normal(size=base.shape), 0.0, 1.0)

                infected = [blob(-0.25, 0.22) for _ in range(14)]
                noninfected = [blob(0.25, 0.35) for _ in range(14)]
                source = "synthetic CT-like images"

            phi_inf = images_to_silhouettes(infected, resolution=24, workers=1)
            phi_non = images_to_silhouettes(noninfected, resolution=24, workers=1)
            causal = make_sarscov2_causal(phi_inf, phi_non, n=20, p=0.70, rng=31)
            """
        ),
        code(
            """
            rng = np.random.default_rng(32)
            po = causal["potential_outcomes"]
            n = po.shape[0]
            X = rng.normal(size=(n, 2))
            A = np.array([0, 1] * (n // 2))
            rng.shuffle(A)
            phi = po[np.arange(n), A]
            tseq = np.linspace(0.0, 1.0, po.shape[-1])

            fit = cross_fit(
                (phi, A, X),
                tseq,
                n_basis=4,
                n_splits=2,
                random_state=33,
                propensity_estimator=LogisticRegression(max_iter=500),
            )
            band = tate_band(fit, method="mbb", d=0, alpha=0.10, n_boot=80, rng=34)
            truth = causal["true_effect"][0]

            pd.DataFrame(
                [
                    {
                        "source": source,
                        "n_images_used": len(infected) + len(noninfected),
                        "coverage_against_constructed_truth": evaluate_coverage(band, truth).coverage,
                        "mean_width": band.mean_width(),
                    }
                ]
            )
            """
        ),
        code(
            """
            fig, ax = plt.subplots(1, 2, figsize=(8, 3))
            ax[0].imshow(infected[0], cmap="gray")
            ax[0].set_title("control image")
            ax[0].axis("off")
            ax[1].plot(tseq, truth, color="black", lw=2, label="constructed TATE")
            ax[1].fill_between(tseq, band.lower, band.upper, alpha=0.25, label="mbb band")
            ax[1].legend(loc="best", fontsize=8)
            ax[1].set_title("H0 cubical silhouette effect")
            fig.tight_layout()
            """
        ),
    ]


def main():
    NOTEBOOKS.mkdir(exist_ok=True)
    specs = [
        ("01_tate_confidence_bands.ipynb", "TATE Confidence Bands", tate_notebook()),
        ("02_ctate_bridge.ipynb", "CTATE Bridge", ctate_notebook()),
        ("03_itte_conformal.ipynb", "ITTE Conformal Prediction", itte_notebook()),
        ("04_orbit_pipeline.ipynb", "ORBIT Pipeline", orbit_notebook()),
        ("05_sarscov2_demo.ipynb", "SARS-CoV-2 CT Demo", sars_notebook()),
    ]
    for filename, title, cells in specs:
        write_notebook(NOTEBOOKS / filename, title, cells)
        print(f"wrote notebooks/{filename}")


if __name__ == "__main__":
    main()
