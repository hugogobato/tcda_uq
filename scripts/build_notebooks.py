#!/usr/bin/env python
"""Build clean tutorial notebooks from the checked-in Python cell sources."""

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
""".strip()


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write_notebook(path: Path, title: str, cells):
    notebook = nbf.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        }
    )
    notebook.cells = [md(f"# {title}"), code(BOOTSTRAP), *cells]
    nbf.write(notebook, path)


def tate_notebook():
    return [
        md(
            """
            This small example estimates a population mean-effect curve and builds
            a Gaussian multiplier confidence band. The fit uses folds randomized
            independently of the observations. The displayed inclusion is on the
            simulation's fixed filtration grid for this one data set; it is not a
            check of the asymptotic assumptions or continuum coverage.

            Interval-wise testing is a separate task. `tate_test(..., method="iwt")`
            returns adjusted p-values, not a confidence band.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import pandas as pd
            from sklearn.linear_model import LogisticRegression

            from tcda_uq import evaluate_coverage, tate_band, tate_test
            from tcda_uq.datasets import TriOracleSimulation
            from tcda_uq.estimators import cross_fit
            """
        ),
        code(
            """
            sim = TriOracleSimulation(
                n_hom_dim=1, resolution=28, n_basis=5, noise_scale=0.3, seed=0
            )
            sample = sim.sample(240, rng=1)
            fit = cross_fit(
                sample.observed,
                sim.tseq,
                n_basis=5,
                n_splits=2,
                random_state=2,
                stratify=False,
                propensity_estimator=LogisticRegression(max_iter=500),
            )
            truth = sim.true_tate()[0]
            band = tate_band(
                fit, method="mbb", d=0, alpha=0.10, n_boot=120, rng=3
            )
            inclusion = evaluate_coverage(band, truth).coverage
            """
        ),
        code(
            """
            pd.DataFrame(
                [{
                    "object": "TATE mean confidence band",
                    "grid_inclusion_for_this_sample": inclusion,
                    "grid_mean_width": band.mean_width(),
                    "multiplier_default": "Gaussian, unstudentized",
                }]
            )
            """
        ),
        code(
            """
            iwt = tate_test(fit, method="iwt", d=0, alpha=0.10, n_boot=120, rng=4)
            pd.DataFrame(
                {
                    "filtration value": sim.tseq,
                    "adjusted p-value": iwt.p_adjusted,
                    "reject on grid": iwt.significant(adjusted=True),
                }
            )
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(sim.tseq, truth, color="black", lw=2, label="known TATE mean")
            ax.plot(sim.tseq, band.center, lw=1.5, label="AIPW estimate")
            ax.fill_between(sim.tseq, band.lower, band.upper, alpha=0.2,
                            label="Gaussian multiplier confidence band")
            ax.set(xlabel="filtration value", ylabel="effect",
                   title="TATE mean inference on a fixed grid")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            """
        ),
    ]


def ctate_notebook():
    return [
        md(
            """
            CTATE inference concerns the conditional mean curve at one prespecified
            covariate value. It does not cover an individual treatment effect at
            that `x`. See the ITTE tutorial for a separate marginal prediction
            example evaluated on draws from its declared target population.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import pandas as pd
            from sklearn.linear_model import LogisticRegression

            from tcda_uq import ctate_band, evaluate_coverage
            from tcda_uq.datasets import TriOracleSimulation
            from tcda_uq.estimators import CTATEDRLearner
            """
        ),
        code(
            """
            sim = TriOracleSimulation(
                n_hom_dim=1, resolution=28, n_basis=5, noise_scale=0.3, seed=4
            )
            sample = sim.sample(300, rng=5)
            x = sim.EX + 0.25
            learner = CTATEDRLearner(n_basis=5).fit(
                sample.observed,
                sim.tseq,
                n_splits=2,
                random_state=6,
                stratify=False,
                propensity_estimator=LogisticRegression(max_iter=500),
            )
            band = ctate_band(
                learner, x, d=0, alpha=0.10, n_boot=120, rng=7
            )
            mean_truth = sim.true_ctate(x[None, :])[0, 0]
            inclusion = evaluate_coverage(band, mean_truth).coverage
            """
        ),
        code(
            """
            pd.DataFrame(
                [{
                    "target": "conditional mean at fixed x",
                    "grid_inclusion_for_this_sample": inclusion,
                    "grid_mean_width": band.mean_width(),
                }]
            )
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(sim.tseq, mean_truth, color="black", lw=2,
                    label="known CTATE mean")
            ax.plot(sim.tseq, band.center, lw=1.5, label="estimated CTATE")
            ax.fill_between(sim.tseq, band.lower, band.upper, alpha=0.23,
                            label="confidence band for the mean")
            ax.set(xlabel="filtration value", ylabel="effect",
                   title="CTATE at a prespecified covariate value")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            """
        ),
    ]


def itte_notebook():
    return [
        md(
            """
            This example compares full, overlap, and matching population targets.
            Each model uses the simulator's known propensity, and each is evaluated
            on a fresh sample from the same target used by its conformal weights.
            Training and calibration are split inside `ITTEConformal.fit`.
            Coverage is a finite-sample lower bound under the stated split and
            identification conditions; empirical inclusion need not equal its
            nominal level. Bounded target weights can still produce infinite bands.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            from tcda_uq import evaluate_coverage
            from tcda_uq.datasets import TriOracleSimulation
            from tcda_uq.uq.conformal import ITTEConformal
            """
        ),
        code(
            """
            sim = TriOracleSimulation(
                n_hom_dim=1,
                resolution=26,
                n_basis=5,
                noise_scale=0.3,
                prop_scale=2.5,
                seed=10,
            )
            train = sim.sample(500, rng=11)
            targets = ("full", "overlap", "matching")
            models = {
                target: ITTEConformal.fit(
                    train.observed,
                    sim.tseq,
                    n_basis=5,
                    target=target,
                    propensity_fn=sim.propensity_score,
                    weights_known=True,
                    random_state=20 + j,
                )
                for j, target in enumerate(targets)
            }
            test_samples = {
                target: sim.sample_target(180, target=target, rng=30 + j)
                for j, target in enumerate(targets)
            }
            """
        ),
        code(
            """
            rows = []
            for target in targets:
                test = test_samples[target]
                model = models[target]
                lower, upper, _ = model.band_bounds(test.X, alpha=0.10, d=0)
                result = evaluate_coverage(
                    (lower, upper), test.oracle_itte[:, 0, :], level=0.90
                )
                finite = np.isfinite(upper - lower).all(axis=1)
                widths = (upper - lower).mean(axis=1)
                rows.append(
                    {
                        "target": target,
                        "weights known": model.weights_known_,
                        "grid inclusion": result.coverage,
                        "finite band share": finite.mean(),
                        "mean width among finite bands": (
                            widths[finite].mean() if finite.any() else np.nan
                        ),
                    }
                )
            pd.DataFrame(rows)
            """
        ),
        code(
            """
            test = test_samples["full"]
            lower, upper, center = models["full"].band_bounds(
                test.X[:1], alpha=0.10, d=0
            )
            individual = test.oracle_itte[0, 0]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.fill_between(sim.tseq, lower[0], upper[0], alpha=0.2,
                            label="prediction band for a full-target draw")
            ax.plot(sim.tseq, center[0], lw=1.5, label="predicted ITTE center")
            ax.plot(sim.tseq, individual, color="black", ls="--",
                    label="realized simulated ITTE")
            ax.set(xlabel="filtration value", ylabel="individual effect",
                   title="One new draw from the full-population target")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            """
        ),
    ]


def orbit_notebook():
    return [
        md(
            """
            This small synthetic example runs point clouds through persistence
            and plain silhouette construction before estimating a TATE mean curve.
            Treatment is randomized independently across units. The notebook plots
            a confidence band and estimate, but does not claim coverage from this
            single generated data set.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.linear_model import LogisticRegression

            from tcda_uq import tate_band
            from tcda_uq.datasets import make_orbit_causal
            from tcda_uq.estimators import cross_fit
            """
        ),
        code(
            """
            rng = np.random.default_rng(20)
            orbit = make_orbit_causal(
                n=30,
                p_reverse=0.25,
                interval=(0.0, 0.25),
                r=3.0,
                resolution=22,
                homology_dims=(0, 1),
                num_pts=40,
                rng=21,
            )
            potential_outcomes = orbit["potential_outcomes"]
            n = potential_outcomes.shape[0]
            X = rng.normal(size=(n, 3))
            A = rng.binomial(1, 0.5, size=n)
            observed = potential_outcomes[np.arange(n), A]
            sample = (observed, A, X)
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
                stratify=False,
                propensity_estimator=LogisticRegression(max_iter=500),
            )
            band = tate_band(fit, method="mbb", d=1, alpha=0.10,
                             n_boot=100, rng=23)
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.fill_between(orbit["tseq"], band.lower, band.upper, alpha=0.22,
                            label="Gaussian multiplier confidence band")
            ax.plot(orbit["tseq"], band.center, lw=1.5, label="AIPW estimate")
            ax.set(xlabel="filtration value", ylabel="H1 silhouette effect",
                   title="ORBIT pipeline on a fixed grid")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            """
        ),
    ]


def sars_notebook():
    return [
        md(
            """
            This tutorial converts CT images to H0 silhouettes, then constructs a
            synthetic randomized example. It can read local images through
            `TCDA_UQ_SARS_ROOT`; potential outcomes and treatment assignment are
            still generated for demonstration. It does not estimate a clinical
            treatment effect or evaluate counterfactual coverage in observed
            patients. The band is shown without a coverage comparison.
            """
        ),
        code(
            """
            import os

            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.linear_model import LogisticRegression

            from tcda_uq import tate_band
            from tcda_uq.datasets.sarscov2 import (
                images_to_silhouettes,
                load_ct_images,
                make_sarscov2_causal,
            )
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
                source = "local CT images with synthetic treatment/outcomes"
            else:
                rng = np.random.default_rng(30)
                grid = np.linspace(-1.0, 1.0, 18)
                xx, yy = np.meshgrid(grid, grid)

                def blob(offset, scale):
                    base = np.exp(-((xx - offset) ** 2 + yy**2) / scale)
                    return np.clip(base + 0.08 * rng.normal(size=base.shape), 0.0, 1.0)

                infected = [blob(-0.25, 0.22) for _ in range(14)]
                noninfected = [blob(0.25, 0.35) for _ in range(14)]
                source = "synthetic CT-like images and outcomes"

            phi_inf = images_to_silhouettes(
                infected, r=1.0, resolution=24, workers=1
            )
            phi_non = images_to_silhouettes(
                noninfected, r=1.0, resolution=24, workers=1
            )
            causal = make_sarscov2_causal(
                phi_inf, phi_non, n=24, p=0.70, rng=31
            )
            """
        ),
        code(
            """
            rng = np.random.default_rng(32)
            potential_outcomes = causal["potential_outcomes"]
            n = potential_outcomes.shape[0]
            X = rng.normal(size=(n, 2))
            A = rng.binomial(1, 0.5, size=n)
            observed = potential_outcomes[np.arange(n), A]
            tseq = np.linspace(0.0, 1.0, potential_outcomes.shape[-1])

            fit = cross_fit(
                (observed, A, X),
                tseq,
                n_basis=4,
                n_splits=2,
                random_state=33,
                stratify=False,
                propensity_estimator=LogisticRegression(max_iter=500),
            )
            band = tate_band(fit, method="mbb", d=0, alpha=0.10,
                             n_boot=100, rng=34)
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].imshow(infected[0], cmap="gray")
            axes[0].set_title(source)
            axes[0].axis("off")
            axes[1].fill_between(tseq, band.lower, band.upper, alpha=0.22,
                                 label="mean confidence band")
            axes[1].plot(tseq, band.center, color="black", lw=1.5,
                         label="estimated mean effect")
            axes[1].set_title("Synthetic randomized example")
            axes[1].set_xlabel("filtration value")
            axes[1].legend(loc="best", fontsize=8)
            fig.tight_layout()
            """
        ),
    ]


def main():
    NOTEBOOKS.mkdir(exist_ok=True)
    specs = [
        ("01_tate_confidence_bands.ipynb", "TATE Mean Confidence Bands", tate_notebook()),
        ("02_ctate_bridge.ipynb", "CTATE Mean Inference", ctate_notebook()),
        ("03_itte_conformal.ipynb", "Target-Matched ITTE Prediction", itte_notebook()),
        ("04_orbit_pipeline.ipynb", "ORBIT Point-Cloud Pipeline", orbit_notebook()),
        ("05_sarscov2_demo.ipynb", "Synthetic CT Image Demonstration", sars_notebook()),
    ]
    for filename, title, cells in specs:
        write_notebook(NOTEBOOKS / filename, title, cells)
        print(f"wrote notebooks/{filename}")


if __name__ == "__main__":
    main()
