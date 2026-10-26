"""Reproduce the October 26 PACCE corrections using investigator-supplied SAS.

Run ``python reproduction/pacce_coauthor_revision.py --help`` for usage. The
trial files must be supplied separately and are never extracted or downloaded.
All analysis objects stay in memory. Only aggregate tables and sanitized method
provenance are written, to an explicitly selected directory outside this
checkout. The synthetic fixture exercises the same pipeline without trial data.

The historical package is deliberately used to isolate the corrected profile
age and calibration-independent prediction scores. Its old documentation is
not an endorsement of a causal interpretation or finite-sample coverage claim.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


CHECKOUT = Path(__file__).resolve().parents[1]
SOURCE_ARCHIVE = (
    "reproduction/sources/"
    "run_sources_5fdce0b48fe341aee1f1b31312ce5a35a27872577e5ed2ed44d25fc3cbf45625.zip"
)
ARCHIVE_SHA256 = "37448bf27f41e2736219aa9ed6d18b13bdae4a25f3a0d54e91ea7145362bb46c"
# All 42 archived Python members were compared byte-for-byte with the preserved
# October 10 workspace during preparation. Only these 36 package files are
# extracted; historical scripts, tests, data and results are never extracted.
FROZEN_SHA256 = {
    "tcda_uq/__init__.py": "cebda2b87eb49e3cfaa5994eadf11a93c9db591c9df3013976e9daf8f10fe5a3",
    "tcda_uq/api.py": "94881751d5a5848b48175fc0371c1c87acb0e589a3019dc1f79305934450e9ee",
    "tcda_uq/datasets/__init__.py": "7487a6e38a45abdfa0283be1801fb0ca6f4543e44c0f9f167bb7c30436c38bf3",
    "tcda_uq/datasets/covariates.py": "bdb1e7355b9fc58a43255a6b092797b2422cc8ca013bbb732830a881a34cf3b2",
    "tcda_uq/datasets/orbit.py": "f9d024c5b91a5507c0b4ebc2e862219bd6fa419f257f27a39c9e21668078fff0",
    "tcda_uq/datasets/pacce.py": "d3d58436cdd830b9172493f6dbe8d55ff12843db0010972766f73154be811199",
    "tcda_uq/datasets/sarscov2.py": "bbb0fbe6b2c70efb092599d5de6e1cc7f4edb61713b2f905eedf318b47d3b385",
    "tcda_uq/datasets/simulation.py": "2a665655ae9b827116a7bda8a299304149a3817585e13296172838ed57177270",
    "tcda_uq/datasets/topological.py": "b834774aa5490c360bb56728b7cfffc8fa22ee30257cc95e90c7a468ad0a0332",
    "tcda_uq/estimators/__init__.py": "8588e30a6debfb6d35ddb019ba65bcd4ca15a5d2a5c20e48e18121e3add89329",
    "tcda_uq/estimators/aipw.py": "ae6caa4576dd86609f51875fa0c65d676db5d2203d972793b130ba00f90b308e",
    "tcda_uq/estimators/ctate_dr_learner.py": "a37d2fc74c73ed8b284898d2125cc49747b8775087b164fb25106027fd035b2f",
    "tcda_uq/estimators/nuisance.py": "6f31f31811f1006de5de6f28e4793256f3f3a6077240812b3a73096618e4ab12",
    "tcda_uq/metrics/__init__.py": "1fc1425df106830976a4a23013c211bb3611fa210785ea0107efa52530117d40",
    "tcda_uq/metrics/bands.py": "19cdc36916a0a69b8d9184767da8ffe038fcf3950f5fb8dfbe44bd2f52ed3da6",
    "tcda_uq/metrics/coverage.py": "f13b5fed8f694f0f50d0ae54a7c11b94f7218a9a87a6ad94062aad5ed5dcdfb8",
    "tcda_uq/metrics/plotting.py": "1bc79b53f5849a864e2d612d68ec0447452c7d719e12287668cb6183b988b054",
    "tcda_uq/silhouette/__init__.py": "94c562ba1f6fe8e3a6429a585e61a55d10aad2c9907c34ee28757f18b0b233db",
    "tcda_uq/silhouette/core.py": "b3b977bdd57ea78e72d449cca78f57adbfac35e0fd6500f59abc00c3203aed0b",
    "tcda_uq/uq/__init__.py": "1e03be650c3c0ebb1e2525053e7683904c5d8f3b3c264adb91bb5448b370d7f7",
    "tcda_uq/uq/asymptotic/__init__.py": "a7a6660276d605618ac386b4a7ff7e562e97598a55c7b4f0c789cf4ec3833626",
    "tcda_uq/uq/asymptotic/covariance.py": "98463991dcb02013b2bdf17352e993b1bb5c482392a32136ea633c0ad3270017",
    "tcda_uq/uq/asymptotic/ctate_bands.py": "2dc35a647e30ee59f6afa96a1964d77872093be5ec3a276586af936a5036db8f",
    "tcda_uq/uq/asymptotic/liebl_reimherr.py": "5b853bc136b812c355b9b55bec1fe474ac033e4128d9c96555f916132b2ea3ce",
    "tcda_uq/uq/asymptotic/multiplier_bootstrap.py": "cfaf5cc626ec48ae8a5a9e6740c85e59269e38d9f022b486aecb59b3ceedafa9",
    "tcda_uq/uq/asymptotic/pini_vantini.py": "c4232f5fae3004a98f86ff0997f88f4efd275a1e0c5089044c483c870d0f3841",
    "tcda_uq/uq/conformal/__init__.py": "d9ebbfc884913fd195a061b4e90f1f0b06102c69def143e0134b6ffd74d6d36b",
    "tcda_uq/uq/conformal/adaptive_cp.py": "bda7646978aed9dde5227952f4fa1533aad601063658bcdf74821ef8f3fc755f",
    "tcda_uq/uq/conformal/composition.py": "628ce3f903e8a46ec6530494f2cc6a925bd46784cbabc110e0075b830a706b69",
    "tcda_uq/uq/conformal/degree_multiplicity.py": "d0e3ebceec268bf3ab570ec2fcc2b958532b7217a350fc48084a657604c4afa8",
    "tcda_uq/uq/conformal/diagram_score.py": "ba7f685dfa329fbd0e84febe0104530390dce6466021eb91572f14bab307f4a1",
    "tcda_uq/uq/conformal/functional_cp.py": "b13d46d76d8b1d6f89e4b2e8f07eefbfb964dec98d2f58e63c3bdec382973418",
    "tcda_uq/uq/conformal/silhouette_bridge.py": "709b5c0d401cdcf45d77d26b06b086288aeaa17b8e219c419529a330428336de",
    "tcda_uq/uq/conformal/stabilized_weights.py": "8b79f8e309ebc330a1d2810f4f96c4e70d9678c28177faaf1a1f4a7954dec3d7",
    "tcda_uq/uq/conformal/weighted_cp.py": "c84dc9f9eaabd806acc35a89b3ae046835cb84ceef4f5709c6c8bd28c6309646",
    "tcda_uq/utils.py": "79448986a24dc56190fff45c93f9cf9f1a4cb65f9634c798231eec1690c6272c",
}

OUTPUT_COLUMNS = {
    "ctate_age61_profiles.csv": (
        "profile", "age_years", "folds", "n", "n_matching", "peak_abs_tau",
        "t_at_peak", "mean_halfwidth", "mean_full_width", "excursion_frac",
        "sup_stat", "sup_pvalue", "bonferroni_p", "band_seed", "test_seed",
    ),
    "prediction_width_summary.csv": (
        "method", "profile", "age_years", "mean_full_width", "median_full_width",
        "support_intersected_mean_full_width", "support_intersection_empty_nodes",
        "support_intersection_zero_included_all", "zero_included_all", "unbounded",
        "peak_abs_center", "full_support_envelope_mean_width",
        "arm0_radius", "arm1_radius", "arm0_scale_mean", "arm1_scale_mean",
        "width_from_arm_radii", "per_arm_alpha",
    ),
    "arm_score_diagnostics.csv": (
        "method", "arm", "n_calibration", "scale_min", "scale_mean", "scale_max",
        "score_min", "score_median", "score_max",
    ),
    "fold_diagnostics.csv": (
        "folds", "seed", "n", "pvalue", "peak_abs_estimate", "mean_full_width",
        "zero_included_all", "fold_heldout_arm_counts",
    ),
}


def outside_checkout(path, *, description="output directory"):
    """Resolve symlinks before enforcing the external-directory boundary."""
    resolved = Path(path).expanduser().resolve()
    if resolved == CHECKOUT or CHECKOUT in resolved.parents:
        raise ValueError(f"The {description} must be outside this checkout")
    return resolved


@contextmanager
def frozen_package():
    """Authenticate the archive and code members before extracting code only."""
    archive = CHECKOUT / SOURCE_ARCHIVE
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Historical source archive checksum mismatch")
    with zipfile.ZipFile(archive) as source:
        if len(set(source.namelist())) != len(source.namelist()):
            raise ValueError("Historical source archive has duplicate members")
        code = {}
        for member, digest in FROZEN_SHA256.items():
            raw = source.read(member)
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError(f"Historical source member checksum mismatch: {member}")
            code[member] = raw
    with tempfile.TemporaryDirectory(prefix="pacce-frozen-code-") as directory:
        root = Path(directory)
        for member, raw in code.items():
            target = root / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        yield root


def design(X):
    """Historical baseline design, with burden bins retained at count >= 15."""
    import numpy as np

    burden = X[:, 4]
    bins = np.column_stack([((burden >= 3) & (burden <= 4)).astype(float),
                            (burden >= 5).astype(float)])
    full = np.column_stack([X, bins])
    keep = np.ones(full.shape[1], dtype=bool)
    for j in range(X.shape[1], full.shape[1]):
        keep[j] = min(full[:, j].sum(), len(full) - full[:, j].sum()) >= 15
    return full[:, keep], keep


def profiles(keep):
    """Six prespecified covariate queries, all at age 61 years."""
    import numpy as np

    names, rows = [], []
    for chemo_name, chemo in (("oxali", 0.), ("iri", 1.)):
        for burden_name, burden in (("low", 2.), ("mid", 4.), ("high", 6.)):
            names.append(f"{chemo_name}_{burden_name}_burden")
            raw = [chemo, 61., 1., 0., burden, 2., 1., 0., 0., 0.,
                   float(3 <= burden <= 4), float(burden >= 5)]
            rows.append(np.asarray(raw)[keep])
    return names, np.asarray(rows)


def constant_arm(nuisance, calibration, arm):
    """Explicit constant-one scale, independent of calibration outcomes."""
    from tcda_uq.uq.conformal.weighted_cp import ConformalArm

    return ConformalArm(nuisance, calibration, 0, arm,
                        modulation_kind="constant", propensity_feature_fn=lambda z: z[:, :1])


def width_summaries(model, queried, names, t):
    """Reduce six query bands immediately; never serialize the grid curves."""
    import numpy as np

    lo, hi, center = model.band_bounds(queried, alpha=.1)
    support = np.sqrt(2.) * t
    lower, upper = np.maximum(lo, -support), np.minimum(hi, support)
    radius0 = model.arms0[0].radius(queried, .05)
    radius1 = model.arms1[0].radius(queried, .05)
    scale0, scale1 = model.arms0[0].s.mean(), model.arms1[0].s.mean()
    rows = []
    for j, name in enumerate(names):
        rows.append(dict(
            method="corrected_constant", profile=name, age_years=61.,
            mean_full_width=float(np.mean(hi[j] - lo[j])),
            median_full_width=float(np.median(hi[j] - lo[j])),
            support_intersected_mean_full_width=float(np.mean(np.maximum(upper[j] - lower[j], 0))),
            support_intersection_empty_nodes=int(np.sum(lower[j] > upper[j])),
            support_intersection_zero_included_all=bool(np.all((lower[j] <= 0) & (upper[j] >= 0))),
            zero_included_all=bool(np.all((lo[j] <= 0) & (hi[j] >= 0))),
            unbounded=bool(np.any(~np.isfinite(hi[j] - lo[j]))),
            peak_abs_center=float(np.abs(center[j]).max()),
            full_support_envelope_mean_width=float(np.mean(2 * support)),
            arm0_radius=float(radius0[j]), arm1_radius=float(radius1[j]),
            arm0_scale_mean=float(scale0), arm1_scale_mean=float(scale1),
            width_from_arm_radii=float(2 * (radius0[j] * scale0 + radius1[j] * scale1)),
            per_arm_alpha=.05,
        ))
    return rows


def _read_frames(sas_dir, adapter):
    """Read only the needed SAS columns, without copying input files."""
    import pandas as pd

    def read(domain, columns):
        path = sas_dir / f"{domain}.sas7bdat"
        try:
            import pyreadstat
        except ImportError:
            return pd.read_sas(path, format="sas7bdat", encoding="utf-8")[columns]
        return pyreadstat.read_sas7bdat(str(path), usecols=columns)[0]

    return read("demo", adapter.DEMO_REQUIRED), read("lesion", adapter.LESION_REQUIRED)


def _fit_cross(sample, t, stratify, seed):
    from sklearn.linear_model import LogisticRegression
    from tcda_uq.estimators import cross_fit

    return cross_fit(sample, t, n_basis=5,
                     propensity_estimator=LogisticRegression(C=1., max_iter=2000),
                     propensity_feature_fn=lambda z: z[:, :1], n_splits=2,
                     stratify=stratify, random_state=seed)


def _ctate_rows(sample, t, cf, queried, names, Xraw, n_boot):
    import numpy as np
    from tcda_uq.estimators import CTATEDRLearner
    from tcda_uq.uq.asymptotic import ctate_confidence_band
    from tcda_uq.uq.asymptotic.multiplier_bootstrap import _draw_multipliers

    learner = CTATEDRLearner(n_basis=4).fit(sample, t, cross_fit_result=cf)
    rows = []
    for j, (name, x) in enumerate(zip(names, queried)):
        band = ctate_confidence_band(learner, x, d=0, alpha=.1, n_boot=n_boot,
                                    multiplier="rademacher", standardize=True, rng=910000 + j)
        weights = learner.weights(x, d=0)
        contrib = weights[:, None] * learner.residuals(0)
        scale = np.sqrt(np.maximum(np.einsum("it,it->t", contrib, contrib), 1e-24))
        stat = float(np.max(np.abs(band.center) / scale))
        xi = _draw_multipliers("rademacher", (n_boot, len(weights)), np.random.default_rng(920000 + j))
        boot = np.max(np.abs((xi @ contrib) / scale), axis=1)
        pvalue = float(np.mean(boot >= stat))
        rows.append(dict(
            profile=name, age_years=61., folds="treatment_stratified", n=len(sample[1]),
            n_matching=int(np.sum((Xraw[:, 0] == (j >= 3)) & (Xraw[:, 4] == (2, 4, 6)[j % 3]))),
            peak_abs_tau=float(np.abs(band.center).max()),
            t_at_peak=float(t[np.argmax(np.abs(band.center))]),
            mean_halfwidth=float(np.mean(band.upper - band.lower)) / 2,
            mean_full_width=float(np.mean(band.upper - band.lower)),
            excursion_frac=float(np.mean((band.lower > 0) | (band.upper < 0))),
            sup_stat=stat, sup_pvalue=pvalue, bonferroni_p=min(1., 6 * pvalue),
            band_seed=910000 + j, test_seed=920000 + j,
        ))
    return rows, int(learner.stages_[0].B.shape[1])


def _fold_row(sample, t, cf, stratify, seed, n_boot):
    import numpy as np
    from sklearn.model_selection import KFold, StratifiedKFold
    from tcda_uq.uq.asymptotic import multiplier_bootstrap_band, topological_effect_test

    band = multiplier_bootstrap_band(cf.influence()[0], t, cf.aipw[0], alpha=.1,
                                     n_boot=n_boot, multiplier="rademacher", standardize=True, rng=seed + 1)
    test = topological_effect_test(cf.influence()[0], cf.aipw[0], n_boot=n_boot,
                                   multiplier="rademacher", standardize=False, rng=seed + 5)
    _, A, X = sample
    splitter = (StratifiedKFold if stratify else KFold)(2, shuffle=True, random_state=seed)
    counts = [np.bincount(A[idx].astype(int), minlength=2).tolist()
              for _, idx in splitter.split(X, A if stratify else None)]
    return dict(folds="treatment_stratified" if stratify else "independent_KFold", seed=seed,
                n=len(A), pvalue=test["pvalue"], peak_abs_estimate=float(np.abs(cf.aipw[0]).max()),
                mean_full_width=float(np.mean(band.upper - band.lower)),
                zero_included_all=bool(np.all((band.lower <= 0) & (band.upper >= 0))),
                fold_heldout_arm_counts=json.dumps(counts))


def _prediction_diagnostics(model, sample):
    import numpy as np
    from tcda_uq.uq.conformal.weighted_cp import ConformalArm

    phi, A, X = sample
    cal = model.calib_index_
    train = np.setdiff1d(np.arange(len(A)), cal)
    calibration = (phi[cal], A[cal], X[cal])
    rows = []
    for method in ("corrected_constant", "legacy_calibration_sd", "corrected_training_sd"):
        for arm in (0, 1):
            if method == "corrected_constant":
                fitted = (model.arms0, model.arms1)[arm][0]
            else:
                training = train[A[train] == arm]
                residuals = phi[training, 0] - model.nuisance_.predict_mu(X[training])[0][arm]
                fitted = ConformalArm(model.nuisance_, calibration, 0, arm,
                                      modulation_kind="pointwise-sd",
                                      modulation_residuals=residuals if method == "corrected_training_sd" else None,
                                      propensity_feature_fn=lambda z: z[:, :1])
            rows.append(dict(method=method, arm=arm, n_calibration=len(fitted.scores),
                             scale_min=float(fitted.s.min()), scale_mean=float(fitted.s.mean()),
                             scale_max=float(fitted.s.max()), score_min=float(fitted.scores.min()),
                             score_median=float(np.median(fitted.scores)), score_max=float(fitted.scores.max())))
    return rows


def _write_outputs(out_dir, tables, provenance):
    """Allowlist serialization: no generic model, frame, or array serializer."""
    import pandas as pd

    if set(tables) != set(OUTPUT_COLUMNS):
        raise ValueError("Unexpected aggregate output table")
    for filename, columns in OUTPUT_COLUMNS.items():
        for row in tables[filename]:
            if set(row) != set(columns):
                raise ValueError(f"Unexpected aggregate columns in {filename}")
    out_dir = outside_checkout(out_dir)
    destinations = [out_dir / name for name in (*OUTPUT_COLUMNS, "method_provenance.json")]
    for path in destinations:
        outside_checkout(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite an existing output: {path.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, columns in OUTPUT_COLUMNS.items():
        pd.DataFrame(tables[filename], columns=columns).to_csv(out_dir / filename, index=False)
    (out_dir / "method_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")


def _run(args):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from skfda.representation.basis import FourierBasis
    import tcda_uq
    from tcda_uq.datasets import pacce as adapter
    from tcda_uq.uq.conformal import ITTEConformal

    # The fresh worker must import the authenticated code, never the installed
    # package. Nothing patient-specific is written in this temporary directory.
    frozen_root = Path(args._frozen_root).resolve()
    if Path(tcda_uq.__file__).resolve().parent != frozen_root / "tcda_uq":
        raise RuntimeError("Historical import isolation failed")
    if args.fixture:
        demo, lesion = adapter.synthetic_pacce_frames(n=args.fixture_subjects, seed=20261026)
    else:
        demo, lesion = _read_frames(outside_checkout(args.sas_dir, description="SAS directory"), adapter)
    spec = adapter.load_outcome_spec()
    # Explicitly preserve the historical outcome geometry and same-cohort scale.
    spec["silhouette"] = {"r": 3., "resolution": 100, "interval": [0., .83]}
    spec["homology_dims"] = [0, 1]
    data = adapter.load_pacce_from_frames(demo, lesion, spec)
    phi, A, t = data.phi[:, [0]], data.A, data.tseq
    if not args.fixture and len(A) != 566:
        raise ValueError("The eligible cohort does not match the prespecified n=566 analysis")
    Xraw = data.X[adapter.BASELINE_COLS].to_numpy(float)
    X, keep = design(Xraw)
    sample = (phi, A, X)
    names, queried = profiles(keep)
    model = ITTEConformal.fit(sample, t, calib_frac=.5, n_basis=5,
                              propensity_estimator=LogisticRegression(C=1., max_iter=2000),
                              propensity_feature_fn=lambda z: z[:, :1],
                              modulation_kind="constant", random_state=20243)
    # Construct the score arms explicitly so the correction never depends on a
    # historical or current default modulation setting.
    cal = model.calib_index_
    calibration = (phi[cal], A[cal], X[cal])
    model.arms0 = [constant_arm(model.nuisance_, calibration, 0)]
    model.arms1 = [constant_arm(model.nuisance_, calibration, 1)]
    for fitted in (model.arms0[0], model.arms1[0]):
        if not np.array_equal(fitted.s, np.ones(len(t))):
            raise RuntimeError("Prediction scores require a constant-one scale")
    print("Fitting six age-61 aggregate profiles.", flush=True)
    cf = _fit_cross(sample, t, True, 20240)
    ctate, second_stage_functions = _ctate_rows(sample, t, cf, queried, names, Xraw, args.ctate_boot)
    folds = [_fold_row(sample, t, cf, True, 20240, args.fold_boot)]
    if args.independent_fold_sensitivity:
        for seed in range(20240, 20245):
            print(f"Computing aggregate independent-fold sensitivity, seed {seed}.", flush=True)
            independent = _fit_cross(sample, t, False, seed)
            folds.append(_fold_row(sample, t, independent, False, seed, args.fold_boot))
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "scikit-learn", "scikit-fda", "gudhi", "pandas")}
    provenance = dict(
        analysis="PACCE October 26 coauthor corrections",
        data_kind="synthetic fixture" if args.fixture else "investigator-supplied controlled SAS",
        n=int(len(A)), arm_counts=np.bincount(A.astype(int), minlength=2).tolist(),
        source_archive=SOURCE_ARCHIVE, source_archive_sha256=ARCHIVE_SHA256,
        authenticated_source_sha256=FROZEN_SHA256, versions=versions,
        geometry=dict(homology=0, persistence_weight_power=3., normalization="legacy GUDHI sqrt(2) times plain silhouette",
                      grid_start=0., grid_stop=.83, grid_nodes=100,
                      standardization="pooled coordinates over the same eligible cohort",
                      support="[-sqrt(2)*t, +sqrt(2)*t]"),
        profile_age_years=61.,
        prediction=dict(modulation="constant one, calibration independent", split_seed=20243,
                        calibration_fraction=.5, nominal_level=.9, per_arm_alpha=.05,
                        outcome_fourier_functions=int(FourierBasis(t[[0, -1]], 5).n_basis),
                        propensity="logistic regression on chemotherapy indicator; C=1; max_iter=2000",
                        weights="estimated inverse propensity; historical numerical clipping retained"),
        ctate=dict(folds="treatment-stratified two-fold", fit_seed=20240,
                   outcome_fourier_functions=int(FourierBasis(t[[0, -1]], 5).n_basis),
                   second_stage_requested_basis=4,
                   second_stage_fourier_functions=second_stage_functions,
                   bootstrap_draws=args.ctate_boot, multiplier="rademacher",
                   band="historical HC0 studentized; variance floor 1e-24; nominal level .9",
                   test="studentized absolute supremum; per-profile diagnostic",
                   band_seeds=list(range(910000, 910006)), test_seeds=list(range(920000, 920006))),
        fold_sensitivity=dict(independent_seeds=list(range(20240, 20245)) if args.independent_fold_sensitivity else [],
                              bootstrap_draws=args.fold_boot, multiplier="rademacher",
                              band="historical studentized", test="unstudentized absolute supremum",
                              band_seed_offset=1, test_seed_offset=5),
        interpretation="descriptive observed-group analysis; no causal identification or exact conformal coverage claim",
        score_diagnostics="legacy calibration SD and training SD are private diagnostic comparisons; the corrected prediction procedure uses constant one",
        privacy="aggregate summaries only; no records, identifiers, input hashes, input paths, split indices, or individual curves",
        resources="one process; BLAS and OpenMP restricted to one thread",
    )
    _write_outputs(args.out_dir, {
        "ctate_age61_profiles.csv": ctate,
        "prediction_width_summary.csv": width_summaries(model, queried, names, t),
        "arm_score_diagnostics.csv": _prediction_diagnostics(model, sample),
        "fold_diagnostics.csv": folds,
    }, provenance)
    print("Wrote four aggregate CSV tables and sanitized method provenance.", flush=True)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sas-dir", type=Path, help="external investigator-supplied folder containing demo.sas7bdat and lesion.sas7bdat")
    inputs.add_argument("--fixture", action="store_true", help="use non-confidential synthetic SAS-schema frames instead of trial data")
    parser.add_argument("--out-dir", required=True, type=Path, help="explicit aggregate output directory OUTSIDE this checkout; existing output files are never overwritten")
    parser.add_argument("--include-fold-sensitivity", "--independent-fold-sensitivity",
                        dest="independent_fold_sensitivity", action="store_true",
                        help="add five independent KFold sensitivities with fit seeds 20240 through 20244")
    parser.add_argument("--ctate-boot", type=int, default=1999, help="CTATE Rademacher draws (1999; reduced values allowed only with --fixture)")
    parser.add_argument("--fold-boot", type=int, default=4999, help="fold sensitivity Rademacher draws (4999; reduced values allowed only with --fixture)")
    parser.add_argument("--fixture-subjects", type=int, default=160, help="synthetic subjects before eligibility filtering (default 160; minimum 80)")
    parser.add_argument("--_frozen-root", help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.out_dir = outside_checkout(args.out_dir)
        if args.sas_dir:
            args.sas_dir = outside_checkout(args.sas_dir, description="SAS directory")
        if args.ctate_boot < 1 or args.fold_boot < 1 or args.fixture_subjects < 80:
            parser.error("bootstrap draws must be positive and fixture subjects at least 80")
        if not args.fixture and (args.ctate_boot != 1999 or args.fold_boot != 4999):
            parser.error("the controlled-data reproduction requires 1999 CTATE and 4999 fold bootstrap draws")
        if args._frozen_root:
            # -I discards PYTHONPATH. Add only the authenticated package root,
            # before importing any scientific package in this fresh process.
            root = Path(args._frozen_root).resolve()
            for member, digest in FROZEN_SHA256.items():
                if hashlib.sha256((root / member).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Historical worker source checksum mismatch: {member}")
            sys.path.insert(0, str(root))
            _run(args)
            return 0
        with frozen_package() as root:
            env = os.environ.copy()
            for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[variable] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env.setdefault("MPLCONFIGDIR", str(root / "matplotlib"))
            # Canonical absolute input/output paths preserve their meaning when
            # the isolated worker changes its working directory to the code.
            arguments = ["--fixture"] if args.fixture else ["--sas-dir", str(args.sas_dir)]
            arguments += ["--out-dir", str(args.out_dir), "--ctate-boot", str(args.ctate_boot),
                          "--fold-boot", str(args.fold_boot), "--fixture-subjects", str(args.fixture_subjects)]
            if args.independent_fold_sensitivity:
                arguments.append("--include-fold-sensitivity")
            return subprocess.call([sys.executable, "-I", "-B", str(Path(__file__).resolve()), *arguments,
                                    "--_frozen-root", str(root)], cwd=root, env=env)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
