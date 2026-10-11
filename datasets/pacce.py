"""PACCE selected-cohort exploratory adapter (Amgen 20040249 release).

Turns released SAS domains into subject-level ``(phi, A, X)`` arrays under
the geometric specification v1.0. Eligibility uses post-randomization visits
and observed lesion measurements, so it may be affected by treatment. Original
randomization alone does not identify a causal effect in this selected cohort.
Outputs are descriptive/associational unless a separate selection argument is
provided. Fitting pooled scales to these same outcomes also does not meet the
fixed-map assumptions of the inference theorems; externally prespecified scales
can be supplied through ``spec['standardization']``.

Construction:

* central target-lesion records, baseline (``DOSREFDY <= 0``) plus Week-12 visit;
* point cloud per subject: ``(u, z) = (DOSREFDY / 117, log(max(LSLD, floor)))``
  with pooled z-score standardization and deterministic duplicate jitter;
* alpha-complex persistence (essential H0 class dropped), power-weighted
  silhouette (r = 3) on the frozen grid [0, 0.83], resolution 100;
* eligibility: >= 2 distinct visit days and >= 4 valid diameters.

Only ``DOSREFDY`` and ``LSLD`` enter the point cloud; response fields
(``LSRESP*``, ``LSSLD``) never touch the geometry and are used exclusively as
aggregate sensitivity outcomes downstream. The adapter is deterministic:
subjects are processed in sorted ``SUBJID`` order and every stochastic-free step
is a pure function of the input frames.

A non-confidential synthetic fixture (:func:`synthetic_pacce_frames`) mirrors the
schema subset used here so the tests in ``tests/test_pacce_adapter.py`` run
without the raw archive.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..silhouette import compute_silhouette, diagrams_from_pointcloud

__all__ = [
    "PACCEData",
    "SPEC_V1",
    "load_outcome_spec",
    "window_records",
    "subject_clouds",
    "compute_phi",
    "build_baseline_X",
    "load_pacce_from_frames",
    "load_pacce",
    "synthetic_pacce_frames",
]

# --------------------------------------------------------------------------
# frozen specification (mirrors data/pacce/outcome_spec.yaml v1.0)
# --------------------------------------------------------------------------
SPEC_V1 = {
    "spec_version": "pacce-outcome-v1.0-locked",
    "records": {"eval_type": "Central", "lesion_cat": "Target lesion"},
    "window": {"baseline_max_day": 0, "week12_visit": "Week 12", "week12_max_day": 117},
    "points": {"u_divisor": 117.0, "floor_mm": 5.0},
    "eligibility": {"min_distinct_visit_days": 2, "min_points": 4},
    "silhouette": {"r": 3.0, "resolution": 100, "interval": [0.0, 0.83]},
    "homology_dims": (0, 1),
}

BASELINE_COLS = ["chemo_iri", "age", "male", "ecog_ge1", "n_target_bl",
                 "n_metsites_bl", "kras_wt", "kras_mut", "kras_missing",
                 "ldh_high_bl"]

DEMO_REQUIRED = ["SUBJID", "TRT", "CHEMO", "AGE", "SEX", "B_ECOG", "B_LESNMT",
                 "B_METANM", "KRAS", "B_LDHYN"]
LESION_REQUIRED = ["SUBJID", "LSTYPE", "LSCAT", "DOSREFDY", "VISIT", "LSLD",
                   "LSNO"]


@dataclass
class PACCEData:
    """Subject-level PACCE analysis object.

    Attributes:
        frame: one row per eligible subject (topology summaries, SLD features,
            treatment labels).
        phi: silhouettes ``[n, n_hom, resolution]``.
        tseq: silhouette grid ``[resolution]``.
        A: assigned treatment (1 = panitumumab + bev + chemo).
        X: baseline design matrix (see :data:`BASELINE_COLS`).
        manifest: one row per released subject with inclusion status/reason.
        spec: the geometric specification used (with fitted standardization
            constants added under ``spec["standardization"]``).
        extras: scope metadata explaining selected-cohort eligibility and
            whether geometry scales were supplied or fitted to this cohort.
    """

    frame: pd.DataFrame
    phi: np.ndarray
    tseq: np.ndarray
    A: np.ndarray
    X: pd.DataFrame
    manifest: pd.DataFrame
    spec: dict
    extras: dict = field(default_factory=dict)


def load_outcome_spec(path=None) -> dict:
    """Load the frozen YAML spec; fall back to the built-in SPEC_V1."""
    if path is None:
        return json.loads(json.dumps(SPEC_V1))       # deep copy
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _check_schema(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


# --------------------------------------------------------------------------
# frozen construction steps
# --------------------------------------------------------------------------
def window_records(lesion: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Central target-lesion records inside the frozen window (valid diameters).

    Two frozen rules: ``rule="visit"`` (baseline days <= 0 plus Week-12-labelled
    visits up to day 117 -- primary) and ``rule="day"`` (all central target
    records with DOSREFDY <= ``window.day_max``, unscheduled visits kept;
    sensitivity variant only).
    """
    _check_schema(lesion, LESION_REQUIRED, "lesion")
    rec = spec["records"]
    win = spec["window"]
    ct = lesion[(lesion["LSTYPE"] == rec["eval_type"])
                & (lesion["LSCAT"] == rec["lesion_cat"])
                & lesion["LSLD"].notna() & lesion["DOSREFDY"].notna()]
    if win.get("rule", "visit") == "day":
        out = ct[ct["DOSREFDY"] <= win.get("day_max", 120)]
    else:
        out = ct[(ct["DOSREFDY"] <= win["baseline_max_day"])
                 | ((ct["VISIT"] == win["week12_visit"])
                    & (ct["DOSREFDY"] <= win["week12_max_day"]))]
    return out.sort_values(["SUBJID", "DOSREFDY", "LSNO"]).reset_index(drop=True)


def subject_clouds(records: pd.DataFrame, spec: dict):
    """Per-subject raw point clouds + eligibility metadata (sorted by SUBJID).

    Returns ``(clouds, meta, manifest)`` where ``manifest`` covers every subject
    present in ``records`` with its inclusion status and exclusion reason.
    """
    elig = spec["eligibility"]
    floor = spec["points"]["floor_mm"]
    divisor = spec["points"]["u_divisor"]
    clouds, meta, manifest = {}, [], []
    for sid, g in records.groupby("SUBJID", sort=True):
        days = g["DOSREFDY"].to_numpy(float)
        ld = g["LSLD"].to_numpy(float)
        n_visits = int(np.unique(days).size)
        if n_visits < elig["min_distinct_visit_days"]:
            manifest.append(dict(SUBJID=int(sid), status="excluded_few_visits",
                                 n_visits=n_visits, n_points=int(len(ld))))
            continue
        if len(ld) < elig["min_points"]:
            manifest.append(dict(SUBJID=int(sid), status="excluded_few_points",
                                 n_visits=n_visits, n_points=int(len(ld))))
            continue
        clouds[int(sid)] = np.column_stack([days / divisor,
                                            np.log(np.maximum(ld, floor))])
        bl = float(g.loc[g["DOSREFDY"] <= spec["window"]["baseline_max_day"],
                         "LSLD"].sum())
        w12 = float(g.loc[g["DOSREFDY"] > spec["window"]["baseline_max_day"],
                          "LSLD"].sum())
        meta.append(dict(SUBJID=int(sid), n_points=int(len(ld)), n_visits=n_visits,
                         bl_sld=bl, w12_sld=w12,
                         pct_change_sld=100.0 * (w12 - bl) / bl if bl > 0 else np.nan))
        manifest.append(dict(SUBJID=int(sid), status="included",
                             n_visits=n_visits, n_points=int(len(ld))))
    return (clouds,
            pd.DataFrame(meta, columns=["SUBJID", "n_points", "n_visits", "bl_sld",
                                        "w12_sld", "pct_change_sld"]),
            pd.DataFrame(manifest, columns=["SUBJID", "status", "n_visits", "n_points"]))


def _dedupe(P):
    seen, out = {}, P.copy()
    for i in range(len(P)):
        key = (round(P[i, 0], 9), round(P[i, 1], 9))
        k = seen.get(key, 0)
        if k > 0:
            out[i, 1] += 1e-6 * k
        seen[key] = k + 1
    return out


def compute_phi(clouds, spec):
    """Plain-tent curves ``{sid: [n_hom, res]}`` with specified standardization.

    Standardization constants (mean/sd of u and z over the pooled eligible
    cloud) are deterministic given the sorted input and recorded in the returned
    spec under ``"standardization"``. If the incoming spec already carries
    ``"standardization"`` (e.g. the frozen real-data constants used to make
    simulations PACCE-matched), those are applied and preserved.
    """
    spec = dict(spec)
    if spec.get("standardization") is not None:
        mu = np.asarray(spec["standardization"]["mean"], dtype=float)
        sd = np.asarray(spec["standardization"]["sd"], dtype=float)
        if (mu.shape != (2,) or sd.shape != (2,) or not np.isfinite(mu).all()
                or not np.isfinite(sd).all() or np.any(sd <= 0)):
            raise ValueError("fixed standardization requires finite length-2 means and positive sds")
        source = spec.get("standardization_source", "externally_fixed")
    elif not clouds:
        spec["standardization_source"] = "no_eligible_clouds"
        return {}, spec
    else:
        pool_u = np.concatenate([c[:, 0] for c in clouds.values()])
        pool_z = np.concatenate([c[:, 1] for c in clouds.values()])
        mu = np.array([pool_u.mean(), pool_z.mean()])
        sd = np.array([pool_u.std(), pool_z.std()])
        sd = np.where(sd == 0, 1.0, sd)
        source = "selected_analysis_cohort"
    spec["standardization"] = {"mean": mu.tolist(), "sd": sd.tolist()}
    spec["standardization_source"] = source

    r = spec["silhouette"]["r"]
    res = spec["silhouette"]["resolution"]
    interval = list(spec["silhouette"]["interval"])
    hom_dims = tuple(spec.get("homology_dims", (0, 1)))
    phis = {}
    for sid in sorted(clouds):
        P = _dedupe((clouds[sid] - mu) / sd)
        diags = diagrams_from_pointcloud(P, homology_dims=hom_dims)
        phi = np.asarray(compute_silhouette(diags, interval=interval, r=r,
                                            resolution=res,
                                            normalization=spec["silhouette"].get("normalization", "plain"),
                                            allow_experimental_r=spec["silhouette"].get("allow_experimental_r", False)))
        if not np.isfinite(phi).all():
            raise FloatingPointError(f"non-finite silhouette for subject {sid}")
        phis[int(sid)] = phi
    return phis, spec


def build_baseline_X(demo: pd.DataFrame) -> pd.DataFrame:
    """Frozen baseline design matrix (no post-treatment variables)."""
    _check_schema(demo, DEMO_REQUIRED, "demo")
    d = demo.sort_values("SUBJID")
    return pd.DataFrame({
        "SUBJID": d["SUBJID"].astype(int).values,
        "chemo_iri": (d["CHEMO"] == "Irinotecan").astype(float).values,
        "age": d["AGE"].astype(float).values,
        "male": (d["SEX"] == "Male").astype(float).values,
        "ecog_ge1": (d["B_ECOG"] == "Symptoms but ambulatory").astype(float).values,
        "n_target_bl": d["B_LESNMT"].astype(float).values,
        "n_metsites_bl": d["B_METANM"].astype(float).values,
        "kras_wt": (d["KRAS"] == "Wild-type").astype(float).values,
        "kras_mut": (d["KRAS"] == "Mutant").astype(float).values,
        "kras_missing": (d["KRAS"].isna() | (d["KRAS"] == "")).astype(float).values,
        "ldh_high_bl": (d["B_LDHYN"] == "Y").astype(float).values,
    })


# --------------------------------------------------------------------------
# top-level loaders
# --------------------------------------------------------------------------
def load_pacce_from_frames(demo, lesion, spec=None) -> PACCEData:
    """Selected-cohort exploratory arrays; causal identification is not established."""
    spec = spec if spec is not None else load_outcome_spec()
    _check_schema(demo, DEMO_REQUIRED, "demo")
    demo = demo.copy()
    demo["SUBJID"] = demo["SUBJID"].astype(int)
    records = window_records(lesion, spec)
    clouds, meta, manifest = subject_clouds(records, spec)
    phis, spec = compute_phi(clouds, spec)

    # every released subject appears in the manifest (auditability)
    seen = set(manifest["SUBJID"].tolist()) if len(manifest) else set()
    missing = sorted(set(demo["SUBJID"]) - seen)
    if missing:
        manifest = pd.concat(
            [manifest,
             pd.DataFrame(dict(SUBJID=missing, status="no_window_records",
                               n_visits=np.nan, n_points=np.nan))],
            ignore_index=True).sort_values("SUBJID").reset_index(drop=True)

    sids = sorted(phis)
    phi = (np.stack([phis[s] for s in sids]) if sids else
           np.empty((0, len(spec.get("homology_dims", (0, 1))),
                     spec["silhouette"]["resolution"])))
    interval = spec["silhouette"]["interval"]
    tseq = np.linspace(interval[0], interval[1], spec["silhouette"]["resolution"])

    Xb = build_baseline_X(demo)
    Xb = Xb[Xb["SUBJID"].isin(sids)].set_index("SUBJID").loc[sids]
    dd = demo.set_index(demo["SUBJID"].astype(int)).loc[sids]
    A = dd["TRT"].str.startswith("panit").astype(float).to_numpy()
    trt = dd["TRT"].to_numpy()
    chemo = dd["CHEMO"].to_numpy()

    summ_cols = [c for c in meta.columns if c != "SUBJID"]
    frame = pd.DataFrame({
        "SUBJID": sids,
        **{c: meta.set_index("SUBJID").loc[sids, c].values for c in summ_cols},
        "TRT": trt, "CHEMO": chemo, "A": A,
    }).reset_index(drop=True)

    return PACCEData(frame=frame, phi=phi, tseq=tseq, A=A, X=Xb.reset_index(drop=True),
                     manifest=manifest, spec=spec,
                     extras={
                         "analysis_scope": "selected-cohort descriptive/associational",
                         "causal_identification": "not established: eligibility uses treatment-affected measurements and visits",
                         "standardization_source": spec.get("standardization_source"),
                     })


def load_pacce(sas_dir, spec_path=None, limit_subjects=None) -> PACCEData:
    """Load the real release from an extracted SAS directory.

    Args:
        sas_dir: directory containing ``demo.sas7bdat`` / ``lesion.sas7bdat``.
        spec_path: optional YAML spec override (default: built-in SPEC_V1).
        limit_subjects: optional cap on released subjects (debugging only).
    """
    import pyreadstat

    sas_dir = Path(sas_dir)
    spec = load_outcome_spec(spec_path)

    def rd(name, cols):
        df, _ = pyreadstat.read_sas7bdat(str(sas_dir / f"{name}.sas7bdat"),
                                         usecols=cols)
        return df

    demo = rd("demo", DEMO_REQUIRED)
    lesion = rd("lesion", LESION_REQUIRED)
    if limit_subjects is not None:
        keep = set(sorted(demo["SUBJID"].astype(int).unique())[:limit_subjects])
        demo = demo[demo["SUBJID"].astype(int).isin(keep)]
        lesion = lesion[lesion["SUBJID"].astype(int).isin(keep)]
    return load_pacce_from_frames(demo, lesion, spec)


# --------------------------------------------------------------------------
# non-confidential synthetic fixture (tests only -- NOT the trial data)
# --------------------------------------------------------------------------
def synthetic_pacce_frames(n: int = 40, seed: int = 0):
    """Synthetic demo/lesion frames with the release's schema subset.

    Treatment accelerates lesion shrinkage so the full pipeline can be exercised
    end-to-end; ~15% of subjects miss the Week-12 visit and one subject carries
    too few lesions, to exercise the manifest exclusions.
    """
    rng = np.random.default_rng(seed)
    sids = np.arange(1000, 1000 + n)
    A = rng.binomial(1, 0.5, n)
    demo = pd.DataFrame({
        "SUBJID": sids,
        "TRT": np.where(A == 1, "panit. plus bevacizumab with chemotherapy",
                        "bevacizumab with chemotherapy"),
        "CHEMO": np.where(rng.random(n) < 0.2, "Irinotecan", "Oxaliplatin"),
        "AGE": rng.normal(61, 12, n).round(0),
        "SEX": np.where(rng.random(n) < 0.57, "Male", "Female"),
        "B_ECOG": np.where(rng.random(n) < 0.41, "Symptoms but ambulatory",
                           "Fully active"),
        "B_LESNMT": rng.integers(1, 6, n),
        "B_METANM": rng.integers(1, 4, n),
        "KRAS": rng.choice(["Wild-type", "Mutant", ""], n, p=[0.49, 0.34, 0.17]),
        "B_LDHYN": np.where(rng.random(n) < 0.2, "Y", "N"),
    })

    rows = []
    for i, sid in enumerate(sids):
        j_max = int(demo.loc[i, "B_LESNMT"])
        j_n = max(j_max - 1, 1) if i == 3 else j_max          # subj idx 3: few points
        base = rng.lognormal(mean=3.0, sigma=0.5, size=j_n)
        shrink = 0.25 + 0.20 * A[i]                            # arm-dependent effect
        miss = rng.random() < 0.15                             # missed week-12
        for j in range(j_n):
            rows.append(dict(SUBJID=sid, LSTYPE="Central",
                             LSCAT="Target lesion", LSNO=float(j + 1),
                             DOSREFDY=-6.0, VISIT="Screening",
                             LSLD=float(base[j]), LSRESP=""))
            if not miss:
                w12 = base[j] * np.exp(-shrink + rng.normal(0, 0.08))
                rows.append(dict(SUBJID=sid, LSTYPE="Central",
                                 LSCAT="Target lesion", LSNO=float(j + 1),
                                 DOSREFDY=81.0, VISIT="Week 12",
                                 LSLD=float(w12), LSRESP="PR"))
            # decoy: post-window record must be ignored
            rows.append(dict(SUBJID=sid, LSTYPE="Central",
                             LSCAT="Target lesion", LSNO=float(j + 1),
                             DOSREFDY=200.0, VISIT="Week 24",
                             LSLD=float(base[j]), LSRESP="PD"))
    lesion = pd.DataFrame(rows)
    return demo, lesion


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        data = load_pacce(sys.argv[1])
        h = hashlib.sha256(data.phi.tobytes()).hexdigest()
        print(json.dumps({"n": len(data.A), "phi_shape": list(data.phi.shape),
                          "sha256_phi": h}, indent=2))
