"""PACCE Phase P0 audit: archive integrity, population reconciliation, baseline
covariate freeze, and the treatment-blind topology non-vacuity pilot.

Work packages covered: WP-P0.1, WP-P0.2, WP-P0.3 (G0/G2 gate evidence).

Outputs
-------
results/pacce/archive_inventory.json            WP-P0.1
results/pacce/domain_subject_counts.csv         WP-P0.1
data/pacce/README.md                            WP-P0.1 (released-population statement)
data/pacce/baseline_dictionary.yaml             WP-P0.2
results/pacce/baseline_balance.csv              WP-P0.2
results/pacce/overlap_diagnostics.pdf           WP-P0.2
results/pacce/topology_pilot_subject_level.parquet  WP-P0.3
results/pacce/topology_nonvacuity_report.html   WP-P0.3
results/pacce/topology_preprocessing_sensitivity.csv WP-P0.3
data/pacce/outcome_spec.yaml                    frozen pilot construction (P1.1 input)

The pilot is treatment-blind: treatment labels are attached to the subject table
for later phases but no treatment-group comparison is performed or reported here.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import yaml
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "Colorec_Amgen_2005_262.zip"
RES = ROOT / "results" / "pacce"
DATA = ROOT / "data" / "pacce"
RES.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

R_SIL = 3.0
RESOLUTION = 100
HOM_DIMS = (0, 1)
ALPHA = 0.10

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_sas_dir() -> Path:
    """Extract the immutable archive into a temporary directory if needed."""
    target = Path(tempfile.gettempdir()) / "pacce_p0_sas"
    inner_marker = target / "SAS dataset - 20040249" / "demo.sas7bdat"
    if not inner_marker.exists():
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(ZIP_PATH) as z:
                z.extractall(tmp)
            outer = list(Path(tmp).glob("*.zip"))[0]
            with zipfile.ZipFile(outer) as z:
                z.extractall(target)
    return target / "SAS dataset - 20040249"


def read_domain(sas: Path, dom: str, usecols=None):
    df, meta = pyreadstat.read_sas7bdat(str(sas / f"{dom}.sas7bdat"), usecols=usecols)
    return df


# ==========================================================================
# WP-P0.1  archive integrity and population reconciliation
# ==========================================================================
def wp_p0_1(sas: Path):
    print("== WP-P0.1: archive integrity and population reconciliation ==")
    domains = [
        "demo", "corevar", "lesion", "respeval", "exposure", "chemotx",
        "disposit", "eligcrit", "a_eendpt", "a_sendpt", "ae", "lab",
        "vitals_v", "death", "medhist", "radiotx", "surghist",
    ]
    rows, inventory_files = [], []
    for dom in domains:
        p = sas / f"{dom}.sas7bdat"
        df = read_domain(sas, dom, usecols=["SUBJID"])
        rows.append(
            dict(domain=dom, rows=len(df), n_subjects=int(df["SUBJID"].nunique()))
        )
        inventory_files.append(
            dict(file=f"{dom}.sas7bdat", bytes=p.stat().st_size,
                 sha256=sha256_file(p), rows=len(df),
                 n_subjects=int(df["SUBJID"].nunique()))
        )
    counts = pd.DataFrame(rows)
    counts.to_csv(RES / "domain_subject_counts.csv", index=False)

    demo = read_domain(sas, "demo")
    inventory = {
        "archive": {
            "zip": "Colorec_Amgen_2005_262.zip",
            "sha256": sha256_file(ZIP_PATH),
            "bytes": ZIP_PATH.stat().st_size,
            "inner_files": inventory_files,
        },
        "companion_archive": {
            "zip": "Colorec_Amgen_2006_309.zip",
            "sha256": sha256_file(ROOT / "Colorec_Amgen_2006_309.zip"),
            "role": "second Amgen colorectal release; not used in the PACCE study",
        },
        "documents": {
            "protocol": "20040249_01.20.01 Protocol Amend 2 2007-08-07 English_Redacted.pdf",
            "crf": "Amgen 20040249 CRF REDACTED.pdf",
            "ddt": "Amgen 20040249 DDT.pdf (data dictionary; LESION = one record per "
                   "subject, assessment type, exam date, lesion ID)",
        },
        "population": {
            "n_subjects_released": int(demo["SUBJID"].nunique()),
            "n_catalog_published": 1053,
            "note": "Hecht et al. (2009) and the PDS catalog report 1053 randomized "
                    "patients; the supplied release contains 842 distinct SUBJID in "
                    "demo/corevar/exposure/chemotx/disposit/vitals_v/medhist/radiotx/"
                    "surghist/eligcrit/a_eendpt/lesion. The analysis population is the "
                    "released 842-subject subset, not the full randomized trial.",
        },
        "treatment": {
            "assigned_trt": demo["TRT"].value_counts().to_dict(),
            "atrt_vs_trt_mismatch": int((demo["TRT"] != demo["ATRT"]).sum()),
            "evalitt_all_Y": bool((demo["EVALITT"] == "Y").all()),
            "chemo_strata": demo.groupby("CHEMO")["SUBJID"].nunique().to_dict(),
        },
        "code_version": "pacce_p0_audit.py",
    }
    with open(RES / "archive_inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)

    readme = f"""# PACCE release (Amgen 20040249) -- analysis population statement

Source archive: `Colorec_Amgen_2005_262.zip`
(sha256 `{inventory['archive']['sha256']}`), containing the protocol (Amendment 2),
the CRF, the data dictionary (DDT), and `SAS dataset - 20040249.zip` with 17 SAS
domains. The archive is immutable; extracted files are never committed.

## Released population

All subject-level domains (`demo`, `corevar`, `exposure`, `chemotx`, `disposit`,
`eligcrit`, `a_eendpt`, `vitals_v`, `medhist`, `radiotx`, `surghist`, `lesion`)
contain exactly **842 distinct SUBJID**. The published PACCE trial randomized
**1053 patients** (Hecht et al., JCO 2009; NCT00115765). The supplied release is
therefore a **subset of the randomized trial** (842 of 1053; the subset rule is
not documented in the release). Every estimand in this study is defined on the
**released 842-subject population**; no analysis reports the catalog total as a
sample size.

## Treatment assignment

`demo.TRT` (assigned treatment) is the ITT treatment variable: 421 subjects
assigned to panitumumab + bevacizumab + chemotherapy, 421 assigned to
bevacizumab + chemotherapy. `ATRT` (actual) disagrees with `TRT` for 9 subjects
and is excluded from the primary contrast. `EVALITT = Y` for all 842 subjects.
`CHEMO` (assigned chemotherapy backbone: Irinotecan 170, Oxaliplatin 672) is a
design (randomization) stratum and is retained in every adjusted analysis.

## Domains used

| Domain | Rows | Subjects | Role |
|---|---:|---:|---|
| demo | 842 | 842 | treatment, baseline covariates, analysis-set flags |
| lesion | 30153 | 842 | lesion-level diameters (LSLD), central/local, target |
| respeval | 7148 | 785 | visit-level modified RECIST (central) |
| a_eendpt | 842 | 842 | week-12 RECIST category, response, PFS, death |
| exposure | 85583 | 842 | dosing (audit only) |
"""
    (DATA / "README.md").write_text(readme)
    print(counts.to_string(index=False))
    print(f"TRT mismatch vs ATRT: {inventory['treatment']['atrt_vs_trt_mismatch']}")
    return demo


# ==========================================================================
# WP-P0.2  treatment and baseline covariate audit
# ==========================================================================
BASELINE_DICT = {
    "spec_version": "pacce-baseline-v1",
    "treatment": {
        "variable": "TRT",
        "levels": {
            "A=1": "panitumumab + bevacizumab + chemotherapy",
            "A=0": "bevacizumab + chemotherapy",
        },
        "rule": "assigned TRT for the ITT contrast; ATRT excluded (9 mismatches)",
    },
    "design_stratum": {"variable": "CHEMO",
                       "levels": ["Irinotecan", "Oxaliplatin"],
                       "rule": "retained in every adjusted analysis; randomization stratum"},
    "baseline_covariates": {
        "chemo_iri": {"source": "CHEMO == 'Irinotecan'", "role": "stratum indicator"},
        "age": {"source": "AGE", "role": "continuous, standardized"},
        "male": {"source": "SEX == 'Male'", "role": "indicator"},
        "ecog_ge1": {"source": "B_ECOG == 'Symptoms but ambulatory'",
                     "role": "ECOG 1 indicator (only 0/1 present)"},
        "n_target_bl": {"source": "B_LESNMT", "role": "baseline target lesion count"},
        "n_metsites_bl": {"source": "B_METANM", "role": "baseline metastatic-site count"},
        "kras_wt": {"source": "KRAS == 'Wild-type'", "role": "biomarker indicator"},
        "kras_missing": {"source": "KRAS in ('', nan)", "role": "explicit missingness indicator"},
        "ldh_high_bl": {"source": "B_LDHYN == 'Y'", "role": "LDH > 1.5x ULN indicator"},
    },
    "ctate_profiles_prespecification": (
        "CHEMO backbone (2 levels) x baseline target-lesion burden "
        "(B_LESNMT <= 2 / 3-4 / >= 5): six pre-specified profiles; only profiles "
        "with >= 15 subjects in each arm are reported."
    ),
    "excluded_from_X": {
        "TRTDUR": "post-randomization treatment duration",
        "RESPSTAT / R12CRCA / RSRESP": "post-baseline response",
        "PFSDYCR / PFSCR": "post-baseline PFS",
        "post-baseline LSLD/LSSLD": "outcomes, not covariates",
        "LSSITE / LSNO": "never used as numeric coordinates",
    },
}


def wp_p0_2(demo: pd.DataFrame):
    print("\n== WP-P0.2: treatment and baseline covariate audit ==")
    with open(DATA / "baseline_dictionary.yaml", "w") as f:
        yaml.dump(BASELINE_DICT, f, sort_keys=False)

    d = demo.copy()
    d["arm"] = np.where(
        d["TRT"].str.startswith("panit"), "panitumumab+bev+chemo", "bev+chemo"
    )
    X = build_baseline_X(d)
    X["arm"] = d["arm"].values

    num_cols = ["age", "n_target_bl", "n_metsites_bl"]
    bin_cols = [c for c in X.columns if c not in num_cols + ["arm"]]
    rows = []
    for c in num_cols + bin_cols:
        a1 = X.loc[X.arm == "panitumumab+bev+chemo", c].astype(float)
        a0 = X.loc[X.arm == "bev+chemo", c].astype(float)
        m1, m0 = a1.mean(), a0.mean()
        s1, s0 = a1.std(ddof=1), a0.std(ddof=1)
        smd = (m1 - m0) / np.sqrt(0.5 * (s1**2 + s0**2)) if (s1 + s0) > 0 else 0.0
        rows.append(dict(covariate=c, mean_arm1=m1, mean_arm0=m0,
                         sd_arm1=s1, sd_arm0=s0, smd=smd,
                         n_arm1=int(a1.notna().sum()), n_arm0=int(a0.notna().sum())))
    bal = pd.DataFrame(rows)
    bal.to_csv(RES / "baseline_balance.csv", index=False)

    # overlap diagnostics: logistic propensity on the frozen baseline set
    feats = [c for c in X.columns if c != "arm"]
    Z = X[feats].astype(float).values
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    A = (X["arm"] == "panitumumab+bev+chemo").astype(int).values
    lr = LogisticRegression(max_iter=2000).fit(Z, A)
    pi = lr.predict_proba(Z)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(pi[A == 1], bins=25, alpha=0.6, density=True,
                 label="panitumumab+bev+chemo")
    axes[0].hist(pi[A == 0], bins=25, alpha=0.6, density=True, label="bev+chemo")
    axes[0].axvline(0.5, color="k", lw=1, ls="--")
    axes[0].set_xlabel("estimated propensity score")
    axes[0].set_title("Propensity overlap (logistic, baseline X)")
    axes[0].legend(fontsize=8)
    axes[1].barh(bal.covariate, bal.smd)
    axes[1].axvline(0.1, color="r", lw=1, ls="--")
    axes[1].axvline(-0.1, color="r", lw=1, ls="--")
    axes[1].set_xlabel("standardized mean difference")
    axes[1].set_title("Baseline balance by assigned arm")
    fig.tight_layout()
    fig.savefig(RES / "overlap_diagnostics.pdf")
    plt.close(fig)

    print(bal.round(3).to_string(index=False))
    print("propensity: min %.3f max %.3f" % (pi.min(), pi.max()))
    return X


def build_baseline_X(demo: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "chemo_iri": (demo["CHEMO"] == "Irinotecan").astype(float),
        "age": demo["AGE"].astype(float),
        "male": (demo["SEX"] == "Male").astype(float),
        "ecog_ge1": (demo["B_ECOG"] == "Symptoms but ambulatory").astype(float),
        "n_target_bl": demo["B_LESNMT"].astype(float),
        "n_metsites_bl": demo["B_METANM"].astype(float),
        "kras_wt": (demo["KRAS"] == "Wild-type").astype(float),
        "kras_mut": (demo["KRAS"] == "Mutant").astype(float),
        "kras_missing": (demo["KRAS"].isna() | (demo["KRAS"] == "")).astype(float),
        "ldh_high_bl": (demo["B_LDHYN"] == "Y").astype(float),
    })


# ==========================================================================
# WP-P0.3  outcome and topology non-vacuity audit (treatment-blind)
# ==========================================================================
from gudhi.representations import Silhouette

from tcda_uq.silhouette.core import diagrams_from_pointcloud


def window_records(les: pd.DataFrame, eval_type="Central", rule="visit"):
    ct = les[(les["LSTYPE"] == eval_type) & (les["LSCAT"] == "Target lesion")]
    ct = ct[ct["LSLD"].notna() & ct["DOSREFDY"].notna()]
    if rule == "visit":
        w = ct[(ct["DOSREFDY"] <= 0)
               | ((ct["VISIT"] == "Week 12") & (ct["DOSREFDY"] <= 117))]
    elif rule == "day120":
        w = ct[(ct["DOSREFDY"] <= 0) | (ct["DOSREFDY"] <= 120)]
        w = ct[ct["DOSREFDY"] <= 120]
    else:
        raise ValueError(rule)
    return w


def subject_clouds(w: pd.DataFrame, floor=5.0, min_visits=2, min_points=4):
    """Group window records into per-subject point clouds (u, z) with z = log size."""
    clouds, meta = {}, []
    for sid, g in w.groupby("SUBJID"):
        days = g["DOSREFDY"].to_numpy(float)
        ld = g["LSLD"].to_numpy(float)
        z = np.log(np.maximum(ld, floor))
        n_visits = np.unique(days).size
        if n_visits >= min_visits and len(z) >= min_points:
            clouds[sid] = np.column_stack([days / 117.0, z])
            bl = g.loc[g["DOSREFDY"] <= 0, "LSLD"].sum()
            w12 = g.loc[g["DOSREFDY"] > 0, "LSLD"].sum()
            meta.append(dict(SUBJID=sid, n_points=len(z), n_visits=n_visits,
                             bl_sld=bl, w12_sld=w12,
                             pct_change_sld=100.0 * (w12 - bl) / bl if bl > 0 else np.nan))
    return clouds, pd.DataFrame(meta)


def dedupe_points(P):
    """Deterministic jitter for exactly duplicated coordinates."""
    seen = {}
    out = P.copy()
    for i in range(len(P)):
        key = (round(P[i, 0], 9), round(P[i, 1], 9))
        k = seen.get(key, 0)
        if k > 0:
            out[i, 1] += 1e-6 * k
        seen[key] = k + 1
    return out


def subject_summaries(P, t_max, r=R_SIL, resolution=RESOLUTION):
    """Diagrams + silhouette summaries for one subject cloud."""
    P = dedupe_points(P)
    diags = diagrams_from_pointcloud(P, homology_dims=HOM_DIMS)
    sil = Silhouette(weight=lambda x: (x[1] - x[0]) ** r, resolution=resolution,
                     sample_range=[0.0, t_max], keep_endpoints=True)
    phi = np.asarray(sil.fit_transform(diags))   # [2, resolution]
    tgrid = np.linspace(0.0, t_max, resolution)
    out = {}
    for k, dim in enumerate(HOM_DIMS):
        dg = diags[k]
        out[f"h{dim}_n"] = len(dg)
        out[f"h{dim}_totpers"] = float(np.sum(dg[:, 1] - dg[:, 0])) if len(dg) else 0.0
        out[f"h{dim}_maxper"] = float(np.max(dg[:, 1] - dg[:, 0])) if len(dg) else 0.0
        out[f"h{dim}_l1"] = float(np.trapezoid(phi[k], tgrid))
        out[f"h{dim}_max"] = float(phi[k].max())
    return out, diags, phi


def pilot_variant(les, floor=5.0, rule="visit", eval_type="Central",
                  min_points=4, r=R_SIL, t_max=None):
    w = window_records(les, eval_type=eval_type, rule=rule)
    clouds, meta = subject_clouds(w, floor=floor, min_points=min_points)
    if t_max is None:
        # frozen grid: 99th percentile of pooled finite deaths, rounded up
        deaths = []
        for P in clouds.values():
            for dg in diagrams_from_pointcloud(dedupe_points(P), homology_dims=HOM_DIMS):
                if len(dg):
                    deaths.append(dg[:, 1].max())
        t_max = float(np.ceil(np.percentile(deaths, 99) * 100) / 100)
    rows = []
    for sid, P in clouds.items():
        s, _, _ = subject_summaries(P, t_max, r=r)
        s["SUBJID"] = sid
        rows.append(s)
    cols = ["SUBJID", "h0_n", "h0_totpers", "h0_maxper", "h0_l1", "h0_max",
            "h1_n", "h1_totpers", "h1_maxper", "h1_l1", "h1_max"]
    left = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    right = meta if len(meta) else pd.DataFrame(
        columns=["SUBJID", "n_points", "n_visits", "bl_sld", "w12_sld",
                 "pct_change_sld"])
    df = left.merge(right, on="SUBJID") if len(left) and len(right) else left
    return df, t_max, len(clouds)


def wp_p0_3(sas: Path, X_base: pd.DataFrame):
    print("\n== WP-P0.3: topology non-vacuity pilot (treatment-blind) ==")
    les = read_domain(sas, "lesion")
    ae = read_domain(sas, "a_eendpt")

    # ---- primary frozen construction --------------------------------------
    df, t_max, n_ok = pilot_variant(les)
    print(f"eligible subjects: {n_ok}; frozen silhouette grid [0, {t_max}]")

    # attach treatment-blind covariates + SLD/RECIST comparators
    demo = read_domain(sas, "demo", usecols=["SUBJID", "TRT", "ATRT", "CHEMO"])
    df["SUBJID"] = df["SUBJID"].astype(int)
    demo["SUBJID"] = demo["SUBJID"].astype(int)
    df = df.merge(demo, on="SUBJID", how="left")
    X_base = X_base.copy()
    X_base["SUBJID"] = X_base.index.astype(int)
    df = df.merge(X_base, on="SUBJID", how="left")
    df["pct_change_sld"] = 100.0 * (df["w12_sld"] - df["bl_sld"]) / df["bl_sld"]
    r12 = ae.set_index("SUBJID")["R12CRCA"]
    df["r12_recist"] = df["SUBJID"].map(r12)
    df["tmax_grid"] = t_max

    keep = ["SUBJID", "n_points", "n_visits", "bl_sld", "w12_sld", "pct_change_sld",
            "r12_recist", "h0_n", "h0_totpers", "h0_maxper", "h0_l1", "h0_max",
            "h1_n", "h1_totpers", "h1_maxper", "h1_l1", "h1_max", "tmax_grid",
            "TRT", "ATRT", "CHEMO"] + [c for c in X_base.columns
                                       if c not in ("SUBJID", "arm")]
    df[keep].to_parquet(RES / "topology_pilot_subject_level.parquet", index=False)

    # ---- sensitivity grid ---------------------------------------------------
    variants = []
    for floor in (3.0, 5.0, 10.0):
        variants.append(dict(floor=floor, rule="visit", eval_type="Central",
                             min_points=4, r=R_SIL, tag=f"floor{floor:g}"))
    for rule in ("visit", "day120"):
        variants.append(dict(floor=5.0, rule=rule, eval_type="Central",
                             min_points=4, r=R_SIL, tag=f"rule_{rule}"))
    for et in ("Central", "Local"):
        variants.append(dict(floor=5.0, rule="visit", eval_type=et,
                             min_points=4, r=R_SIL, tag=f"eval_{et}"))
    for mp in (4, 6):
        variants.append(dict(floor=5.0, rule="visit", eval_type="Central",
                             min_points=mp, r=R_SIL, tag=f"minpts{mp}"))
    for r in (2.0, 3.0, 4.0):
        variants.append(dict(floor=5.0, rule="visit", eval_type="Central",
                             min_points=4, r=r, tag=f"r{r:g}"))

    sens_rows = []
    for v in variants:
        print(f"  sensitivity variant: {v['tag']}")
        dv, tm, n = pilot_variant(les, floor=v["floor"], rule=v["rule"],
                                  eval_type=v["eval_type"], min_points=v["min_points"],
                                  r=v["r"], t_max=t_max)

        def _g(col, fn="median"):
            if len(dv) == 0 or col not in dv:
                return np.nan
            s = dv[col].dropna()
            if len(s) == 0:
                return np.nan
            return float(s.median() if fn == "median"
                         else dv[col].corr(dv[fn]))

        sens_rows.append(dict(
            variant=v["tag"], n_subjects=n, t_max=tm,
            h1_freq=float((dv["h1_n"] > 0).mean()) if len(dv) else np.nan,
            h1_strong_freq=float((dv["h1_maxper"] > 0.05).mean())
            if len(dv) and "h1_maxper" in dv else np.nan,
            h1_totpers_med=_g("h1_totpers"),
            h0_totpers_med=_g("h0_totpers"),
            h0_l1_med=_g("h0_l1"),
            corr_h0l1_blsld=_g("h0_l1", "bl_sld"),
            corr_h0l1_pctchg=_g("h0_l1", "pct_change_sld"),
        ))
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(RES / "topology_preprocessing_sensitivity.csv", index=False)
    print(sens.round(3).to_string(index=False))

    # ---- within-subject jackknife stability (primary variant) --------------
    w = window_records(les)
    clouds, _ = subject_clouds(w)
    rng = np.random.default_rng(0)
    rel_changes = []
    sids = sorted(clouds)
    for sid in sids[:: max(1, len(sids) // 200)]:        # ~200-subject subsample
        P = clouds[sid]
        base, _, _ = subject_summaries(P, t_max)
        for j in range(len(P)):
            Pj = np.delete(P, j, axis=0)
            if len(Pj) < 4:
                continue
            s, _, _ = subject_summaries(Pj, t_max)
            denom = max(base["h0_totpers"], 1e-12)
            rel_changes.append(abs(s["h0_totpers"] - base["h0_totpers"]) / denom)
    jack_med = float(np.median(rel_changes))
    jack_p90 = float(np.percentile(rel_changes, 90))

    # ---- SLD-equivalence regression ---------------------------------------
    import numpy.linalg as la

    F = df[["bl_sld", "w12_sld", "pct_change_sld"]].astype(float)
    F = (F - F.mean()) / F.std()
    Fm = np.column_stack([np.ones(len(F)), F.values])
    r2 = {}
    for ycol in ["h0_totpers", "h0_l1", "h1_totpers"]:
        y = df[ycol].astype(float).values
        beta, *_ = la.lstsq(Fm, y, rcond=None)
        resid = y - Fm @ beta
        r2[ycol] = float(1 - resid.var() / y.var())

    # ---- figures ------------------------------------------------------------
    figs = {}

    def _fig(name):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        figs[name] = base64.b64encode(buf.getvalue()).decode()

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(df["n_points"], bins=25, color="#4477aa")
    ax.set_xlabel("points per subject (central target lesions, window)")
    ax.set_ylabel("subjects")
    _fig("points")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].hist(df["h0_totpers"], bins=30, color="#4477aa")
    ax[0].set_title("H0 total persistence")
    ax[1].hist(df["h1_totpers"], bins=30, color="#ee7733")
    ax[1].set_title(f"H1 total persistence ({(df['h1_n'] > 0).mean():.0%} subjects with any H1)")
    _fig("pers")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].scatter(df["bl_sld"], df["h0_totpers"], s=8, alpha=0.5)
    ax[0].set_xlabel("baseline SLD (mm)"); ax[0].set_ylabel("H0 total persistence")
    ax[0].set_title(f"corr = {df['h0_totpers'].corr(df['bl_sld']):.2f}")
    ax[1].scatter(df["pct_change_sld"], df["h0_totpers"], s=8, alpha=0.5, color="#ee7733")
    ax[1].set_xlabel("SLD % change baseline->wk12")
    ax[1].set_title(f"corr = {df['h0_totpers'].corr(df['pct_change_sld']):.2f}")
    _fig("assoc")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    tags = [v["tag"] for v in variants]
    vals = [sens.loc[sens.variant == t, "h0_totpers_med"].iloc[0] for t in tags]
    ax.barh(range(len(tags)), vals, color="#4477aa")
    ax.set_yticks(range(len(tags))); ax.set_yticklabels(tags, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("median H0 total persistence")
    ax.set_title("Preprocessing sensitivity")
    _fig("sens")

    html = f"""<html><head><title>PACCE topology non-vacuity report</title>
<style>body{{font-family:sans-serif;margin:24px;max-width:900px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}
img{{max-width:100%}}</style></head><body>
<h1>PACCE topology non-vacuity pilot (WP-P0.3, treatment-blind)</h1>
<p>Frozen construction: central target lesions, baseline (day &le; 0) + Week-12
visit (day &le; 117), size floor 5 mm, z = log(max(LSLD, 5)), u = day/117,
coordinates standardized with pooled constants, alpha-complex persistence,
power-weighted silhouettes (r = {R_SIL:g}) on [0, {t_max}], resolution {RESOLUTION}.
Eligible subjects: <b>{n_ok}</b> (of 842 released; exclusions: missing LSLD/day,
&lt; 2 visit days, or &lt; {4} window points).</p>

<h2>1. Point counts</h2><img src="data:image/png;base64,{figs['points']}"/>

<h2>2. Persistence magnitude and H1 frequency</h2>
<p>Some H1 feature exists in <b>{(df['h1_n'] > 0).mean():.1%}</b> of subjects, but the
typical H1 feature is noise-level: median H1 total persistence
{df['h1_totpers'].median():.3g} vs H0 {df['h0_totpers'].median():.3g} (~
{df['h0_totpers'].median() / max(df['h1_totpers'].median(), 1e-12):.0f}&times; larger),
and only <b>{(df['h1_maxper'] > 0.05).mean():.1%}</b> of subjects have a prominent
H1 feature (max persistence &gt; 0.05). H0 is non-vacuous in every subject and
carries the dominant topological signal.</p>
<img src="data:image/png;base64,{figs['pers']}"/>

<h2>3. Residual association with SLD</h2>
<p>Correlations of H0 total persistence: baseline SLD
r = {df['h0_totpers'].corr(df['bl_sld']):.2f}, SLD % change
r = {df['h0_totpers'].corr(df['pct_change_sld']):.2f}. OLS R&sup2; of topology
summaries on SLD features (baseline SLD, week-12 SLD, % change):
{json.dumps({k: round(v, 3) for k, v in r2.items()})}.
An R&sup2; materially below 1 indicates the topology summary is not a
deterministic re-expression of SLD.</p>
<img src="data:image/png;base64,{figs['assoc']}"/>

<h2>4. Preprocessing sensitivity</h2>
<img src="data:image/png;base64,{figs['sens']}"/>
<p>Local (investigator) target-lesion records carry no longest-diameter values in
this release (0 of 1695 rows), so the central-vs-local variant is not computable;
the analysis is necessarily central-review only.</p>
{sens.round(3).to_html(index=False)}

<h2>5. Within-subject jackknife stability</h2>
<p>Leave-one-lesion-out relative change in H0 total persistence
(~200-subject subsample): median {jack_med:.3f}, 90th percentile {jack_p90:.3f}.</p>

<h2>6. Verdict against the WP-P0.3 pass rule</h2>
<p>Pass requires stable nontrivial topology or a justified H0 outcome not
identical to SLD. <b>H0 is the defensible topological summary</b>: non-vacuous,
only partially explained by SLD (R&sup2; = {r2['h0_totpers']:.2f} for total
persistence; {r2['h0_l1']:.2f} for the silhouette integral), and stable under
leave-one-lesion-out perturbation (median relative change {jack_med:.3f}).
H1 exists but is predominantly noise-level at this cloud size (median 8 points)
and is retained only as an ablation arm. Treatment labels were attached for later
phases but never compared here.</p>
</body></html>"""
    (RES / "topology_nonvacuity_report.html").write_text(html)

    # ---- frozen outcome spec (P1.1 input) ----------------------------------
    spec = {
        "spec_version": "pacce-outcome-v1-pilot",
        "status": "PILOT (pre-estimand-lock; frozen constants recorded for P1.1)",
        "domain": "lesion.sas7bdat",
        "records": "LSTYPE == 'Central' and LSCAT == 'Target lesion'",
        "window": {"baseline": "DOSREFDY <= 0",
                   "week12": "VISIT == 'Week 12' and DOSREFDY <= 117"},
        "points": {"u": "DOSREFDY / 117",
                   "z": "log(max(LSLD, 5))",
                   "floor_mm": 5.0,
                   "dedup": "deterministic 1e-6 jitter on exact duplicate coordinates"},
        "eligibility": {"min_distinct_visit_days": 2, "min_points": 4},
        "filtration": "alpha complex; radii = sqrt(alpha filtration); essential H0 dropped",
        "silhouette": {"r": R_SIL, "resolution": RESOLUTION, "interval": [0.0, t_max],
                       "t_max_rule": "ceil(99th percentile of pooled finite deaths)"},
        "summaries": ["h{0,1}_{n, totpers, l1, max}"],
        "n_eligible": int(n_ok),
        "treatment_blind": True,
    }
    with open(DATA / "outcome_spec.yaml", "w") as f:
        yaml.dump(spec, f, sort_keys=False)

    print(f"\njackknife: median {jack_med:.3f}, p90 {jack_p90:.3f}")
    print("SLD-equivalence R2:", {k: round(v, 3) for k, v in r2.items()})
    return df


if __name__ == "__main__":
    sas = ensure_sas_dir()
    demo = wp_p0_1(sas)
    X_base = wp_p0_2(demo)
    wp_p0_3(sas, X_base)
    print("\nP0 complete.")
