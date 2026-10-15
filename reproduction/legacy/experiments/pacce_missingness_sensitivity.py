"""PACCE Phase P4, WP-P4.1: missingness sensitivity (IPCW + MNAR tilt).

Re-expresses the frozen TATE contrast under (a) IPCW identification of the
full-population curve under MAR given (X, A), and (b) a deterministic MNAR
tilt family on the eligibility odds per unit of baseline burden.

Output: results/pacce/missingness_theory_checks.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from experiments.pacce_p0_audit import DATA, RES, ensure_sas_dir
from experiments.pacce_p3_controls import _frames
from tcda_uq.datasets.pacce import BASELINE_COLS, SPEC_V1, load_pacce_from_frames


def main():
    sas = ensure_sas_dir()
    demo, lesion = _frames(sas)
    data = load_pacce_from_frames(demo.copy(), lesion.copy(),
                                  load_outcome_spec())
    phi = data.phi[:, 0, :]
    A = data.A.astype(int)
    X = data.X[BASELINE_COLS].to_numpy(float)
    tseq = np.linspace(0.0, 0.83, 100)

    # eligibility indicator over the full release: R=1 for adapter-included
    # subjects; rebuild R on the released frame via the manifest
    manifest = pd.read_csv(RES / "processed_subject_manifest.csv")
    elig_ids = set(manifest.loc[manifest.status == "included", "SUBJID"].astype(int))
    all_ids = demo["SUBJID"].astype(int).to_numpy()

    # full-release design: baseline X for everyone; phi only where eligible.
    from tcda_uq.datasets.pacce import build_baseline_X

    Xall = build_baseline_X(demo)
    Xall = Xall.set_index("SUBJID").loc[all_ids].reset_index(drop=True)
    R = np.isin(all_ids, list(elig_ids)).astype(int)

    # arm vector on the full release
    A_all = (~demo["TRT"].str.startswith("panit")).to_numpy().astype(int)
    order = {sid: k for k, sid in enumerate(all_ids)}
    pos = np.array([order[s] for s in sorted(elig_ids)])
    phi_full = np.full((len(all_ids), phi.shape[1]), np.nan)
    id_elig = sorted(elig_ids)
    phi_full[pos] = phi[[list(id_elig).index(s) for s in all_ids[pos]]]

    burden_z = (Xall["n_target_bl"].values - Xall["n_target_bl"].mean()) \
        / Xall["n_target_bl"].std()
    Z = np.column_stack([np.ones(len(R)), Xall.values, A_all])
    lr = LogisticRegression(C=1.0, max_iter=4000).fit(Z, R)
    logit_R = lr.decision_function(Z)

    def hajek_contrast(weights):
        wA = weights * A_all * R
        wB = weights * (1 - A_all) * R
        m1 = np.nan_to_num(phi_full * (wA / wA.sum())[:, None]).sum(axis=0)
        m0 = np.nan_to_num(phi_full * (wB / wB.sum())[:, None]).sum(axis=0)
        return m1 - m0

    rows = []
    for gamma in (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75):
        if gamma == 0.0:
            w = R / np.clip(lr.predict_proba(Z)[:, 1], 0.02, 0.98)
            tag = "IPCW_MAR"
        else:
            pi_g = 1 / (1 + np.exp(-(logit_R + gamma * burden_z)))
            w = R / np.clip(pi_g, 0.02, 0.98)
            tag = f"MNAR_tilt_{gamma:+.2f}"
        psi = hajek_contrast(w)
        rows.append(dict(check=tag, gamma=gamma,
                         peak_abs_effect=float(np.abs(psi).max()),
                         peak_t=float(tseq[np.argmax(np.abs(psi))]),
                         sup_abs=float(np.abs(psi).max()),
                         n_weighted=int((w > 0).sum())))

    # unweighted eligible-population reference (the primary target)
    psi_ref = (phi[A == 1] - 0)  # placeholder to compute below
    m1 = phi[A == 1].mean(axis=0)
    m0 = phi[A == 0].mean(axis=0)
    rows.append(dict(check="observed_trajectory_primary", gamma=np.nan,
                     peak_abs_effect=float(np.abs(m1 - m0).max()),
                     peak_t=float(tseq[np.argmax(np.abs(m1 - m0))]),
                     sup_abs=float(np.abs(m1 - m0).max()), n_weighted=int(A.sum() * 2)))

    out = pd.DataFrame(rows)
    out.to_csv(RES / "missingness_theory_checks.csv", index=False)
    print(out.round(4).to_string(index=False))


def load_outcome_spec():
    import yaml

    with open(DATA / "outcome_spec.yaml") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
