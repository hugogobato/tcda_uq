"""PACCE Phase P1, WP-P1.2: deterministic adapter smoke run on the real release.

Outputs
-------
results/pacce/processed_subject_manifest.csv
results/pacce/pacce_smoke.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments.pacce_p0_audit import DATA, RES, ensure_sas_dir
from tcda_uq.datasets.pacce import load_outcome_spec, load_pacce


def spec_hash(spec) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def main():
    sas = ensure_sas_dir()
    spec = load_outcome_spec(DATA / "outcome_spec.yaml")
    data = load_pacce(sas, spec_path=DATA / "outcome_spec.yaml")

    manifest = data.manifest.copy()
    manifest["spec_version"] = data.spec["spec_version"]
    manifest["spec_hash"] = spec_hash(data.spec)
    manifest.to_csv(RES / "processed_subject_manifest.csv", index=False)

    smoke = {
        "spec_version": data.spec["spec_version"],
        "spec_hash": spec_hash(data.spec),
        "n_released": 842,
        "n_included": int((manifest.status == "included").sum()),
        "exclusions": manifest[manifest.status != "included"]["status"]
            .value_counts().to_dict(),
        "phi_shape": list(data.phi.shape),
        "sha256_phi": hashlib.sha256(data.phi.tobytes()).hexdigest(),
        "A_balance": {str(k): int(v) for k, v in pd.Series(data.A)
                      .value_counts().items()},
        "standardization": data.spec.get("standardization"),
        "tseq_range": [float(data.tseq[0]), float(data.tseq[-1])],
        "finite_phi": bool(pd.notna(data.phi).all()),
    }
    with open(RES / "pacce_smoke.json", "w") as f:
        json.dump(smoke, f, indent=2)
    print(json.dumps(smoke, indent=2))


if __name__ == "__main__":
    main()
