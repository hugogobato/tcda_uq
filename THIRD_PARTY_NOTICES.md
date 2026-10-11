# Third-Party Notices

The `tcda_uq` library code in this repository is MIT licensed.

Datasets are not bundled:

- ORBIT-style data are generated locally by `tcda_uq.datasets.orbit`.
- SARS-CoV-2 CT data are downloaded by the optional `[data]` extra and may
  require Kaggle credentials.
- Graph datasets such as GEOM-Drugs are not included.

The optional Liebl-Reimherr R comparison route can use the external FFSCB R
package if `FFSCB_R_DIR` points to an FFSCB `R/` source directory. FFSCB is not
vendored here and keeps its own GPL license.
