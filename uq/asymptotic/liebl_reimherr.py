"""Liebl-Reimherr fast-and-fair benchmark for smooth standardized limits.

Expected-excursion theory provides asymptotic coverage at least the nominal level
when the standardized Gaussian limit and covariance derivative estimates meet
its smoothness conditions. Lipschitz silhouettes or a process CLT alone do not
verify them. The Python and optional R backends implement numerical critical
functions on an evenly spaced domain rescaled to [0,1], allocating error in
proportion to interval length. This module does not certify those assumptions.
Zero variance or unusable roughness yields a whole real-line band.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, t as student_t

from ...metrics import Band
from .covariance import eif_covariance, eif_pointwise_variance
from ._validation import validate_alpha, validate_process, pointwise_scale


# --------------------------------------------------------------------------- #
# roughness parameter  tau(t)  (port of ffscb::cov2tau_fun)
# --------------------------------------------------------------------------- #
def cov2tau(cov):
    """Roughness function ``tau(t)`` from a covariance kernel (port of ``cov2tau_fun``).

    ``tau(t)`` is the pointwise sd of the standardized-and-differentiated process,
    obtained from the second difference of the *correlation* kernel:
    ``tau ~ sqrt((1 - corr(t, t+h)) / (2 h^2))``. Scale-invariant (uses the
    correlation), so it may be applied to the EIF covariance directly.

    Args:
        cov: covariance kernel ``[p, p]`` on a grid assumed to span ``[0, 1]``.

    Returns:
        ``tau`` of shape ``[p]``.
    """
    cov = np.asarray(cov, dtype=float)
    p = cov.shape[0]
    grid = np.linspace(0.0, 1.0, p)
    if cov.ndim != 2 or cov.shape != (p,p) or p < 2 or not np.isfinite(cov).all() or np.any(np.diag(cov) <= 0):
        raise ValueError("roughness needs a finite covariance with positive diagonal on at least two grid points")
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)

    a2 = corr[np.arange(p - 1), np.arange(1, p)]      # corr(t, t+step)
    h = (grid[1] - grid[0]) / 2.0
    with np.errstate(invalid="ignore"):
        tau = np.sqrt((2.0 - 2.0 * a2) / (4.0 * h ** 2))   # length p-1

    # R fills negative/NaN tau by linear interpolation over valid indices
    if np.any(~np.isfinite(tau)):
        idx = np.arange(p - 1)
        good = np.isfinite(tau)
        if not good.any():
            raise ValueError("covariance does not yield any finite roughness estimates")
        if not good[0]:
            tau[0] = tau[good][0]
            good[0] = True
        if not good[-1]:
            tau[-1] = tau[good][-1]
            good[-1] = True
        tau = np.interp(idx, idx[good], tau[good])

    # interpolate the (p-1) midpoint values back onto the p-point grid
    xx = np.concatenate([[0.0], grid[:-1] + h, [1.0]])
    yy = np.concatenate([[tau[0]], tau, [tau[-1]]])
    return np.interp(grid, xx, yy)


# --------------------------------------------------------------------------- #
# numpy port of the FFSCB critical function  u(t)
# --------------------------------------------------------------------------- #
def _uniroot_downX(f, lo, hi, tol=None, max_expand=40):
    """Root of a decreasing ``f`` on ``[lo, hi]``, expanding the bracket if needed.

    Mirrors R's ``uniroot(..., extendInt="downX")`` for a function that crosses
    from positive to negative.
    """
    flo, fhi = f(lo), f(hi)
    width = hi - lo
    k = 0
    while flo * fhi > 0 and k < max_expand:
        # extend in the direction that should contain the down-crossing
        if flo < 0:      # already negative at lo -> move lo left
            lo -= width
            flo = f(lo)
        else:            # still positive at hi -> move hi right
            hi += width
            fhi = f(hi)
        width *= 2
        k += 1
    if flo * fhi > 0:
        # no bracket found; return the endpoint with |f| closest to 0
        return lo if abs(flo) < abs(fhi) else hi
    return brentq(f, lo, hi, xtol=tol or 1e-10)


def _ffscb_u_z(tau, conf_level=0.95, n_int=4):
    """Critical function ``u(t)`` for the Gaussian FFSCB band (port of ``.make_band_FFSCB_z``).

    Returns ``u`` of shape ``[p]``; the band half-width is ``u * sqrt(diag.cov)``.
    """
    tau = np.asarray(tau, dtype=float)
    p = tau.shape[0]
    alpha = 1.0 - conf_level
    tt = np.linspace(0.0, 1.0, p)
    dt = tt[1] - tt[0]

    def tau_f(t):
        return np.interp(t, tt, tau)

    if n_int == 1:  # constant Kac-Rice band
        tau01 = tau.sum() * dt
        f = lambda c: norm.sf(c) + np.exp(-c ** 2 / 2) * tau01 / (2 * np.pi) - alpha / 2
        c = _uniroot_downX(f, 0.0, 10.0)
        return np.full(p, c)

    knots = np.linspace(0.0, 1.0, n_int + 1)
    c_v = np.zeros(n_int)

    def ufun(t, cv):
        t = np.asarray(t, dtype=float)
        out = np.full(t.shape, cv[0]) if t.ndim else cv[0]
        for j in range(1, n_int):            # interior knots knots[1..n_int-1]
            out = out + cv[j] * np.maximum(t - knots[j], 0.0)
        return out

    # initial intercept c_v[0]
    mask0 = (tt >= knots[0]) & (tt <= knots[1])
    tau_init = tau[mask0].sum() * dt
    f0 = lambda c: norm.cdf(-c) + np.exp(-c ** 2 / 2) * tau_init / (2 * np.pi) - (alpha / 2) / n_int
    c_v[0] = _uniroot_downX(f0, 0.0, 10.0)

    for j in range(1, n_int):               # solve slopes c_v[1..n_int-1]
        c_prev_sum = c_v[1:j].sum() if j >= 2 else 0.0
        seg = (tt > knots[j]) & (tt <= knots[j + 1])
        t_seg = tt[seg]

        def res(cj):
            S = c_prev_sum + cj             # u'(t) on interval j
            cv_j = c_v.copy()
            cv_j[j] = cj
            uj = ufun(t_seg, cv_j)
            tf = tau_f(t_seg)
            fn1 = (tf / (2 * np.pi)) * np.exp(-uj ** 2 / 2) * np.exp(-S ** 2 / (2 * tf ** 2))
            intgr1 = fn1.sum() * dt
            odd = (j + 1) % 2 != 0          # R's j is (python j)+1; parity on R-j
            if odd:
                fn2 = S / np.sqrt(2 * np.pi) * np.exp(-uj ** 2 / 2) * norm.cdf(S / tf)
                intgr2 = fn2.sum() * dt
                intgr3 = 0.0
                uend = ufun(knots[j + 1], cv_j)
                base = norm.cdf(-uend)
            else:
                intgr2 = 0.0
                fn3 = S / np.sqrt(2 * np.pi) * np.exp(-uj ** 2 / 2) * norm.cdf(-S / tf)
                intgr3 = fn3.sum() * dt
                uleft = ufun(knots[j], cv_j)
                base = norm.cdf(-uleft)
            return base + intgr1 + intgr2 - intgr3 - (alpha / 2) / n_int

        c_v[j] = _uniroot_downX(res, -10.0, 10.0)

    return ufun(tt, c_v)


def _ffscb_u_t(tau, df, conf_level=0.95, n_int=4):
    """Critical function ``u(t)`` for the t-distribution FFSCB band (port of ``.make_band_FFSCB_t``).

    For ``df > 101`` ffscb itself uses the Gaussian band; we mirror that.
    """
    if df > 101:
        return _ffscb_u_z(tau, conf_level=conf_level, n_int=n_int)

    tau = np.asarray(tau, dtype=float)
    p = tau.shape[0]
    alpha = 1.0 - conf_level
    tt = np.linspace(0.0, 1.0, p)
    dt = tt[1] - tt[0]
    nu = df
    nup = nu + 1
    from math import gamma as _gamma

    def tau_f(t):
        return np.interp(t, tt, tau)

    if n_int == 1:
        tau01 = tau.sum() * dt
        f = lambda c: (student_t.sf(c, nu) + (tau01 / (2 * np.pi)) * (1 + c ** 2 / nu) ** (-nu / 2) - alpha / 2)
        c = _uniroot_downX(f, 0.0, 10.0)
        return np.full(p, c)

    knots = np.linspace(0.0, 1.0, n_int + 1)
    c_v = np.zeros(n_int)

    def ufun(t, cv):
        t = np.asarray(t, dtype=float)
        out = np.full(t.shape, cv[0]) if t.ndim else cv[0]
        for j in range(1, n_int):
            out = out + cv[j] * np.maximum(t - knots[j], 0.0)
        return out

    mask0 = (tt >= knots[0]) & (tt <= knots[1])
    tau_init = tau[mask0].sum() * dt
    f0 = lambda c: student_t.cdf(-c, nu) + (tau_init / (2 * np.pi)) * (1 + c ** 2 / nu) ** (-nu / 2) - (alpha / 2) / n_int
    c_v[0] = _uniroot_downX(f0, 0.0, 10.0)

    gfac = _gamma(nup / 2) * np.sqrt(nup * np.pi) / _gamma((nup + 1) / 2)

    for j in range(1, n_int):
        c_prev_sum = c_v[1:j].sum() if j >= 2 else 0.0
        seg = (tt > knots[j]) & (tt <= knots[j + 1])
        t_seg = tt[seg]

        def res(cj):
            S = c_prev_sum + cj
            cv_j = c_v.copy()
            cv_j[j] = cj
            uj = ufun(t_seg, cv_j)
            tf = tau_f(t_seg)
            afun = np.sqrt(nu * tf ** 2 * (1 + uj ** 2 / nu) / nup)
            fn1 = tf * (1 + uj ** 2 / nu + S ** 2 / (nu * tf ** 2)) ** (-nu / 2) / (2 * np.pi)
            intgr1 = fn1.sum() * dt
            odd = (j + 1) % 2 != 0
            if odd:
                fn2 = (S / (2 * np.pi * tf)) * (1 + uj ** 2 / nu) ** (-nu / 2 - 1) * gfac * afun * student_t.cdf(S / afun, nup)
                intgr2 = fn2.sum() * dt
                intgr3 = 0.0
                base = student_t.cdf(-ufun(knots[j + 1], cv_j), nu)
            else:
                intgr2 = 0.0
                fn3 = (S / (2 * np.pi * tf)) * (1 + uj ** 2 / nu) ** (-nu / 2 - 1) * gfac * afun * student_t.cdf(-S / afun, nup)
                intgr3 = fn3.sum() * dt
                base = student_t.cdf(-ufun(knots[j], cv_j), nu)
            return base + intgr1 + intgr2 - intgr3 - (alpha / 2) / n_int

        c_v[j] = _uniroot_downX(res, -10.0, 10.0)

    return ufun(tt, c_v)


# --------------------------------------------------------------------------- #
# R backend (faithful) via Rscript subprocess
# --------------------------------------------------------------------------- #
def _find_ffscb_source(explicit=None):
    """Locate ``ffscb-master/R``; search ``explicit``, env var, then up from cwd/module."""
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("FFSCB_R_DIR"):
        cands.append(os.environ["FFSCB_R_DIR"])
    here = os.path.abspath(os.getcwd())
    mod = os.path.dirname(os.path.abspath(__file__))
    for start in (here, mod):
        d = start
        for _ in range(6):
            cands.append(os.path.join(d, "ffscb-master", "R"))
            d = os.path.dirname(d)
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "make_band.R")):
            return c
    return None


def r_backend_available(ffscb_dir=None):
    """True iff ``Rscript`` is on PATH and the ffscb R source is found."""
    return shutil.which("Rscript") is not None and _find_ffscb_source(ffscb_dir) is not None


_R_DRIVER = r"""
suppressWarnings(suppressMessages({{
  base <- "{ffscb}/"
  source(paste0(base, "tau_fun.R"))
  source(paste0(base, "make_band.R"))
}}));
args <- commandArgs(trailingOnly = TRUE)
io   <- args[1]
dat  <- as.matrix(read.csv(file.path(io, "in.csv")))
prm  <- as.list(read.csv(file.path(io, "params.csv")))
x        <- dat[, "x"]
diagcov  <- dat[, "diagcov"]
tau      <- dat[, "tau"]
conf     <- prm$conf.level
n_int    <- prm$n_int
if (prm$dist == "z") {{
  band <- make_band_FFSCB_z(x = x, diag.cov.x = diagcov, tau = tau, conf.level = conf, n_int = n_int)
}} else {{
  band <- make_band_FFSCB_t(x = x, diag.cov.x = diagcov, tau = tau, df = prm$df, conf.level = conf, n_int = n_int)
}}
half <- band[, 2] - band[, 1]
write.csv(data.frame(half = half), file.path(io, "out.csv"), row.names = FALSE)
"""


def _ffscb_u_R(x, diag_cov, tau, *, distribution, df, conf_level, n_int, ffscb_dir):
    """Call the original ffscb band via Rscript; returns the half-width vector."""
    ffscb = _find_ffscb_source(ffscb_dir)
    if ffscb is None:
        raise RuntimeError("ffscb R source not found (set FFSCB_R_DIR).")
    with tempfile.TemporaryDirectory() as io:
        import csv

        with open(os.path.join(io, "in.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x", "diagcov", "tau"])
            for xi, ci, ti in zip(x, diag_cov, tau):
                w.writerow([repr(float(xi)), repr(float(ci)), repr(float(ti))])
        with open(os.path.join(io, "params.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["conf.level", "n_int", "dist", "df"])
            w.writerow([conf_level, int(n_int), distribution, int(df)])
        driver = os.path.join(io, "driver.R")
        with open(driver, "w") as fh:
            fh.write(_R_DRIVER.format(ffscb=ffscb))
        proc = subprocess.run(
            ["Rscript", "--vanilla", driver, io],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Rscript failed:\n{proc.stderr[-2000:]}")
        out = np.genfromtxt(os.path.join(io, "out.csv"), delimiter=",", names=True)
        return np.atleast_1d(out["half"].astype(float))


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def liebl_reimherr_band(
    influence,
    tseq,
    estimate,
    *,
    alpha: float = 0.05,
    n_int: int = 4,
    distribution: str = "z",
    backend: str = "auto",
    variance=None,
    covariance=None,
    tau=None,
    ffscb_dir=None,
) -> Band:
    """Liebl-Reimherr benchmark under additional smooth-process conditions.

    Expected excursion theory gives an asymptotic coverage lower bound, not
    automatically equality. Continuous or Lipschitz silhouettes alone do not
    verify standardized-process smoothness or covariance derivative consistency.
    The implementation rescales an evenly spaced domain to [0,1] for proportional
    fair error allocation. Zero variance or unusable roughness returns the whole line.

    Args:
        influence: centered EIF process ``[n, res]`` (``cross_fit(...).influence()[d]``).
        tseq: grid ``[res]``.
        estimate: ``psi_hat_d(t)`` ``[res]`` (band center).
        alpha: 1 - simultaneous confidence level.
        n_int: number of "fair" sub-intervals (``1`` = constant Kac-Rice band;
            ffscb default ``4``).
        distribution: ``"z"`` (Gaussian limit; default, matches Theorem 5.2) or
            ``"t"`` with ``df = n - 1``.
        backend: ``"auto"`` (R if available else Python), ``"R"``, or ``"python"``.
        variance: optional pointwise variance ``sigma_hat^2(t)`` of the EIF; if
            ``None`` it is estimated from ``influence``. ``diag.cov.x`` is this
            over ``n``.
        covariance: optional EIF covariance kernel ``[res, res]`` (reuse to avoid
            recomputation); used only to derive ``tau`` if ``tau`` is not given.
        tau: optional precomputed roughness ``[res]``.
        ffscb_dir: path to ``ffscb-master/R`` (else auto-discovered).

    Returns:
        :class:`~tcda_uq.metrics.Band` with ``kind="confidence"``.
    """
    validate_alpha(alpha)
    phi, estimate = validate_process(influence, estimate, tseq)
    n = phi.shape[0]
    if not isinstance(n_int, (int, np.integer)) or n_int < 1:
        raise ValueError("n_int must be a positive integer")
    if distribution not in ("z", "t") or backend not in ("auto", "R", "python"):
        raise ValueError("invalid distribution or backend")
    grid = np.asarray(tseq, float)
    if grid.size < 2 or not np.allclose(np.diff(grid), (grid[-1]-grid[0])/(grid.size-1)):
        raise ValueError("Liebl-Reimherr implementation requires an evenly spaced grid, rescaled internally to [0,1]")
    var = pointwise_scale(phi, variance)**2
    if np.any(var == 0):
        return Band(tseq=tseq, lower=np.full_like(estimate, -np.inf),
                    upper=np.full_like(estimate, np.inf), center=estimate,
                    level=1-alpha, kind="confidence")
    diag_cov = var / n                                  # Var(psi_hat(t))
    if tau is None:
        cov = eif_covariance(phi) if covariance is None else np.asarray(covariance, float)
        tau = cov2tau(cov)
    tau = np.asarray(tau, dtype=float)
    if tau.shape != estimate.shape or not np.isfinite(tau).all() or np.any(tau < 0):
        raise ValueError("tau must be finite, nonnegative, and match the grid")
    if np.any(tau == 0):
        # The segmented numerical formula divides by roughness; abstain instead
        # of substituting an arbitrary smoothness/variance floor.
        return Band(tseq=tseq, lower=np.full_like(estimate, -np.inf),
                    upper=np.full_like(estimate, np.inf), center=estimate,
                    level=1-alpha, kind="confidence")

    if backend == "auto":
        backend = "R" if r_backend_available(ffscb_dir) else "python"

    if backend == "R":
        half = _ffscb_u_R(
            estimate, diag_cov, tau,
            distribution=distribution, df=n - 1,
            conf_level=1.0 - alpha, n_int=n_int, ffscb_dir=ffscb_dir,
        )
    elif backend == "python":
        if distribution == "z":
            u = _ffscb_u_z(tau, conf_level=1.0 - alpha, n_int=n_int)
        else:
            u = _ffscb_u_t(tau, df=n - 1, conf_level=1.0 - alpha, n_int=n_int)
        half = u * np.sqrt(diag_cov)
    else:
        raise ValueError(f"backend must be 'auto'/'R'/'python', got {backend!r}")

    if not np.isfinite(half).all() or np.any(half < 0):
        half = np.full_like(estimate, np.inf)
    return Band(
        tseq=tseq,
        lower=estimate - half,
        upper=estimate + half,
        center=estimate,
        level=1.0 - alpha,
        kind="confidence",
    )


def liebl_reimherr_bands(cross_fit_result, *, alpha: float = 0.05, **kwargs):
    """One Liebl-Reimherr band per homology dim from a :class:`CrossFitResult`."""
    influence = cross_fit_result.influence()
    tseq = cross_fit_result.tseq
    return [
        liebl_reimherr_band(
            influence[d], tseq, cross_fit_result.aipw[d], alpha=alpha, **kwargs
        )
        for d in range(len(influence))
    ]
