"""SARS-CoV-2 CT-scan modality (0-dim cubical / lower-star filtration).

Requires the ``[data]`` extra (``kagglehub``, ``pillow``). Faithful to
``SARS_COV2/main.ipynb``: grayscale CT images -> H0 cubical silhouettes; the
causal dataset mixes infected / non-infected images to form potential outcomes.
"""

from __future__ import annotations

import os

import numpy as np

from ..silhouette import silhouette_from_image

_KAGGLE_SLUG = "plameneduardo/sarscov2-ctscan-dataset"


def download_ct(cache_dir="./") -> str:
    """Download the SARS-CoV-2 CT dataset via kagglehub; return its root path."""
    import kagglehub

    os.environ.setdefault("KAGGLEHUB_CACHE", cache_dir)
    return kagglehub.dataset_download(_KAGGLE_SLUG)


def load_ct_images(root_dir):
    """Load COVID / non-COVID grayscale images (first channel, normalised to [0,1])."""
    from PIL import Image

    def _load(folder):
        imgs = []
        for f in os.listdir(folder):
            arr = np.array(Image.open(os.path.join(folder, f))) / 255.0
            imgs.append(arr[:, :, 0] if arr.ndim == 3 else arr)
        return imgs

    infected = _load(os.path.join(root_dir, "COVID"))
    noninfected = _load(os.path.join(root_dir, "non-COVID"))
    return infected, noninfected


def _silhouette_task(args):
    image, interval, r, resolution, normalization, allow_experimental_r = args
    return silhouette_from_image(
        image, interval, r, resolution, homology_dims=(0,),
        normalization=normalization, allow_experimental_r=allow_experimental_r,
    )


def _limit_threads():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


def images_to_silhouettes(
    images, interval=(0.0, 1.0), r=1.0, resolution=100, workers=1,
    *, normalization="plain", allow_experimental_r=False,
):
    """H0 cubical plain tents ``[n, 1, resolution]``, with supported r=1.

    Legacy GUDHI units and r<1 require the explicit corresponding opt-ins.
    """
    tasks = [(im, interval, r, resolution, normalization, allow_experimental_r)
             for im in images]
    if not tasks:
        # Validate the public options even for an empty image collection.
        from ..silhouette import compute_silhouette
        compute_silhouette([], interval, r, resolution, normalization=normalization,
                           allow_experimental_r=allow_experimental_r)
        return np.empty((0, 1, resolution))
    if workers is None or int(workers) <= 1:
        return np.stack([_silhouette_task(task) for task in tasks])

    from multiprocessing import Pool

    with Pool(processes=int(workers), initializer=_limit_threads) as pool:
        return np.stack(pool.map(_silhouette_task, tasks, chunksize=1))


def make_sarscov2_causal(
    phi_infected,
    phi_noninfected,
    n=500,
    p=0.75,
    rng=None,
):
    """Synthetic i.i.d. paired outcomes from empirical image distributions.

    Control images are infected; each treated image is independently drawn
    from the non-infected pool with probability p, otherwise the infected pool.
    This artificial pairing is a simulation, not an identified clinical effect.
    ``true_effect`` is the realized finite-sample average paired contrast;
    ``population_effect`` is the mean under the empirical-pool generating law.
    """
    rng = np.random.default_rng(rng)
    from .covariates import _sample_size
    n = _sample_size(n)
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be a probability in [0, 1]")
    phi_infected = np.asarray(phi_infected, dtype=float)
    phi_noninfected = np.asarray(phi_noninfected, dtype=float)
    if (phi_infected.ndim != 3 or phi_noninfected.ndim != 3
            or phi_infected.shape[1:] != phi_noninfected.shape[1:]
            or not len(phi_infected) or not len(phi_noninfected)
            or not np.isfinite(phi_infected).all() or not np.isfinite(phi_noninfected).all()):
        raise ValueError("both image pools must be nonempty arrays with matching [n, hom, grid] shapes")
    noninfected = rng.binomial(1, p, size=n).astype(bool)
    potential_ctrl = phi_infected[rng.choice(len(phi_infected), n)]
    potential_trt = np.empty_like(potential_ctrl)
    potential_trt[noninfected] = phi_noninfected[
        rng.choice(len(phi_noninfected), int(noninfected.sum()))]
    potential_trt[~noninfected] = phi_infected[
        rng.choice(len(phi_infected), int((~noninfected).sum()))]
    potential_outcomes = np.stack([potential_ctrl, potential_trt], axis=1)
    true_effect = potential_trt.mean(0) - potential_ctrl.mean(0) if n else np.full(phi_infected.shape[1:], np.nan)
    population_effect = p * (phi_noninfected.mean(0) - phi_infected.mean(0))
    return {"potential_outcomes": potential_outcomes, "true_effect": true_effect,
            "population_effect": population_effect}
