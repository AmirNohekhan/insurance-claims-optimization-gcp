import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def drift_summary(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if reference.size == 0 or current.size == 0:
        raise ValueError("reference and current must be non-empty")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("reference and current must contain only finite values")
    # Infinite outer edges ensure shifted current values are counted instead of discarded.
    inner_edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.r_[-np.inf, inner_edges, np.inf]
    ref, _ = np.histogram(reference, edges)
    cur, _ = np.histogram(current, edges)
    rp = np.maximum(ref / max(ref.sum(), 1), 1e-6)
    cp = np.maximum(cur / max(cur.sum(), 1), 1e-6)
    psi = float(np.sum((cp - rp) * np.log(cp / rp)))
    return {
        "psi": psi,
        "ks": float(ks_2samp(reference, current).statistic),
        "wasserstein": float(wasserstein_distance(reference, current)),
    }
