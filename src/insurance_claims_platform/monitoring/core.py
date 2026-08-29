import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def drift_summary(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> dict[str, float]:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
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
