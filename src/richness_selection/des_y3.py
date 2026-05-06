"""DES Y3 cluster bin definitions.

Four richness bins crossed with three observed-redshift bins (the
z <= 0.65 range used in Costanzi 2026 and the weak-lensing analyses)
yield the 12 (Delta lambda, Delta z) joint bins that the projection
pipeline is validated on.

Also published: convenience arrays of bin centres (midpoints of the
edges, with the open-ended (60, 200] richness bin reported at 130).
"""
from __future__ import annotations
import numpy as np


Y3_LAM_BINS = [(20.0, 30.0), (30.0, 45.0), (45.0, 60.0), (60.0, 200.0)]
Y3_Z_BINS = [(0.20, 0.35), (0.35, 0.50), (0.50, 0.65)]

Y3_LAM_MEAN = np.array([25.0, 37.5, 52.5, 130.0])
Y3_Z_MEAN = np.array([0.275, 0.425, 0.575])


def iter_bins():
    """Yield ((i, j), (lam_edges, z_edges), (lam_bar, z_bar)) for the 12 bins."""
    for i, (lam_edges, lam_bar) in enumerate(zip(Y3_LAM_BINS, Y3_LAM_MEAN)):
        for j, (z_edges, z_bar) in enumerate(zip(Y3_Z_BINS, Y3_Z_MEAN)):
            yield (i, j), (lam_edges, z_edges), (float(lam_bar), float(z_bar))
