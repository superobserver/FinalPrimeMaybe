#!/usr/bin/env python3
"""
Quasi-LCM Search: Compute First Zeta Zero from Log(p) Stack
"""

import sys
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# Get deterministic holes
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
holes = []
for k in classes:
    amp = elder.generate_amplitude_map(k, 5000)   # adjust for more precision
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 5:
                holes.append(p)
holes = sorted(set(holes))[:1000]   # first 1000 holes for speed/accuracy
print(f"Using {len(holes)} holes (largest p ≈ {holes[-1]})")

def S(t):
    return np.sum(np.cos(t * np.log(holes)) / np.sqrt(holes))

# Coarse grid search
t_grid = np.arange(10.0, 18.0, 0.001)
S_grid = np.array([S(t) for t in t_grid])
t0 = t_grid[np.argmin(S_grid)]
print(f"Grid candidate: t ≈ {t0:.6f}")

# Fine optimization
res = minimize_scalar(S, bounds=(t0-1, t0+1), method='bounded', tol=1e-10)
print(f"Quasi-LCM zero: t = {res.x:.8f}   S(t) = {res.fun:.6f}")