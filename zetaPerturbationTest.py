#!/usr/bin/env python3
"""
Degradation Test: Zeta-Zero Signal with One Off-1/√p Datapoint
"""

import sys
import numpy as np
sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# Get deterministic holes
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
holes = []
for k in classes:
    amp = elder.generate_amplitude_map(k, 20000)
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 5:
                holes.append(p)
holes = np.array(sorted(set(holes)))
print(f"Using {len(holes)} holes (largest p ≈ {holes[-1]:,.0f})")

def S(t, holes, perturb_idx=None, perturb_factor=1.0):
    logp = np.log(holes)
    sqrtp = np.sqrt(holes)
    contrib = np.cos(t * logp) / sqrtp
    if perturb_idx is not None:
        contrib[perturb_idx] *= perturb_factor
    return np.sum(contrib)

t_zero = 14.134725
S_normal = S(t_zero, holes)
print(f"Normal 1/√p weighting → S(t) = {S_normal:8.4f}")

# Example perturbations
print("\nPerturbation tests:")
for factor in [0.5, 2.0, 0.0, 1.0 / holes[0]]:  # halve, double, remove, replace with 1/p
    S_pert = S(t_zero, holes, perturb_idx=0, perturb_factor=factor)
    degradation = abs(S_normal) - abs(S_pert)
    print(f"  Factor {factor:6.3f} → S(t) = {S_pert:8.4f}  (degradation {degradation:6.4f})")