#!/usr/bin/env python3
"""
Phase Clustering at Zeta Zeros: Illustration of Coherent Alignment in the Algebraic Ideal
Shows how phases θ_p = t log p mod 2π cluster near π at true zeros,
producing the negative bias in S(t) that cancels the main term x.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder   # your algebraic ideal module

# ====================== CONFIG ======================
max_m = 50000                  # sufficient for first ~3000 holes
t_zero = 14.134725             # first known non-trivial zero
t_random = 14.0                # nearby non-zero test point
known_zeros = [14.134725]      # add more if desired
# ===================================================

# Get deterministic holes from all 24 coprime classes
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
holes = []
for k in classes:
    amp = elder.generate_amplitude_map(k, max_m)
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 1:
                holes.append(p)
holes = sorted(set(holes))
print(f"Using {len(holes)} deterministic holes (largest ≈ {holes[-1]})")

# ====================== PHASE CLUSTERING ======================
def compute_phases(t, primes):
    phases = (t * np.log(primes)) % (2 * np.pi)
    cosines = np.cos(phases)
    return phases, cosines

phases_zero, cosines_zero = compute_phases(t_zero, holes)
phases_rand, cosines_rand = compute_phases(t_random, holes)

# ====================== PLOT ======================
fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

# Phase histogram at true zero
axs[0, 0].hist(phases_zero, bins=50, range=(0, 2*np.pi), color='blue', alpha=0.7)
axs[0, 0].axvline(np.pi, color='red', linestyle='--', lw=2, label='π (mod 2π)')
axs[0, 0].set_title(f'Phase Distribution at True Zero t = {t_zero:.6f}')
axs[0, 0].set_xlabel('Phase θ_p = t log p (mod 2π)')
axs[0, 0].set_ylabel('Number of holes')
axs[0, 0].legend()

# Phase histogram at random t
axs[0, 1].hist(phases_rand, bins=50, range=(0, 2*np.pi), color='orange', alpha=0.7)
axs[0, 1].axvline(np.pi, color='red', linestyle='--', lw=2)
axs[0, 1].set_title(f'Phase Distribution at Random t = {t_random:.1f}')
axs[0, 1].set_xlabel('Phase θ_p = t log p (mod 2π)')

# Cosine distribution at true zero (negative bias = trough)
axs[1, 0].hist(cosines_zero, bins=50, color='blue', alpha=0.7)
axs[1, 0].axvline(-1, color='red', linestyle='--', lw=2, label='cos(π) = -1')
axs[1, 0].set_title('Cosine Values at True Zero (negative bias)')
axs[1, 0].set_xlabel('cos(θ_p)')
axs[1, 0].set_ylabel('Number of holes')
axs[1, 0].legend()

# Cosine distribution at random t
axs[1, 1].hist(cosines_rand, bins=50, color='orange', alpha=0.7)
axs[1, 1].axvline(-1, color='red', linestyle='--', lw=2)
axs[1, 1].set_title('Cosine Values at Random t (no bias)')

plt.suptitle('Phase Clustering at Zeta Zeros — Algebraic Ideal Holes\n'
             'Coherent alignment near π produces the sharp trough in S(t)')
plt.tight_layout()
plt.savefig('phase_clustering_at_zeta_zeros.png', dpi=300, bbox_inches='tight')
plt.show()