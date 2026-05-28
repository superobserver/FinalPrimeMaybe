#!/usr/bin/env python3
"""
Full-AP vs Holes-Only Resonance: Coarse Harmonic Nodes
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

k = 17                          # change class as desired
max_m = 5000000
t_values = np.arange(0.0, 80.0, 0.05)
known_zeros = [14.134725, 21.022039, 25.010857, 30.424876, 32.935062, 37.586178]

# Full AP
N_full = np.array([90 * m + k for m in range(max_m + 1) if 90 * m + k > 1])

# Holes only
amp = elder.generate_amplitude_map(k, max_m)
holes_m = [m for m in range(len(amp)) if amp[m] == 0]
N_holes = np.array([90 * m + k for m in holes_m if 90 * m + k > 1])

def compute_S(t_vals, N_list):
    S = np.zeros(len(t_vals))
    logN = np.log(N_list)
    sqrtN = np.sqrt(N_list)
    for i, t in enumerate(t_vals):
        S[i] = np.sum(np.cos(t * logN) / sqrtN)
    return S

S_full = compute_S(t_values, N_full)
S_holes = compute_S(t_values, N_holes)

fig, axs = plt.subplots(2, 1, figsize=(14, 9), dpi=300, sharex=True)
axs[0].plot(t_values, S_full, 'b-', lw=1.5, label=f'Full AP (all 90n+{k})')
axs[1].plot(t_values, S_holes, 'orange', lw=1.5, label=f'Holes only (primes ≡{k} mod 90)')
for z in known_zeros:
    for ax in axs:
        ax.axvline(z, color='red', linestyle='--', alpha=0.6)

axs[0].set_title('Full Arithmetic Progression Resonance (Coarse Harmonic Nodes)')
axs[1].set_title('Holes-Only Resonance (Exact Class-Bound Zeros)')
axs[1].set_xlabel('t')
for ax in axs:
    ax.set_ylabel('S(t)')
    ax.grid(True, alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig(f'fullAP_vs_holes_class_{k}.png', dpi=300, bbox_inches='tight')
plt.show()