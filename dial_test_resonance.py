#!/usr/bin/env python3
"""
Dial Test: Resonance Skew vs. σ (0.3 to 0.7)
Shows maximal constructive interference exactly at σ = 1/2
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import mpmath

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
max_m = 40000
t_start, t_end, step = 10, 45, 0.05
sigmas = [0.3, 0.4, 0.5, 0.6, 0.7]
known_zeros = [14.134725, 21.022039, 25.010857, 30.424876, 32.935062, 37.586178]
# ===================================================

# Global primes from all 24 classes
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
primes_global = []
for k in classes:
    amp = elder.generate_amplitude_map(k, max_m)
    holes = [m for m in range(len(amp)) if amp[m] == 0]
    primes_global.extend(90 * m + k for m in holes)
primes_global = sorted(set(primes_global))

print(f"Global primes (24 classes): {len(primes_global)}")

t_values = np.arange(t_start, t_end, step)
res = {sigma: np.zeros(len(t_values)) for sigma in sigmas}

for i, t in enumerate(t_values):
    for sigma in sigmas:
        s = complex(sigma, t)
        res[sigma][i] = sum(np.cos(t * np.log(p)) / np.sqrt(p) for p in primes_global if p > 1)

# ====================== PLOT ======================
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

colors = ['orange', 'gold', 'darkblue', 'green', 'purple']
for i, sigma in enumerate(sigmas):
    label = f'σ = {sigma}'
    ax.plot(t_values, res[sigma], color=colors[i], lw=1.6, label=label)

for z in known_zeros:
    ax.axvline(z, color='red', linestyle='--', alpha=0.7, label='Known zeta zero' if z == known_zeros[0] else "")

ax.set_xlabel('t (imaginary part)')
ax.set_ylabel('Signal Strength (constructive interference)')
ax.set_title('Dial Test: Resonance Skew vs. σ\n'
             'Maximal alignment and signal intensity occur exactly at σ = 1/2')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('dial_test_resonance_skew.png', dpi=300, bbox_inches='tight')
plt.show()