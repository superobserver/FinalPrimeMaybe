#!/usr/bin/env python3
"""
Direct Zero Computation from Algebraic Ideal Holes
Larger dataset + improved peak detection for clearer alignment.
"""

import sys
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import mpmath

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
max_m = 200000            # larger = sharper early zeros
t_start, t_end, step = 10, 100, 0.01
known_zeros = [14.134725, 21.022039, 25.010857, 30.424876, 32.935062, 37.586178, 40.918719, 43.327073]
# ===================================================

classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]

primes_global = []
for k in classes:
    amp = elder.generate_amplitude_map(k, max_m)
    holes = [m for m in range(len(amp)) if amp[m] == 0]
    primes_global.extend(90 * m + k for m in holes)
primes_global = sorted(set(primes_global))

print(f"Global primes from 24 classes: {len(primes_global)} (largest ≈ {primes_global[-1]})")

t_values = np.arange(t_start, t_end, step)
S = np.zeros(len(t_values))

for i, t in enumerate(t_values):
    S[i] = sum(np.cos(t * np.log(p)) / np.sqrt(p) for p in primes_global if p > 1)

# Find local maxima (candidate zeros)
peaks, properties = find_peaks(S, prominence=1.0, distance=50/step)
candidate_zeros = t_values[peaks]

print("\nTop candidate zeros located directly from hole data:")
for tz in candidate_zeros[:12]:
    print(f"  t ≈ {tz:.6f}")

print("\nKnown zeta zeros for comparison:")
for kz in known_zeros:
    print(f"  t ≈ {kz:.6f}")

# Plot
plt.figure(figsize=(14, 6), dpi=300)
plt.plot(t_values, S, 'b-', lw=1.2, label='Global resonance S(t) from algebraic ideal holes')
for z in known_zeros:
    plt.axvline(z, color='red', linestyle='--', alpha=0.7, label='Known zeta zero' if z == known_zeros[0] else "")
plt.plot(candidate_zeros, S[peaks], 'ro', markersize=6, label='Candidate zeros from holes')
plt.title('Direct Computation of Candidate Zeta Zeros from Algebraic Ideal Holes')
plt.xlabel('t (imaginary part)')
plt.ylabel('Signal Strength (constructive interference)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('direct_zero_computation_from_holes_larger.png', dpi=300, bbox_inches='tight')
plt.show()