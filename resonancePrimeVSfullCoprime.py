#!/usr/bin/env python3
"""
Resonance Comparison: Prime Holes vs. Full Coprime-to-90 Field
Side-by-side or overlaid plot to show the zero signature is carried by the coprime field itself.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
max_m = 100000                 # increase for sharper troughs
t_min, t_max, dt = 0.0, 250.0, 0.02
known_zeros = [14.134725, 21.022039, 25.010857, 30.424876, 32.935062,
               37.586178, 40.918719, 43.327073, 48.005150, 49.773832]
# ===================================================

classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]

# 1. Prime-only holes (as in your original script)
primes_holes = []
for k in classes:
    amp = elder.generate_amplitude_map(k, max_m)
    holes = [m for m in range(len(amp)) if amp[m] == 0]
    primes_holes.extend(90 * m + k for m in holes)
primes_holes = sorted(set(primes_holes))
print(f"Prime holes used: {len(primes_holes)} (largest ≈ {primes_holes[-1]})")

# 2. Full coprime-to-90 field (every index m in the 24 classes)
coprime_N = []
for k in classes:
    for m in range(max_m + 1):
        N = 90 * m + k
        if N > 1:
            coprime_N.append(N)
coprime_N = sorted(set(coprime_N))
print(f"Full coprime-to-90 numbers used: {len(coprime_N)} (largest ≈ {coprime_N[-1]})")

# t-grid
t_values = np.arange(t_min, t_max, dt)

# Compute both resonance signals
S_holes = np.zeros(len(t_values))
S_full = np.zeros(len(t_values))

for i, t in enumerate(t_values):
    # Prime-only
    S_holes[i] = sum(np.cos(t * np.log(p)) / np.sqrt(p) for p in primes_holes if p > 1)
    # Full coprime field
    S_full[i] = sum(np.cos(t * np.log(N)) / np.sqrt(N) for N in coprime_N if N > 1)

# ====================== PLOT ======================
fig, axs = plt.subplots(2, 1, figsize=(14, 9), dpi=300, sharex=True)

axs[0].plot(t_values, S_holes, 'b-', lw=1.8, label='Prime holes only (S_holes(t))')
for z in known_zeros:
    axs[0].axvline(z, color='red', linestyle='--', alpha=0.7, label='Known zeta zero' if z == known_zeros[0] else "")
axs[0].set_ylabel('Signal Strength (prime holes)')
axs[0].set_title('Resonance from Prime Holes (Algebraic Ideal)')
axs[0].grid(True, alpha=0.3)
axs[0].legend()

axs[1].plot(t_values, S_full, 'orange', lw=1.8, label='All coprime-to-90 numbers (S_full(t))')
for z in known_zeros:
    axs[1].axvline(z, color='red', linestyle='--', alpha=0.7)
axs[1].set_xlabel('t (imaginary part)')
axs[1].set_ylabel('Signal Strength (full coprime field)')
axs[1].set_title('Resonance from Full Coprime-to-90 Field')
axs[1].grid(True, alpha=0.3)
axs[1].legend()

plt.suptitle('Prime Holes vs. Full Coprime-to-90 Field\n'
             'The zeta-zero signature is a class property of the coprime-to-90 field')
plt.tight_layout()
plt.savefig('resonance_prime_vs_full_coprime.png', dpi=300, bbox_inches='tight')
plt.show()