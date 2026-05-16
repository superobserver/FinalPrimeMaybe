#!/usr/bin/env python3
"""
Resonance Comparison: Holes vs. Composites vs. Full Coprime-to-90 Field
Shows that the zeta-zero signature is carried by the holes (prime signal)
while the composites add only a smooth background.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
max_m = 1000                 # increase for sharper higher-t troughs
t_min, t_max, dt = 0.0, 250.0, 0.02
known_zeros = [14.134725, 21.022039, 25.010857, 30.424876, 32.935062,
               37.586178, 40.918719, 43.327073, 48.005150, 49.773832]
# ===================================================

classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]

# Pre-compute the three lists from the amplitude maps
primes_holes = []
composites_coprime = []
full_coprime = []

for k in classes:
    amp = elder.generate_amplitude_map(k, max_m)
    for m in range(max_m + 1):
        N = 90 * m + k
        if N <= 1:
            continue
        full_coprime.append(N)
        if amp[m] == 0:
            primes_holes.append(N)
        else:
            composites_coprime.append(N)

primes_holes = sorted(set(primes_holes))
composites_coprime = sorted(set(composites_coprime))
full_coprime = sorted(set(full_coprime))

print(f"Prime holes: {len(primes_holes)}")
print(f"Coprime composites: {len(composites_coprime)}")
print(f"Full coprime-to-90: {len(full_coprime)}")

# t-grid
t_values = np.arange(t_min, t_max, dt)

# Compute the three resonance signals
S_holes = np.zeros(len(t_values))
S_composites = np.zeros(len(t_values))
S_full = np.zeros(len(t_values))

for i, t in enumerate(t_values):
    # Prime holes only
    S_holes[i] = sum(np.cos(t * np.log(p)) / np.sqrt(p) for p in primes_holes if p > 1)
    # Coprime composites only
    S_composites[i] = sum(np.cos(t * np.log(c)) / np.sqrt(c) for c in composites_coprime if c > 1)
    # Full coprime field
    S_full[i] = sum(np.cos(t * np.log(N)) / np.sqrt(N) for N in full_coprime if N > 1)

# ====================== PLOT ======================
fig, axs = plt.subplots(3, 1, figsize=(14, 10), dpi=300, sharex=True)

axs[0].plot(t_values, S_holes, 'b-', lw=1.8, label='Prime holes only')
for z in known_zeros:
    axs[0].axvline(z, color='red', linestyle='--', alpha=0.7, label='Known zeta zero' if z == known_zeros[0] else "")
axs[0].set_ylabel('Signal Strength (holes)')
axs[0].set_title('Resonance from Prime Holes (Algebraic Ideal)')
axs[0].grid(True, alpha=0.3)
axs[0].legend()

axs[1].plot(t_values, S_composites, 'orange', lw=1.8, label='Coprime-to-90 composites only')
for z in known_zeros:
    axs[1].axvline(z, color='red', linestyle='--', alpha=0.7)
axs[1].set_ylabel('Signal Strength (composites)')
axs[1].set_title('Resonance from Coprime Composites')
axs[1].grid(True, alpha=0.3)
axs[1].legend()

axs[2].plot(t_values, S_full, 'purple', lw=1.8, label='Full coprime-to-90 field')
for z in known_zeros:
    axs[2].axvline(z, color='red', linestyle='--', alpha=0.7)
axs[2].set_xlabel('t (imaginary part)')
axs[2].set_ylabel('Signal Strength (full coprime)')
axs[2].set_title('Resonance from Full Coprime-to-90 Field')
axs[2].grid(True, alpha=0.3)
axs[2].legend()

plt.suptitle('Prime Holes vs. Composites vs. Full Coprime-to-90 Field\n'
             'The zeta-zero signature is carried almost entirely by the holes')
plt.tight_layout()
plt.savefig('resonance_holes_vs_composites_vs_full.png', dpi=300, bbox_inches='tight')
plt.show()