#!/usr/bin/env python3
"""
Global 24-Class Resonance vs. Single-Class Resonance
Side-by-side with known zeta zeros and theoretical -ζ'/ζ overlay.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import mpmath

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
k_single = 11
max_m = 40000                  # increase for sharper peaks
t_start, t_end, step = 10, 45, 0.05
known_zeros = [14.134725, 21.022039, 25.010857, 30.424876, 32.935062, 37.586178]
# ===================================================

# Get the 24 classes (module maps 1 → 91)
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]

print(f"Computing single-class (k={k_single}) and global resonance over all 24 classes...")

# Single class
amplitude_single = elder.generate_amplitude_map(k_single, max_m)
holes_single = [m for m in range(len(amplitude_single)) if amplitude_single[m] == 0]
primes_single = [90 * m + k_single for m in holes_single]

# All 24 classes
primes_global = []
for k in classes:
    amp = elder.generate_amplitude_map(k, max_m)
    holes = [m for m in range(len(amp)) if amp[m] == 0]
    primes_global.extend(90 * m + k for m in holes)
primes_global = sorted(set(primes_global))  # deduplicate

print(f"Single class primes: {len(primes_single)}")
print(f"Global primes (all 24 classes): {len(primes_global)}")

# Compute resonance
t_values = np.arange(t_start, t_end, step)
res_single = np.zeros(len(t_values))
res_global = np.zeros(len(t_values))

for i, t in enumerate(t_values):
    # Single class
    res_single[i] = sum(np.cos(t * np.log(p)) / np.sqrt(p) for p in primes_single if p > 1)
    # Global
    res_global[i] = sum(np.cos(t * np.log(p)) / np.sqrt(p) for p in primes_global if p > 1)

# Theoretical -ζ'(s)/ζ(s) at s = 0.01 + it (scaled by 4/15 for density)
theo = []
for t in t_values:
    s = complex(0.01, t)
    z = mpmath.zeta(s)
    zd = mpmath.zeta(s, derivative=1)
    log_deriv = -zd / z
    theo.append(float(log_deriv.real) * 4 / 15)   # <-- FIXED: use .real

# ====================== PLOT ======================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

# Left: Single class
ax1.plot(t_values, res_single, 'b-', lw=1.5, label='Single-class resonance (k=11)')
for z in known_zeros:
    ax1.axvline(z, color='red', linestyle='--', alpha=0.7, label='Known zeta zero' if z == known_zeros[0] else "")
ax1.set_title('Single-Class Resonance (Dirac Comb Holes)')
ax1.set_ylabel('Signal Strength')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Right: Global 24-class + theoretical overlay
ax2.plot(t_values, res_global, 'darkblue', lw=1.8, label='Global 24-class resonance')
ax2.plot(t_values, theo, 'orange', lw=1.5, label=r'Scaled theoretical $-\zeta''/\zeta$ (×4/15)')
for z in known_zeros:
    ax2.axvline(z, color='red', linestyle='--', alpha=0.7)
ax2.set_title('Global Resonance over All 24 Classes')
ax2.set_xlabel('t (imaginary part)')
ax2.set_ylabel('Signal Strength')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.suptitle('Alignment of Algebraic Ideal Holes with Zeta Zeros\n'
             f'Single class (k={k_single}) vs. Full 24-class global sum', fontsize=16)
plt.tight_layout()
plt.savefig('global_vs_single_resonance_fixed.png', dpi=300, bbox_inches='tight')
plt.show()