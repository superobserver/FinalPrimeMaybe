#!/usr/bin/env python3
"""
Divergence Test + Index-Zeta Analogue
Side-by-side comparison of base-10 resonance vs. pure-index resonance.
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
known_zeros = [14.134725, 21.022039, 25.010857, 30.424876, 32.935062, 37.586178]
# ===================================================

classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]

# Global primes (base-10)
primes_global = []
indices_global = []          # for pure-index version
for k in classes:
    amp = elder.generate_amplitude_map(k, max_m)
    holes = [m for m in range(len(amp)) if amp[m] == 0]
    primes_global.extend(90 * m + k for m in holes)
    indices_global.extend(m for m in holes)
primes_global = sorted(set(primes_global))
indices_global = sorted(set(indices_global))

print(f"Global primes (base-10): {len(primes_global)}")
print(f"Global indices (pure m): {len(indices_global)}")

t_values = np.arange(t_start, t_end, step)
res_base10 = np.zeros(len(t_values))
res_index   = np.zeros(len(t_values))

for i, t in enumerate(t_values):
    # Base-10 resonance
    res_base10[i] = sum(np.cos(t * np.log(p)) / np.sqrt(p) for p in primes_global if p > 1)
    # Pure index resonance
    res_index[i]   = sum(np.cos(t * np.log(m + 1)) / np.sqrt(m + 1) for m in indices_global if m > 0)  # +1 avoids log(0)

# Theoretical reference (scaled -ζ'/ζ at σ=0.5)
theo = []
for t in t_values:
    s = complex(0.5, t)   # on the critical line for fair comparison
    z = mpmath.zeta(s)
    zd = mpmath.zeta(s, derivative=1)
    log_deriv = -zd / z
    theo.append(float(log_deriv.real) * 4 / 15)

# ====================== PLOT ======================
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

ax.plot(t_values, res_base10, 'darkblue', lw=1.8, label='Global resonance (base-10 primes)')
ax.plot(t_values, res_index, 'green', lw=1.8, label='Global resonance (pure index m)')
ax.plot(t_values, theo, 'orange', lw=1.5, linestyle='--', label=r'Scaled theoretical −ζ′/ζ (×4/15)')

for z in known_zeros:
    ax.axvline(z, color='red', linestyle='--', alpha=0.7, label='Known zeta zero' if z == known_zeros[0] else "")

ax.set_xlabel('t (imaginary part)')
ax.set_ylabel('Signal Strength')
ax.set_title('Base-10 vs. Pure-Index Resonance from Algebraic Ideal Holes')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('base10_vs_pure_index_resonance.png', dpi=300, bbox_inches='tight')
plt.show()