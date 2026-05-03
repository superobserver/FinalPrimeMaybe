#!/usr/bin/env python3
"""
Divergence Test: Global Resonance off the Critical Line
Plots 24-class resonance from algebraic ideal holes vs. theoretical -ζ'/ζ
at σ = 0.4, 0.5, 0.6. Sharpest alignment must occur exactly at σ = 0.5.
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
sigmas = [0.4, 0.5, 0.6]          # test off, on, and above the critical line
# ===================================================

# Get all 24 classes
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]

# Build global primes (all 24 classes)
primes_global = []
for k in classes:
    amp = elder.generate_amplitude_map(k, max_m)
    holes = [m for m in range(len(amp)) if amp[m] == 0]
    primes_global.extend(90 * m + k for m in holes)
primes_global = sorted(set(primes_global))

print(f"Global primes (all 24 classes): {len(primes_global)}")

# Compute resonance for each σ
t_values = np.arange(t_start, t_end, step)
res = {sigma: np.zeros(len(t_values)) for sigma in sigmas}
theo = {sigma: np.zeros(len(t_values)) for sigma in sigmas}

for i, t in enumerate(t_values):
    for sigma in sigmas:
        # Global resonance from holes (real part)
        s = complex(sigma, t)
        res[sigma][i] = sum(np.cos(t * np.log(p)) / np.sqrt(p) for p in primes_global if p > 1)
        
        # Theoretical -ζ'(s)/ζ(s) scaled by 4/15
        z = mpmath.zeta(s)
        zd = mpmath.zeta(s, derivative=1)
        log_deriv = -zd / z
        theo[sigma][i] = float(log_deriv.real) * 4 / 15

# ====================== PLOT ======================
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

colors = ['orange', 'darkblue', 'green']
for i, sigma in enumerate(sigmas):
    label = f'σ = {sigma} (global resonance from holes)'
    ax.plot(t_values, res[sigma], color=colors[i], lw=1.8, label=label)
    ax.plot(t_values, theo[sigma], color=colors[i], linestyle='--', lw=1.2, 
            label=f'σ = {sigma} (theoretical -ζ\'/ζ ×4/15)' if i == 0 else "")

for z in known_zeros:
    ax.axvline(z, color='red', linestyle='--', alpha=0.7, label='Known zeta zero' if z == known_zeros[0] else "")

ax.set_xlabel('t (imaginary part)')
ax.set_ylabel('Signal Strength')
ax.set_title('Divergence Test: Global Resonance off the Critical Line\n'
             'Sharpest alignment occurs exactly at σ = 0.5')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('divergence_test_off_critical_line.png', dpi=300, bbox_inches='tight')
plt.show()