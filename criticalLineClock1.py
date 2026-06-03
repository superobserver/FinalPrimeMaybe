#!/usr/bin/env python3
"""
Find the next good recurrence of a specific permissible clock-face configuration
(at the first true zeta zero).
"""

import numpy as np
from scipy.optimize import differential_evolution

# Your exact data at t = 14.134725
primes = np.array([7,11,13,17,19,23,29,31,37,41,43,47])
target_phases_deg = np.array([135.9,142.0,277.3,134.5,224.6,19.3,207.0,261.0,44.3,127.5,166.0,238.1])
target_phases = np.radians(target_phases_deg)
t0 = 14.134725

freqs = np.log(primes)

def phase_error(t):
    achieved = (t * freqs) % (2 * np.pi)
    diffs = np.abs(achieved - target_phases)
    diffs = np.minimum(diffs, 2*np.pi - diffs)   # shortest arc on the circle
    return np.sum(diffs**2)

# Search in [t0 + 0.01, t0 + 500] for the best return
bounds = [(t0 + 0.01, t0 + 500)]
res = differential_evolution(phase_error, bounds, tol=1e-12, popsize=30, mutation=0.7)
t_best = res.x[0]
rms_error_deg = np.degrees(np.sqrt(res.fun / len(primes)))
max_error_deg = np.degrees(np.max(np.abs((t_best * freqs) % (2*np.pi) - target_phases)))

# Recompute D(t) at the best return (using the same length weighting)
lengths = 1.0 / np.sqrt(primes)
phases_best = (t_best * freqs) % (2 * np.pi)
neg_mask = (np.pi/2 < phases_best) & (phases_best < 3*np.pi/2)
R_neg = np.sum(lengths[neg_mask] * np.exp(1j * phases_best[neg_mask]))
R_pos = np.sum(lengths[~neg_mask] * np.exp(1j * phases_best[~neg_mask]))
D_best = np.abs(R_neg) - np.abs(R_pos)

print(f"Original t0 = {t0:.6f}")
print(f"Best return t' = {t_best:.6f}   (Δt = {t_best - t0:.2f})")
print(f"RMS phase error = {rms_error_deg:.3f}°")
print(f"Maximum phase error = {max_error_deg:.3f}°")
print(f"D(t') at this recurrence = {D_best:.4f}")

# Quick visual check of the achieved phases vs target
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7,7), dpi=150)
theta = np.linspace(0, 2*np.pi, 400)
ax.plot(theta, np.ones_like(theta), 'k--', lw=0.6, alpha=0.3)
for phi in phases_best:
    color = 'crimson' if np.pi/2 < phi < 3*np.pi/2 else 'royalblue'
    ax.arrow(0, 0, np.cos(phi), np.sin(phi), head_width=0.05, head_length=0.07,
             fc=color, ec=color, lw=1.5, alpha=0.85)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect('equal')
ax.set_title(f'Recurrence of the permissible face at t = {t_best:.4f}\n'
             f'RMS error {rms_error_deg:.2f}°   D = {D_best:.3f}')
plt.tight_layout()
plt.savefig('recurrence_of_permissible_face.png', dpi=180)
print("✅ Saved: recurrence_of_permissible_face.png")
plt.show()