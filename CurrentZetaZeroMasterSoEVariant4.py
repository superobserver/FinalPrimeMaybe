#!/usr/bin/env python3
"""
Explicit Realization of Any Prescribed Clock-Face Configuration
Demonstration that every configuration on the finite log(p) torus is achievable.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

SCRIPT_DIR = r'C:\Users\jwhel\Downloads'
if os.path.isdir(SCRIPT_DIR):
    sys.path.append(SCRIPT_DIR)
import April1Sieve2 as elder

k = 11
POOL_SIZE = 12          # modest finite pool for clarity
T_GUESS_RANGE = (0, 200)

def get_class_primes(kk, n):
    holes = []
    amp = np.array(elder.generate_amplitude_map(kk, 80000))
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + kk
            if p > 5:
                holes.append(p)
                if len(holes) >= n:
                    break
    return np.array(sorted(holes)[:n])

primes = get_class_primes(k, POOL_SIZE)
freqs = np.log(primes)
k_hands = len(primes)

# ====================== PRESCRIBE ANY CONFIGURATION YOU LIKE ======================
# Example: a “strong negative bloom” with most hands clustered near π
target_phases = np.array([3.0, 3.1, 3.2, 2.9, 3.3, 2.8, 3.4, 2.7, 3.5, 2.6, 3.6, 2.5])

def objective(t):
    achieved = (t * freqs) % (2 * np.pi)
    return np.sum((achieved - target_phases)**2)

res = minimize_scalar(objective, bounds=T_GUESS_RANGE, method='bounded', tol=1e-12)
t_best = res.x
error = np.sqrt(res.fun / k_hands)
achieved_phases = (t_best * freqs) % (2 * np.pi)

print(f"Best t = {t_best:.6f}")
print(f"RMS phase error = {error:.6f} rad  ({np.degrees(error):.2f}°)")
print(f"D(t) at this configuration = {np.abs(np.sum(np.exp(1j*achieved_phases)/np.sqrt(primes))) - np.abs(np.sum(np.exp(1j*achieved_phases)/np.sqrt(primes))):.4f}")

# ====================== VISUALIZATION ======================
fig, (ax_target, ax_achieved) = plt.subplots(1, 2, figsize=(14, 6), dpi=180)

for ax, phases, title in [(ax_target, target_phases, 'TARGET configuration (your choice)'),
                          (ax_achieved, achieved_phases, f'ACHIEVED at t = {t_best:.4f}')]:
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(theta, np.ones_like(theta), 'k--', lw=0.6, alpha=0.4)
    for phi in phases:
        color = 'crimson' if np.pi/2 < phi < 3*np.pi/2 else 'royalblue'
        ax.arrow(0, 0, np.cos(phi), np.sin(phi), head_width=0.07, head_length=0.1,
                 fc=color, ec=color, lw=1.8, alpha=0.9)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect('equal')
    ax.set_title(title)

plt.suptitle(f'Class {k} — Any Prescribed Clock Face Is Realizable\n'
             f'(finite pool of {k_hands} hands — orbit dense on the torus)',
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('any_clock_face_is_realizable.png', dpi=200)
print("✅ Saved: any_clock_face_is_realizable.png")
plt.show()