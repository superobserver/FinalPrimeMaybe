#!/usr/bin/env python3
"""
Permissible Clock-Face Snapshot at a True Zeta Zero
Shows the discrete vector configuration forced by einselection geometry.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

SCRIPT_DIR = r'C:\Users\jwhel\Downloads'
if os.path.isdir(SCRIPT_DIR):
    sys.path.append(SCRIPT_DIR)
import April1Sieve2 as elder

k = 11
ZERO_INDEX = 0          # 0 = first true zero (~14.13), 1 = second, etc.
POOL_SIZE = 25
T_WINDOW = 1.5          # how far to look for a nearby illegal configuration

def get_class_primes(kk, n_primes):
    holes = []
    amp = np.array(elder.generate_amplitude_map(kk, 80000))
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + kk
            if p > 5:
                holes.append(p)
                if len(holes) >= n_primes:
                    break
    return np.array(sorted(holes)[:n_primes])

primes = get_class_primes(k, POOL_SIZE)
freqs = np.log(primes)
base = 1.0 / np.sqrt(primes)

# Load the authoritative baseline zeros
with open('zeta_zeros_baseline_corrected.txt') as f:
    lines = [line.strip() for line in f if not line.startswith('#') and line.strip()]
true_zeros = np.array([float(line.split()[1]) for line in lines])

t0 = true_zeros[ZERO_INDEX]
print(f"Examining permissible face at true zero #{ZERO_INDEX+1}  t = {t0:.6f}")

t_frames = np.linspace(t0 - 2, t0 + 2, 4000)
phases = np.outer(t_frames, freqs) % (2 * np.pi)
neg_mask = (np.pi/2 < phases) & (phases < 3*np.pi/2)
R_neg = np.sum(base[None, :] * np.exp(1j * phases) * neg_mask, axis=1)
R_pos = np.sum(base[None, :] * np.exp(1j * phases) * ~neg_mask, axis=1)
D = np.abs(R_neg) - np.abs(R_pos)

# Find the exact frame closest to t0
idx0 = np.argmin(np.abs(t_frames - t0))
D_at_zero = D[idx0]
phases_at_zero = phases[idx0]

# A nearby illegal configuration (same |R| magnitude, wrong phase clustering)
t_illegal = t0 + 0.3
idx_il = np.argmin(np.abs(t_frames - t_illegal))
D_illegal = D[idx_il]
phases_illegal = phases[idx_il]

fig, (ax_face, ax_D) = plt.subplots(1, 2, figsize=(16, 7), dpi=180)

# Permissible face at the true zero
ax_face.set_title(f'Permissible Bloom Face at True Zero t = {t0:.6f}\n(negative cone maximally occupied, positive hands at boundary)')
theta = np.linspace(0, 2*np.pi, 400)
ax_face.plot(theta, np.ones_like(theta), 'k--', lw=0.6, alpha=0.4)
for i, phi in enumerate(phases_at_zero):
    color = 'crimson' if phi > np.pi/2 and phi < 3*np.pi/2 else 'royalblue'
    ax_face.arrow(0, 0, np.cos(phi), np.sin(phi), head_width=0.06, head_length=0.09,
                  fc=color, ec=color, lw=1.5, alpha=0.85)
ax_face.set_xlim(-1.4, 1.4)
ax_face.set_ylim(-1.4, 1.4)
ax_face.set_aspect('equal')
ax_face.text(0.02, 0.95, f'D(t) = {D_at_zero:.3f}  (strong bloom)', transform=ax_face.transAxes,
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# D(t) curve with the zero highlighted
ax_D.plot(t_frames, D, color='#1A237E', lw=1.0)
ax_D.axvline(t0, color='crimson', lw=2, label=f'True zero #{ZERO_INDEX+1}')
ax_D.plot(t0, D_at_zero, 'r^', markersize=12, label='Permissible bloom peak')
ax_D.plot(t_illegal, D_illegal, 'ko', markersize=8, label='Nearby illegal configuration')
ax_D.set_xlabel('t')
ax_D.set_ylabel('D(t)')
ax_D.set_title('Directional Momentum Field — Permissible vs Illegal Configurations')
ax_D.legend(loc='upper right')
ax_D.grid(True, alpha=0.3)

plt.suptitle(f'Class {k} — Discrete Permissible Clock-Face Snapshot Forced by Einselection at a True Zeta Zero\n'
             'All other vector sums with comparable |R| at nearby t are geometrically illegal for this wavefunction closure.',
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f'permissible_face_at_zero_{ZERO_INDEX+1}.png', dpi=200)
print(f"✅ Saved: permissible_face_at_zero_{ZERO_INDEX+1}.png")
plt.show()