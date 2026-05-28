#!/usr/bin/env python3
"""
Clock-Face Phase Alignment at Zeta Zeros
Visualizes the stack of log(p) as clock hands on the unit circle.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
max_m = 30000                  # enough for first ~1000 holes
t_zero = 14.134725             # first known zero
t_random = 14.0
n_clocks = 12                  # number of clock faces to show
# ===================================================

# Get deterministic holes
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
holes = []
for k in classes:
    amp = elder.generate_amplitude_map(k, max_m)
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 1:
                holes.append(p)
holes = sorted(set(holes))[:n_clocks * 10]   # plenty for selection

print(f"Using {len(holes)} holes for clock-face visualization.")

def get_phases(t, primes):
    return (t * np.log(primes)) % (2 * np.pi)

phases_zero = get_phases(t_zero, holes)
phases_rand = get_phases(t_random, holes)

# ====================== PLOT ======================
fig = plt.figure(figsize=(14, 8), dpi=300)

# Left: Clock faces at true zero
ax1 = fig.add_subplot(1, 2, 1, projection='polar')
ax1.set_title(f'Clock-Face Alignment at True Zero t = {t_zero:.6f}\n'
              '(phases cluster near π → sharp trough)', pad=20)
for i in range(n_clocks):
    theta = phases_zero[i]
    r = 1.0
    ax1.plot([0, theta], [0, r], 'b-', lw=1.5, alpha=0.7)
    ax1.scatter(theta, r, s=30, color='blue', alpha=0.7)
    # faint clock circle
    circle = Circle((0, 0), 1.05, transform=ax1.transData._b, color='gray', fill=False, lw=0.5, alpha=0.3)
    ax1.add_artist(circle)
ax1.set_rticks([])
ax1.grid(True, alpha=0.3)

# Right: Clock faces at random t (scattered)
ax2 = fig.add_subplot(1, 2, 2, projection='polar')
ax2.set_title(f'Clock Faces at Random t = {t_random:.1f}\n'
              '(phases scattered → no alignment)')
for i in range(n_clocks):
    theta = phases_rand[i]
    r = 1.0
    ax2.plot([0, theta], [0, r], 'orange', lw=1.5, alpha=0.7)
    ax2.scatter(theta, r, s=30, color='orange', alpha=0.7)
    circle = Circle((0, 0), 1.05, transform=ax2.transData._b, color='gray', fill=False, lw=0.5, alpha=0.3)
    ax2.add_artist(circle)
ax2.set_rticks([])
ax2.grid(True, alpha=0.3)

plt.suptitle('Log(p) Stack as Clock Faces — Phase Alignment at Zeta Zeros\n'
             'Algebraic Ideal Holes Produce Coherent Clustering Only at t_n')
plt.tight_layout()
plt.savefig('clock_face_phase_alignment.png', dpi=300, bbox_inches='tight')
plt.show()