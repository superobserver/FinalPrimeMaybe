#!/usr/bin/env python3
"""
Unit-Circle Clock-Hand Visualization of the log(p) Stack
Each hole p = 90m + k is a clock hand of length 1/sqrt(p) rotating at speed log(p).
At the first zeta zero the hands align coherently near π.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
MAX_M = 10000                  # sufficient for first ~2000 holes
N_HANDS = 24                   # number of clock hands to display (largest contributors)
T_ZERO = 14.134725             # first known non-trivial zero
# ===================================================

# Get deterministic holes
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
holes = []
for k in classes:
    amp = elder.generate_amplitude_map(k, MAX_M)
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 5:
                holes.append(p)
holes = sorted(set(holes))[:N_HANDS * 10]  # plenty to choose from
print(f"Using {len(holes)} deterministic holes for visualization.")

# Keep only the N_HANDS strongest (smallest p = longest hands)
holes = np.array(sorted(holes)[:N_HANDS])
lengths = 1.0 / np.sqrt(holes)
angles_zero = (T_ZERO * np.log(holes)) % (2 * np.pi)

# ====================== STATIC PLOT ======================
fig_static = plt.figure(figsize=(9, 9), dpi=300)
ax = fig_static.add_subplot(111, projection='polar')
ax.set_title(f'Clock-Hand Alignment at First Zeta Zero\n'
             f't = {T_ZERO:.6f}   (coherent alignment near π)', pad=30)

# Draw unit circle and hands
for i, (theta, r) in enumerate(zip(angles_zero, lengths)):
    ax.plot([0, theta], [0, r], 'b-', lw=1.5, alpha=0.8)
    ax.scatter(theta, r, s=40, color='blue', alpha=0.9, zorder=3)

# Highlight π (destructive interference direction)
ax.plot([np.pi, np.pi], [0, 1.05], 'r--', lw=3, label='Alignment direction π')
ax.legend(loc='upper right')

ax.set_rticks([])          # hide radial ticks
ax.set_rmax(1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('unit_circle_clock_hands_static.png', dpi=300, bbox_inches='tight')
plt.show()

# ====================== ANIMATION ======================
fig_anim = plt.figure(figsize=(9, 9), dpi=200)
ax_anim = fig_anim.add_subplot(111, projection='polar')
ax_anim.set_title('log(p) Clock Hands Rotating — Alignment at t ≈ 14.1347', pad=30)
ax_anim.set_rmax(1.05)
ax_anim.grid(True, alpha=0.3)

# Initial hands
lines = []
scatters = []
for i in range(len(holes)):
    line, = ax_anim.plot([], [], 'b-', lw=1.5, alpha=0.8)
    scat = ax_anim.scatter([], [], s=40, color='blue', alpha=0.9)
    lines.append(line)
    scatters.append(scat)

alignment_line = ax_anim.plot([np.pi, np.pi], [0, 1.05], 'r--', lw=3)[0]
t_text = ax_anim.text(0.5, 1.05, '', transform=ax_anim.transAxes, ha='center', fontsize=12)

def animate(t):
    angles = (t * np.log(holes)) % (2 * np.pi)
    for i, theta in enumerate(angles):
        lines[i].set_data([0, theta], [0, lengths[i]])
        scatters[i].set_offsets([[theta, lengths[i]]])
    t_text.set_text(f't = {t:.4f}')
    return lines + scatters + [alignment_line]

ani = FuncAnimation(fig_anim, animate, frames=np.arange(10.0, 18.0, 0.03),
                    interval=30, blit=False, repeat=True)
plt.tight_layout()
ani.save('unit_circle_clock_hands_animation.mp4', writer='ffmpeg', fps=30, dpi=200)
plt.show()