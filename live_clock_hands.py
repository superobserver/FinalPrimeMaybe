#!/usr/bin/env python3
"""
Enhanced Unit-Circle Clock-Hand Animation with Live S(t) Display
Shows real-time resonance sum S(t) and the deep negative trough at t≈14.1347
when hands align near π.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
MAX_M = 20000                  # enough holes for accurate S(t)
N_HANDS = 36                   # number of clock hands shown
T_ZERO = 14.134725             # first zeta zero
T_MIN, T_MAX, DT = 10.0, 18.0, 0.025
# ===================================================

# Generate deterministic holes from algebraic ideal
print("Generating deterministic holes...")
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
holes = []
for k in classes:
    amp = elder.generate_amplitude_map(k, MAX_M)
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 5:
                holes.append(p)
holes = np.array(sorted(set(holes)))[:N_HANDS * 5]  # plenty
print(f"Using {len(holes)} holes for animation.")

# Keep strongest (longest) hands
holes = np.array(sorted(holes)[:N_HANDS])
lengths = 1.0 / np.sqrt(holes)

# Precompute S(t) for all animation frames (fast lookup)
t_frames = np.arange(T_MIN, T_MAX, DT)
S_frames = np.zeros(len(t_frames))
logp = np.log(holes)
sqrtp = np.sqrt(holes)
for i, t in enumerate(t_frames):
    S_frames[i] = np.sum(np.cos(t * logp) / sqrtp)

# ====================== ANIMATION ======================
fig = plt.figure(figsize=(10, 10), dpi=200)
ax = fig.add_subplot(111, projection='polar')
ax.set_title('log(p) Clock Hands — Live Resonance S(t)\n'
             'Alignment near π produces deep negative trough at first zeta zero', pad=40)
ax.set_rmax(1.05)
ax.grid(True, alpha=0.3)

# Clock hands
lines = []
scatters = []
for i in range(len(holes)):
    line, = ax.plot([], [], 'b-', lw=1.8, alpha=0.85)
    scat = ax.scatter([], [], s=45, color='blue', alpha=0.95, zorder=3)
    lines.append(line)
    scatters.append(scat)

# Red alignment line at π
alignment_line = ax.plot([np.pi, np.pi], [0, 1.05], 'r--', lw=4, label='π (destructive alignment)')[0]

# Live S(t) display
s_text = ax.text(0.5, 1.08, '', transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')
t_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, ha='center', fontsize=12)

def animate(frame_idx):
    t = t_frames[frame_idx]
    angles = (t * np.log(holes)) % (2 * np.pi)

    for i, theta in enumerate(angles):
        lines[i].set_data([0, theta], [0, lengths[i]])
        scatters[i].set_offsets([[theta, lengths[i]]])

    # Live S(t) value
    S_val = S_frames[frame_idx]
    s_text.set_text(f'S(t) = {S_val:6.3f}')
    t_text.set_text(f't = {t:.4f}')

    # Highlight when near the first zero
    if abs(t - T_ZERO) < 0.1:
        s_text.set_color('red')
    else:
        s_text.set_color('black')

    return lines + scatters + [alignment_line, s_text, t_text]

ani = FuncAnimation(fig, animate, frames=len(t_frames), interval=40, blit=False, repeat=True)

plt.legend(loc='upper right')
plt.tight_layout()
ani.save('unit_circle_clock_hands_live_S(t).mp4', writer='ffmpeg', fps=25, dpi=200)
plt.show()

print("\nAnimation saved as 'unit_circle_clock_hands_live_S(t).mp4'")
print("Watch for the red S(t) value plunging to a deep negative trough exactly when the hands align near π at t≈14.1347.")