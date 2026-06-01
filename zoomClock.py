#!/usr/bin/env python3
"""
Amplified Clock-Hands Animation — Proportional Length Scaling
Clear visibility of more hands ("p orbitals") with adjustable amplification.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
MAX_M = 80000
N_HANDS = 120                  # increase freely — amplification keeps it clean
AMP_FACTOR = 7.0               # change this to zoom in/out (try 4.0 or 12.0)
T_MIN, T_MAX, DT = 13.5, 15.0, 0.012
known_zeros = [14.134725, 21.022039, 25.01085, 30.42487, 32.93506, 37.58617, 40.91871, 43.32707, 48.0051, 49.77383, 52.9703]
# ===================================================

# Generate deterministic holes
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
holes = []
for k in classes:
    amp = elder.generate_amplitude_map(k, MAX_M)
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 5:
                holes.append(p)
holes = np.array(sorted(set(holes)))[:N_HANDS]
freqs = np.log(holes)
base_lengths = 1.0 / np.sqrt(holes)
visual_lengths = base_lengths * AMP_FACTOR #+ 0.15   # offset prevents zero-length collapse

print(f"Showing {N_HANDS} amplified hands (AMP_FACTOR = {AMP_FACTOR})")

t_frames = np.arange(T_MIN, T_MAX, DT)
S_frames = np.zeros(len(t_frames))
for i, t in enumerate(t_frames):
    S_frames[i] = np.sum(np.cos(t * freqs) / np.sqrt(holes))

fig = plt.figure(figsize=(10, 10), dpi=240)
ax = fig.add_subplot(111, projection='polar')
ax.set_title(f'Amplified Clock Hands (×{AMP_FACTOR}) — Bloom & Alignment at Zeta Zero\n'
             'More hands • Clear structure • Variable speed + pause', pad=30)
ax.set_rmax(visual_lengths.max() * 1.15)
ax.grid(True, alpha=0.3)
ax.plot([np.pi, np.pi], [0, ax.get_rmax()], 'r--', lw=3, alpha=0.8, label='π — trough')

lines = [ax.plot([], [], 'b-', lw=1.8)[0] for _ in holes]
scatters = [ax.scatter([], [], s=45, color='blue', zorder=3) for _ in holes]
result_line, = ax.plot([], [], 'gold', lw=7, solid_capstyle='round', zorder=5)
result_tip = ax.scatter([], [], s=180, color='gold', marker='o', zorder=6)

s_text = ax.text(0.5, 1.12, '', transform=ax.transAxes, ha='center', fontsize=16, fontweight='bold')
t_text = ax.text(0.5, 1.06, '', transform=ax.transAxes, ha='center', fontsize=12)

def animate(frame_idx):
    t = t_frames[frame_idx]
    angles = (t * freqs) % (2 * np.pi)

    for i, theta in enumerate(angles):
        lines[i].set_data([0, theta], [0, visual_lengths[i]])
        scatters[i].set_offsets([[theta, visual_lengths[i]]])

    R = np.sum(np.exp(1j * t * freqs) / np.sqrt(holes))
    arg = np.angle(R)
    visual_mag = min(ax.get_rmax() * 0.95, 0.7 + 2.8 * abs(S_frames[frame_idx])**1.5)
    result_line.set_data([0, arg], [0, visual_mag])
    result_tip.set_offsets([[arg, visual_mag]])

    S_val = S_frames[frame_idx]
    s_text.set_text(f'S(t) = {S_val:7.3f}')
    t_text.set_text(f't = {t:.5f}')

    if any(abs(t - z) < 0.06 for z in known_zeros):
        s_text.set_color('red')
        t_text.set_text(f't = {t:.5f}  ← ZETA ZERO (pause)')

    return lines + scatters + [result_line, result_tip, s_text, t_text]

ani = FuncAnimation(fig, animate, frames=len(t_frames), interval=28, blit=False, repeat=True)
plt.tight_layout()
ani.save('amplified_clock_hands_bloom1.mp4', writer='ffmpeg', fps=30, dpi=240)
plt.show()

print("Animation saved as 'amplified_clock_hands_bloom.mp4'")
print(f"• {N_HANDS} hands shown • Lengths amplified ×{AMP_FACTOR} • Clear bloom structure")