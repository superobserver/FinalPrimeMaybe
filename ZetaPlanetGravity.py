#!/usr/bin/env python3
"""
Tidal Pull Animation — Maximum Lateral Force at Zeta Zero
Each log(p) mass exerts a vector force; the resultant pulls the central body maximally at π.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

MAX_M = 20000
N_HANDS = 36
T_MIN, T_MAX, DT = 13.5, 15.0, 0.008
T_ZERO = 14.134725

# Generate holes (same as before)
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
holes = []
for k in classes:
    amp = elder.generate_amplitude_map(k, MAX_M)
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 5:
                holes.append(p)
holes = np.array(sorted(set(holes)))[:N_HANDS * 5]
holes = np.array(sorted(holes)[:N_HANDS])
lengths = 1.0 / np.sqrt(holes)

t_frames = np.arange(T_MIN, T_MAX, DT)
complex_sum_frames = np.zeros(len(t_frames), dtype=complex)
S_frames = np.zeros(len(t_frames))
logp = np.log(holes)
sqrtp = np.sqrt(holes)
for i, t in enumerate(t_frames):
    exp_i = np.exp(1j * t * logp)
    contrib = exp_i / sqrtp
    complex_sum_frames[i] = np.sum(contrib)
    S_frames[i] = np.real(complex_sum_frames[i])

# Animation
fig = plt.figure(figsize=(10, 10), dpi=220)
ax = fig.add_subplot(111, projection='polar')
ax.set_title('Tidal Pull Model — Maximum Lateral Force at First Zeta Zero\n'
             'Masses align at π → deepest negative pull', pad=40)
ax.set_rmax(2.0)
ax.grid(True, alpha=0.3)

# Clock hands (masses)
lines = [ax.plot([], [], 'b-', lw=1.8, alpha=0.85)[0] for _ in holes]
scatters = [ax.scatter([], [], s=50, color='blue', alpha=0.95, zorder=3) for _ in holes]

# Force arrows (pulls toward center)
force_arrows = [ax.plot([], [], 'r-', lw=1.2, alpha=0.6)[0] for _ in holes]

# Resultant tidal force vector (thick gold/red)
resultant_line, = ax.plot([], [], 'gold', lw=8, solid_capstyle='round', alpha=0.95, zorder=5)
resultant_tip = ax.scatter([], [], s=180, color='gold', marker='o', zorder=6)

# Central body (displaced by net force)
central_body = ax.scatter([0], [0], s=300, color='black', zorder=10)

# Red reference line at π
ax.plot([np.pi, np.pi], [0, 2.0], 'r--', lw=3, alpha=0.7)

s_text = ax.text(0.5, 1.14, '', transform=ax.transAxes, ha='center', fontsize=16, fontweight='bold')
t_text = ax.text(0.5, 1.08, '', transform=ax.transAxes, ha='center', fontsize=12)

def animate(frame_idx):
    t = t_frames[frame_idx]
    angles = (t * np.log(holes)) % (2 * np.pi)
    R = complex_sum_frames[frame_idx]
    arg = np.angle(R)
    mag = np.abs(R)
    visual_mag = 0.8 + 2.5 * (abs(S_frames[frame_idx]) ** 1.5)   # strong amplification for audience
    visual_mag = min(visual_mag, 1.95)

    # Update hands and force arrows
    for i, theta in enumerate(angles):
        lines[i].set_data([0, theta], [0, lengths[i]])
        scatters[i].set_offsets([[theta, lengths[i]]])
        # force arrow from mass to center
        force_arrows[i].set_data([theta, 0], [lengths[i], 0])

    # Resultant vector
    resultant_line.set_data([0, arg], [0, visual_mag])
    resultant_tip.set_offsets([[arg, visual_mag]])

    # Central body displaced by resultant (tiny offset for visual effect)
    central_body.set_offsets([[arg * 0.08, visual_mag * 0.08]])

    # Live values
    S_val = S_frames[frame_idx]
    s_text.set_text(f'S(t) = {S_val:7.3f}   |R(t)| = {mag:6.3f}')
    t_text.set_text(f't = {t:.5f}')
    if abs(t - T_ZERO) < 0.08:
        s_text.set_color('red')
    else:
        s_text.set_color('white')

    return lines + scatters + force_arrows + [resultant_line, resultant_tip, central_body, s_text, t_text]

ani = FuncAnimation(fig, animate, frames=len(t_frames), interval=25, blit=False, repeat=True)
plt.tight_layout()
ani.save('tidal_pull_clock_hands.mp4', writer='ffmpeg', fps=30, dpi=220)
plt.show()

print("Tidal-pull animation saved as 'tidal_pull_clock_hands.mp4'")