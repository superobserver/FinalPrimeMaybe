#!/usr/bin/env python3
"""
Metronome Model with Negative Half-Plane Highlight
Shows exactly when each metronome is in negative vector space and how coherent clustering at π produces the zeta zero.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arc

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

MAX_M = 20000
N_METRONOMES = 24
T_MIN, T_MAX, DT = 13.5, 15.0, 0.008
T_ZERO = 14.134725

# Generate holes
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
holes = []
for k in classes:
    amp = elder.generate_amplitude_map(k, MAX_M)
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 5:
                holes.append(p)
holes = np.array(sorted(set(holes)))[:N_METRONOMES * 5]
holes = np.array(sorted(holes)[:N_METRONOMES])
freqs = np.log(holes)

t_frames = np.arange(T_MIN, T_MAX, DT)
S_frames = np.zeros(len(t_frames))
for i, t in enumerate(t_frames):
    S_frames[i] = np.sum(np.cos(t * freqs) / np.sqrt(holes))

fig, (ax_phase, ax_S) = plt.subplots(2, 1, figsize=(14, 9), dpi=220,
                                     gridspec_kw={'height_ratios': [3, 1]})

ax_phase.set_title('Metronome Model — Negative Half-Plane Highlight\n'
                   'Coherent clustering near π (not raw count) produces the zeta zero')
ax_phase.set_xlim(-1.3, 1.3)
ax_phase.set_ylim(-1.3, 1.3)
ax_phase.set_aspect('equal')
ax_phase.grid(True, alpha=0.3)

# Negative half-plane shade
ax_phase.add_patch(Arc((0,0), 2.4, 2.4, theta1=90, theta2=270, color='red', alpha=0.15, linewidth=0))

arms = [ax_phase.plot([], [], 'b-', lw=2.5, alpha=0.9)[0] for _ in holes]
bobs = [ax_phase.scatter([], [], s=80, color='blue', zorder=3) for _ in holes]

# Red reference line at π
ax_phase.plot([-1.3, -1.3], [-1.3, 1.3], 'r--', lw=3, alpha=0.7)

line_S, = ax_S.plot([], [], 'k-', lw=3)
ax_S.set_xlim(T_MIN, T_MAX)
ax_S.set_ylim(S_frames.min()*1.1, S_frames.max()*1.1)
ax_S.set_xlabel('t')
ax_S.set_ylabel('S(t)')
ax_S.grid(True, alpha=0.3)

s_text = ax_phase.text(0.02, 0.95, '', transform=ax_phase.transAxes, fontsize=14, fontweight='bold')
neg_count_text = ax_phase.text(0.02, 0.88, '', transform=ax_phase.transAxes, fontsize=12)

def animate(frame_idx):
    t = t_frames[frame_idx]
    phases = (t * freqs) % (2 * np.pi)

    neg_count = 0
    for i, phi in enumerate(phases):
        x = np.cos(phi)
        y = np.sin(phi)
        arms[i].set_data([0, x], [0, y])
        bobs[i].set_offsets([[x, y]])
        if np.pi/2 < phi < 3*np.pi/2:   # negative half-plane
            neg_count += 1

    line_S.set_data(t_frames[:frame_idx+1], S_frames[:frame_idx+1])
    s_text.set_text(f'S(t) = {S_frames[frame_idx]:7.3f}')
    neg_count_text.set_text(f'Negative-space metronomes: {neg_count}/{N_METRONOMES}')

    if abs(t - T_ZERO) < 0.08:
        s_text.set_color('red')
    else:
        s_text.set_color('black')

    return arms + bobs + [line_S, s_text, neg_count_text]

ani = FuncAnimation(fig, animate, frames=len(t_frames), interval=25, blit=False, repeat=True)
plt.tight_layout()
ani.save('metronome_negative_half_plane.mp4', writer='ffmpeg', fps=30, dpi=220)
plt.show()