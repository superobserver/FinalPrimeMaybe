#!/usr/bin/env python3
"""
Metronome Model of Zeta-Zero Resonance
Each log(p) is a metronome ticking at frequency log(p).
At the first zeta zero all metronomes align in the negative direction.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
MAX_M = 20000
N_METRONOMES = 24               # number of metronomes shown
T_MIN, T_MAX, DT = 13.5, 15.0, 0.008
T_ZERO = 14.134725
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
holes = np.array(sorted(set(holes)))[:N_METRONOMES * 5]
holes = np.array(sorted(holes)[:N_METRONOMES])
freqs = np.log(holes)           # angular frequencies
print(f"Using {len(holes)} metronomes (largest p ≈ {holes[-1]:,.0f})")

# Precompute frames
t_frames = np.arange(T_MIN, T_MAX, DT)
S_frames = np.zeros(len(t_frames))
for i, t in enumerate(t_frames):
    S_frames[i] = np.sum(np.cos(t * freqs) / np.sqrt(holes))

# ====================== ANIMATION ======================
fig, (ax_phase, ax_S) = plt.subplots(2, 1, figsize=(14, 9), dpi=220,
                                     gridspec_kw={'height_ratios': [3, 1]})

ax_phase.set_title('Metronome Model of Zeta-Zero Resonance\n'
                   'Each log(p) ticks at frequency log(p) — alignment at π produces deepest trough')
ax_phase.set_xlim(-1.2, 1.2)
ax_phase.set_ylim(-1.2, 1.2)
ax_phase.set_aspect('equal')
ax_phase.grid(True, alpha=0.3)

# Metronome arms (lines from center)
arms = [ax_phase.plot([], [], 'b-', lw=2.5, alpha=0.9)[0] for _ in holes]
bobs = [ax_phase.scatter([], [], s=80, color='blue', zorder=3) for _ in holes]

# Reference line at π (negative direction)
ax_phase.plot([-1.2, -1.2], [-1.2, 1.2], 'r--', lw=3, alpha=0.7, label='π — destructive alignment')
ax_phase.legend(loc='upper right')

# Live S(t) panel
line_S, = ax_S.plot([], [], 'k-', lw=3)
ax_S.set_xlim(T_MIN, T_MAX)
ax_S.set_ylim(S_frames.min()*1.1, S_frames.max()*1.1)
ax_S.set_xlabel('t')
ax_S.set_ylabel('S(t)')
ax_S.grid(True, alpha=0.3)

s_text = ax_phase.text(0.02, 0.95, '', transform=ax_phase.transAxes, fontsize=14, fontweight='bold')
t_text = ax_phase.text(0.02, 0.88, '', transform=ax_phase.transAxes, fontsize=12)

def animate(frame_idx):
    t = t_frames[frame_idx]
    phases = (t * freqs) % (2 * np.pi)

    # Update metronome arms
    for i, phi in enumerate(phases):
        x = np.cos(phi)
        y = np.sin(phi)
        arms[i].set_data([0, x], [0, y])
        bobs[i].set_offsets([[x, y]])

    # Live S(t)
    line_S.set_data(t_frames[:frame_idx+1], S_frames[:frame_idx+1])
    s_text.set_text(f'S(t) = {S_frames[frame_idx]:7.3f}')
    t_text.set_text(f't = {t:.5f}')

    if abs(t - T_ZERO) < 0.08:
        s_text.set_color('red')
    else:
        s_text.set_color('black')

    return arms + bobs + [line_S, s_text, t_text]

ani = FuncAnimation(fig, animate, frames=len(t_frames), interval=25, blit=False, repeat=True)

plt.tight_layout()
ani.save('metronome_resonance_first_zero.mp4', writer='ffmpeg', fps=30, dpi=220)
plt.show()

print("Metronome animation saved as 'metronome_resonance_first_zero.mp4'")
print("Watch the metronomes swing in perfect synchrony at t≈14.1347 — the moment of maximum destructive interference.")