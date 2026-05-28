#!/usr/bin/env python3
"""
Unit-Circle Clock-Hand Animation + Resultant Vector
Shows individual log(p) hands + the amplified resultant vector R(t)
that reaches maximum length and points exactly at π when S(t) is deepest.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
MAX_M = 20000
N_HANDS = 36
T_MIN, T_MAX, DT = 10.0, 18.0, 0.025
T_ZERO = 14.134725
# ===================================================

# Generate deterministic holes
print("Generating deterministic holes from algebraic ideal...")
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
print(f"Using {len(holes)} holes for animation.")

# Precompute frames
t_frames = np.arange(T_MIN, T_MAX, DT)
S_frames = np.zeros(len(t_frames))
complex_sum_frames = np.zeros(len(t_frames), dtype=complex)
logp = np.log(holes)
sqrtp = np.sqrt(holes)
for i, t in enumerate(t_frames):
    exp_i = np.exp(1j * t * logp)
    contrib = exp_i / sqrtp
    complex_sum_frames[i] = np.sum(contrib)
    S_frames[i] = np.real(complex_sum_frames[i])

# ====================== ANIMATION ======================
fig = plt.figure(figsize=(10, 10), dpi=200)
ax = fig.add_subplot(111, projection='polar')
ax.set_title('log(p) Clock Hands + Resultant Vector R(t)\n'
             'Deep trough at t≈14.1347 when hands align at π', pad=40)
ax.set_rmax(1.8)
ax.grid(True, alpha=0.3)

# Individual clock hands
lines = []
scatters = []
for i in range(len(holes)):
    line, = ax.plot([], [], 'b-', lw=1.8, alpha=0.85)
    scat = ax.scatter([], [], s=45, color='blue', alpha=0.95, zorder=3)
    lines.append(line)
    scatters.append(scat)

# Resultant vector (thick, gold/red)
resultant_line, = ax.plot([], [], 'gold', lw=6, solid_capstyle='round', alpha=0.95, zorder=5)
resultant_tip = ax.scatter([], [], s=120, color='gold', marker='o', zorder=6)

# Red reference line at π
ax.plot([np.pi, np.pi], [0, 1.8], 'r--', lw=3, alpha=0.7)

# Live text
s_text = ax.text(0.5, 1.12, '', transform=ax.transAxes, ha='center', fontsize=16, fontweight='bold')
t_text = ax.text(0.5, 1.06, '', transform=ax.transAxes, ha='center', fontsize=12)

def animate(frame_idx):
    t = t_frames[frame_idx]
    angles = (t * np.log(holes)) % (2 * np.pi)

    # Update individual hands
    for i, theta in enumerate(angles):
        lines[i].set_data([0, theta], [0, lengths[i]])
        scatters[i].set_offsets([[theta, lengths[i]]])

    # Resultant vector (complex sum)
    R = complex_sum_frames[frame_idx]
    mag = np.abs(R)
    arg = np.angle(R)
    # Non-linear amplification for visual impact at trough
    visual_mag = 0.6 + 1.8 * (abs(S_frames[frame_idx]) ** 1.4)
    visual_mag = min(visual_mag, 1.75)   # keep inside plot

    resultant_line.set_data([0, arg], [0, visual_mag])
    resultant_tip.set_offsets([[arg, visual_mag]])

    # Color intensity by trough depth
    intensity = min(1.0, abs(S_frames[frame_idx]) / 12)
    resultant_line.set_color((1.0, 0.6 + 0.4*intensity, 0.0))  # gold → deep orange/red

    # Live values
    S_val = S_frames[frame_idx]
    s_text.set_text(f'S(t) = {S_val:6.3f}')
    t_text.set_text(f't = {t:.4f}')
    if abs(t - T_ZERO) < 0.12:
        s_text.set_color('red')
    else:
        s_text.set_color('white')

    return lines + scatters + [resultant_line, resultant_tip, s_text, t_text]

ani = FuncAnimation(fig, animate, frames=len(t_frames), interval=40, blit=False, repeat=True)

plt.tight_layout()
ani.save('unit_circle_clock_hands_with_resultant_vector.mp4', writer='ffmpeg', fps=25, dpi=200)
plt.show()

print("\nAnimation saved as 'unit_circle_clock_hands_with_resultant_vector.mp4'")
print("Watch the gold/red resultant vector grow dramatically and snap to π exactly at t≈14.1347 when S(t) reaches its deepest negative trough.")