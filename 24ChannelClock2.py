#!/usr/bin/env python3
"""
Class 11 24-Channel Quantum Clock Marking System — Fixed
24 independent grating clocks + probability landscape of holes + resonance D(t).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

k = 11
MAX_N = 1200          # index line length for visualization
T_MAX = 220.0
DT = 0.018

# Get exact 24 pairs for class 11
pairs = elder.get_pairs_for_class(k)
print(f"Class {k} — 24 marking pairs: {pairs}")

# Compute correct ground-state y0, p, q for each pair
channels = []
for z, o in pairs:
    l = 180 - (z + o)
    m = 90 - (z + o) + (z * o // 90)
    y0 = 90 * 1 * 1 - l * 1 + m          # ground-state launch
    p = z + 90 * (1 - 1)                 # period for z-channel
    q = o + 90 * (1 - 1)                 # period for o-channel
    channels.append((y0, p, q, z, o))

# Generate amplitude map and convert to NumPy array (fix for division)
amp = np.array(elder.generate_amplitude_map(k, MAX_N))   # <-- fixed here

# Resonance data for class 11 (p and q as frequency operators)
class_periods = []
for y0, p, q, z, o in channels:
    class_periods.append(p)
    class_periods.append(q)
class_periods = np.unique(class_periods)
class_freqs = np.log(class_periods)
class_base = 1.0 / np.sqrt(class_periods)

t_frames = np.arange(0.1, T_MAX, DT)
phases = np.outer(t_frames, class_freqs) % (2 * np.pi)
R_complex = np.sum(class_base[None, :] * np.exp(1j * phases), axis=1)

fig = plt.figure(figsize=(20, 11), dpi=200)
gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])

ax_clock = fig.add_subplot(gs[0, 0], projection='polar')
ax_amp = fig.add_subplot(gs[1, 0])
ax_res = fig.add_subplot(gs[0, 1], projection='polar')
ax_prob = fig.add_subplot(gs[1, 1])

ax_clock.set_title(f'Class {k} — 24 Independent Grating Clocks\nLive Noon-Strikes', fontsize=14)
ax_res.set_title(f'Class {k} Resonance Clock · Golden Vector R(t)', fontsize=14)
ax_prob.set_title('Probability Landscape of Unmarked Indices (Holes)')

def animate(frame):
    n = frame * 1   # step along index line

    # LEFT: Index-line clock with 24 independent channels
    ax_clock.clear()
    theta = np.linspace(0, 2*np.pi, 500)
    ax_clock.plot(theta, np.ones(500), 'k--', lw=1, alpha=0.3)
    strikes = 0
    for y0, p, q, z, o in channels:
        if n < y0:
            continue
        phase_p = ((n - y0) % p) * (2 * np.pi / p)
        phase_q = ((n - y0) % q) * (2 * np.pi / q)
        ax_clock.plot([0, phase_p], [0, 1.0], 'r-', lw=1.8, alpha=0.7)
        ax_clock.plot([0, phase_q], [0, 1.0], 'r-', lw=1.8, alpha=0.7)
        if abs(phase_p) < 0.08 or abs(phase_p - 2*np.pi) < 0.08:
            ax_clock.scatter(phase_p, 1.2, s=80, color='gold')
            strikes += 1
        if abs(phase_q) < 0.08 or abs(phase_q - 2*np.pi) < 0.08:
            ax_clock.scatter(phase_q, 1.2, s=80, color='gold')
            strikes += 1
    ax_clock.set_title(f'Class {k} · 24 Grating Clocks · n = {n} · Strikes: {strikes}')
    ax_clock.set_rmax(1.8)

    # Amplitude map
    ax_amp.clear()
    current_amp = amp#[:300]
    ax_amp.bar(np.arange(len(current_amp)), current_amp, width=1, color='orange', alpha=0.7)
    ax_amp.set_title(f'Class {k} Amplitude Map · n = 1..300')
    ax_amp.set_ylabel('Noon-strikes')
    holes_in_view = np.sum(current_amp == 0)
    ax_amp.text(0.05, 0.9, f'Holes in view: {holes_in_view}', transform=ax_amp.transAxes, fontsize=12)

    # Probability landscape (1 - multiplicity/24)
    current_amp_np = np.array(current_amp)   # ensure NumPy
    prob_unmarked = 1.0 - current_amp_np / 24.0
    ax_prob.clear()
    ax_prob.plot(np.arange(len(prob_unmarked)), prob_unmarked, 'b-', lw=1.5)
    ax_prob.set_title('Probability of Unmarked Index (Hole)')
    ax_prob.set_ylabel('P(unmarked at n)')
    ax_prob.set_ylim(0, 1.05)

    # RIGHT: Resonance clock
    ax_res.clear()
    ax_res.plot(theta, np.ones(500), 'k--', lw=1, alpha=0.3)
    r_phases = (frame * 0.3 * class_freqs) % (2 * np.pi)
    for phi in r_phases:
        ax_res.plot([0, phi], [0, 1.0], 'b-', lw=1.8, alpha=0.6)
    r_angle = np.angle(R_complex[frame % len(t_frames)])
    r_mag = 0.8 + 1.5 * abs(R_complex[frame % len(t_frames)])
    ax_res.plot([0, r_angle], [0, r_mag], 'gold', lw=5)
    ax_res.scatter(r_angle, r_mag, s=180, color='gold')
    ax_res.set_title(f'Class {k} Resonance · D(t) = {abs(R_complex[frame % len(t_frames)]):.3f}')

    return []

ani = FuncAnimation(fig, animate, frames=300, interval=40, blit=False, repeat=True)

ani.save('class11_24_grating_clocks_probability_landscape.mp4', writer='ffmpeg', fps=25, dpi=180)
print("✅ Video saved as 'class11_24_grating_clocks_probability_landscape.mp4'")

plt.suptitle(f"Class {k} — 24 Independent Grating Clocks + Probability Landscape of Holes", fontsize=16)
plt.tight_layout()
plt.show()