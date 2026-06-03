#!/usr/bin/env python3
"""
Class 11 Global Clock Face — Corrected 24-Channel Marking Clock + Resonance Golden Vector
Each marking rod launches at its exact y0 and rotates with its true period.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

k = 11
MAX_N = 1200          # index line length
EPOCHS = 8            # how many epochs to evolve (adds new rods)
AMP_FACTOR = 1.0

# Get exact 24 pairs for class 11
pairs = elder.get_pairs_for_class(k)
print(f"Class {k} — 24 marking pairs: {pairs}")

# Compute correct ground-state y0, p, q for each pair (x=1 launch)
channels = []
for z, o in pairs:
    l = 180 - (z + o)
    m = 90 - (z + o) + (z * o // 90)
    y0 = 90 * 1 * 1 - l * 1 + m          # ground-state launch position
    p = z + 90 * (1 - 1)                 # period for z-channel
    q = o + 90 * (1 - 1)                 # period for o-channel
    channels.append((y0, p, q, z, o))    # store y0 and periods

# Generate amplitude map for verification
amp = elder.generate_amplitude_map(k, MAX_N)

fig = plt.figure(figsize=(20, 11), dpi=200)
gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])

ax_clock = fig.add_subplot(gs[0, 0], projection='polar')
ax_amp = fig.add_subplot(gs[1, 0])
ax_res = fig.add_subplot(gs[0, 1], projection='polar')
ax_info = fig.add_subplot(gs[1, 1])

ax_clock.set_title(f'Class {k} Index-Line Marking Clock\n24 Channels · Launch at y0', fontsize=14)
ax_res.set_title(f'Class {k} Resonance Clock · Golden Vector R(t)', fontsize=14)

# Resonance data for class 11 primes
class_primes = np.array([p for p in [90*m + k for m in range(MAX_N)] if amp[m] == 0 and p > 5])
class_freqs = np.log(class_primes[:24])
class_base = 1.0 / np.sqrt(class_primes[:24])

def animate(frame):
    n = frame * 1   # step along index line n

    # LEFT: Index-line clock with 24 marking channels
    ax_clock.clear()
    theta = np.linspace(0, 2*np.pi, 500)
    ax_clock.plot(theta, np.ones(500), 'k--', lw=1, alpha=0.3)
    strikes = 0
    for y0, p, q, z, o in channels:
        if n < y0:
            continue  # rod not yet launched
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
    ax_clock.set_title(f'Class {k} · 24 Marking Channels · n = {n} · Strikes: {strikes}')
    ax_clock.set_rmax(1.8)

    # Amplitude map (multiplicity)
    ax_amp.clear()
    current_amp = amp[:300]
    ax_amp.bar(np.arange(len(current_amp)), current_amp, width=1, color='orange', alpha=0.7)
    ax_amp.set_title(f'Class {k} Amplitude Map (Multiplicity) · n = 1..300')
    ax_amp.set_ylabel('Noon-strikes (A_{11}(n))')
    holes_in_view = np.sum(current_amp[:300] == 0)
    ax_amp.text(0.05, 0.9, f'Holes in view: {holes_in_view}', transform=ax_amp.transAxes, fontsize=12, bbox=dict(facecolor='white'))

    # RIGHT: Resonance clock for class 11 primes
    ax_res.clear()
    ax_res.plot(theta, np.ones(500), 'k--', lw=1, alpha=0.3)
    current_hands = min(frame // 8 + 1, len(class_primes))
    if current_hands > 0:
        r_phases = (frame * 0.3 * np.log(class_primes[:current_hands])) % (2 * np.pi)
        for phi in r_phases:
            ax_res.plot([0, phi], [0, 1.0], 'b-', lw=1.8, alpha=0.6)
        R = np.sum(np.exp(1j * r_phases))
        r_angle = np.angle(R)
        r_mag = 0.8 + 2.5 * abs(R)
        ax_res.plot([0, r_angle], [0, r_mag], 'gold', lw=5)
        ax_res.scatter(r_angle, r_mag, s=180, color='gold')
    ax_res.set_title(f'Class {k} Resonance · {current_hands} hands')

    return []

ani = FuncAnimation(fig, animate, frames=600, interval=40, blit=False, repeat=True)

ani.save('class11_global_clock_face_corrected.mp4', writer='ffmpeg', fps=25, dpi=180)
print("✅ Video saved as 'class11_global_clock_face_corrected.mp4'")

plt.suptitle(f"Class {k} Global Clock Face — 24-Threaded Marking Clock + Resonance Golden Vector", fontsize=16)
plt.tight_layout()
plt.show()