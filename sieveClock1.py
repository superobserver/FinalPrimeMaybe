#!/usr/bin/env python3
"""
Class 11 Global Clock Face — Index-Line Marking Clock + Resonance Golden Vector
24 marking channels evolving along n, synchronized with resonance clock.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

k = 11
MAX_N = 1200          # how far to show on index line
EPOCHS = 8            # how many epochs to evolve
AMP_FACTOR = 7.0

# Get pairs and ground-state triples for class k=11
pairs = elder.get_pairs_for_class(k)
print(f"Class {k} — 24 marking channels (z,o pairs): {pairs}")

# Extract y0, p, q for each channel (from the module's logic)
channels = []
for z, o in pairs:
    l = 180 - (z + o)
    m = 90 - (z + o) + (z * o // 90)
    for x in range(1, EPOCHS + 1):
        y0 = 90 * x * x - l * x + m
        p = z + 90 * (x - 1)
        q = o + 90 * (x - 1)
        channels.append((y0, p, q))
channels = channels[:24]  # exactly 24 channels per class

# Generate amplitude map for visualization
amp = elder.generate_amplitude_map(k, MAX_N)
holes = [i for i in range(1, MAX_N+1) if amp[i] == 0]

fig = plt.figure(figsize=(20, 11), dpi=200)
gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])

ax_clock = fig.add_subplot(gs[0, 0], projection='polar')
ax_amp = fig.add_subplot(gs[1, 0])
ax_res = fig.add_subplot(gs[0, 1], projection='polar')
ax_info = fig.add_subplot(gs[1, 1])

ax_clock.set_title(f'Class {k} Index-Line Marking Clock\n24 Channels · Live Noon-Strikes', fontsize=14)
ax_res.set_title(f'Class {k} Resonance Clock · Golden Vector R(t)', fontsize=14)

# Pre-compute resonance data for class 11 primes
class_primes = np.array([p for p in [90*m + k for m in range(MAX_N)] if amp[m] == 0 and p > 5])
class_freqs = np.log(class_primes[:24])
class_base = 1.0 / np.sqrt(class_primes[:24])

def animate(frame):
    # LEFT: Index-line clock with 24 marking channels
    ax_clock.clear()
    theta = np.linspace(0, 2*np.pi, 500)
    ax_clock.plot(theta, np.ones(500), 'k--', lw=1, alpha=0.3)
    strikes = 0
    for i, (y0, p, q) in enumerate(channels):
        phase = (frame * 2 * np.pi / p + y0 / p) % (2 * np.pi)
        ax_clock.plot([0, phase], [0, 1.0], 'r-', lw=1.8, alpha=0.7)
        if abs(phase) < 0.08 or abs(phase - 2*np.pi) < 0.08:
            ax_clock.scatter(phase, 1.2, s=80, color='gold')
            strikes += 1
    ax_clock.set_title(f'Class {k} · 24 Marking Channels · Frame {frame} · Strikes: {strikes}')
    ax_clock.set_rmax(1.8)

    # Amplitude map
    ax_amp.clear()
    current_amp = amp[:300] if frame < 300 else amp[:300]
    ax_amp.bar(np.arange(len(current_amp)), current_amp, width=1, color='orange', alpha=0.7)
    ax_amp.set_title(f'Class {k} Amplitude Map (Multiplicity) · n = 1..300')
    ax_amp.set_ylabel('Noon-strikes (A_{11}(n))')
    holes_in_view = np.sum(current_amp == 0)
    ax_amp.text(0.05, 0.9, f'Holes in view: {holes_in_view}', transform=ax_amp.transAxes, fontsize=12)

    # RIGHT: Resonance clock for class 11 primes
    ax_res.clear()
    ax_res.plot(theta, np.ones(500), 'k--', lw=1, alpha=0.3)
    current_hands = min(frame // 10 + 1, len(class_primes))
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

ani = FuncAnimation(fig, animate, frames=300, interval=40, blit=False, repeat=True)

ani.save('class11_global_clock_face.mp4', writer='ffmpeg', fps=25, dpi=180)
print("✅ Video saved as 'class11_global_clock_face.mp4'")

plt.suptitle(f"Class {k} Global Clock Face — 24-Threaded Marking Clock + Resonance Golden Vector", fontsize=16)
plt.tight_layout()
plt.show()