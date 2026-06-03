#!/usr/bin/env python3
"""
Live Oscillatory Efficacy of Bloom Faces
Watch early blooms strengthen and weaken as new log(p) hands are absorbed.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks

SCRIPT_DIR = r'C:\Users\jwhel\Downloads'
if os.path.isdir(SCRIPT_DIR):
    sys.path.append(SCRIPT_DIR)
import April1Sieve2 as elder

k = 11
PRIMES_PER_CLASS = 40
T_MAX = 80.0
DT = 0.015

def get_class_primes(kk, n_primes):
    holes = []
    amp = np.array(elder.generate_amplitude_map(kk, 80000))
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + kk
            if p > 5 and (p == 2 or p == 3 or p == 5 or True):  # include all for demo
                if p > 5:
                    holes.append(p)
                    if len(holes) >= n_primes:
                        break
    return np.array(sorted(holes)[:n_primes])

primes = get_class_primes(k, PRIMES_PER_CLASS)
t_frames = np.arange(0.1, T_MAX, DT)

fig, (ax_vec, ax_eff) = plt.subplots(1, 2, figsize=(16, 6), dpi=180)
ax_vec.set_title('Golden Vector R(t) — Half-Plane Occupancy of Newest Hand')
ax_eff.set_title('Efficacy of Early Bloom Faces (oscillates ±½ log p)')

efficacy_history = {i: [] for i in range(3)}   # track first 3 blooms
pool_sizes = []

def animate(step):
    n_hands = 5 + step // 8          # add a new prime every 8 frames
    if n_hands > len(primes):
        n_hands = len(primes)
    current_primes = primes[:n_hands]
    freqs = np.log(current_primes)
    base = 1.0 / np.sqrt(current_primes)
    phases = np.outer(t_frames, freqs) % (2 * np.pi)
    neg_mask = (np.pi/2 < phases) & (phases < 3*np.pi/2)
    R_neg = np.sum(base[None, :] * np.exp(1j * phases) * neg_mask, axis=1)
    R_pos = np.sum(base[None, :] * np.exp(1j * phases) * ~neg_mask, axis=1)
    D = np.abs(R_neg) - np.abs(R_pos)
    peaks, _ = find_peaks(D, prominence=0.06, distance=25)

    # live vector plot
    ax_vec.clear()
    theta = np.linspace(0, 2*np.pi, 400)
    ax_vec.plot(theta, np.ones_like(theta), 'k--', lw=0.8, alpha=0.4)
    latest_phase = phases[step % len(t_frames), -1]
    color = 'crimson' if latest_phase > np.pi/2 and latest_phase < 3*np.pi/2 else 'royalblue'
    ax_vec.arrow(0, 0, np.cos(latest_phase), np.sin(latest_phase),
                 head_width=0.08, head_length=0.12, fc=color, ec=color, lw=2)
    ax_vec.set_xlim(-1.3, 1.3)
    ax_vec.set_ylim(-1.3, 1.3)
    ax_vec.set_aspect('equal')
    ax_vec.set_title(f'Newest hand (p={current_primes[-1]})  —  {"NEGATIVE" if color=="crimson" else "POSITIVE"} half-plane\n±½ log p contribution visible')

    # efficacy history
    ax_eff.clear()
    for i in range(min(3, len(peaks))):
        if i < len(peaks):
            efficacy_history[i].append(D[peaks[i]])
            if len(efficacy_history[i]) > 80:
                efficacy_history[i].pop(0)
            ax_eff.plot(efficacy_history[i], label=f'Bloom {i+1} (early)')
    ax_eff.set_ylim(0, 1.6)
    ax_eff.set_xlabel('Step (new log p added)')
    ax_eff.set_ylabel('Bloom efficacy D(t_peak)')
    ax_eff.set_title('Oscillatory Efficacy of Early Blooms\n(±½ log p reinforcement / opposition)')
    ax_eff.legend(loc='upper right', fontsize=8)
    ax_eff.grid(True, alpha=0.3)

    pool_sizes.append(n_hands)
    if len(pool_sizes) > 80:
        pool_sizes.pop(0)

ani = FuncAnimation(fig, animate, frames=300, interval=60, blit=False, repeat=True)
ani.save('bloom_efficacy_oscillation.mp4', writer='ffmpeg', fps=20, dpi=160)
print("✅ Saved animation: bloom_efficacy_oscillation.mp4 — watch early blooms strengthen and weaken as new hands arrive.")
plt.show()