#!/usr/bin/env python3
"""
Ouroboros Turnover — Early Blooms Destroyed as the Operator Stack Grows
Watch the lowest-t bloom rise, wobble, and eventually collapse while new higher-t blooms emerge.
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
MAX_HANDS = 145
T_MAX = 120.0
DT = 0.01

def get_class_primes(kk, n):
    holes = []
    amp = np.array(elder.generate_amplitude_map(kk, 100000))
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + kk
            if p > 5:
                holes.append(p)
                if len(holes) >= n:
                    break
    return np.array(sorted(holes)[:n])

primes = get_class_primes(k, MAX_HANDS)
t_frames = np.arange(0.1, T_MAX, DT)

fig, (ax_vec, ax_ouro) = plt.subplots(1, 2, figsize=(16, 6), dpi=180)
ax_vec.set_title('Golden Vector — Increasing Wobble Destroys Early Lock')
ax_ouro.set_title('Ouroboros Turnover: Efficacy of Lowest-t Bloom Collapses as Stack Grows')

early_efficacy = []
new_bloom_count = []

def animate(frame):
    n = 5 + frame // 4
    if n > len(primes):
        n = len(primes)
    current = primes[:n]
    freqs = np.log(current)
    base = 1.0 / np.sqrt(current)
    phases = np.outer(t_frames, freqs) % (2 * np.pi)
    neg_mask = (np.pi/2 < phases) & (phases < 3*np.pi/2)
    R_neg = np.sum(base[None, :] * np.exp(1j * phases) * neg_mask, axis=1)
    R_pos = np.sum(base[None, :] * np.exp(1j * phases) * ~neg_mask, axis=1)
    D = np.abs(R_neg) - np.abs(R_pos)
    peaks, _ = find_peaks(D, prominence=0.04, distance=15)

    # Vector
    ax_vec.clear()
    theta = np.linspace(0, 2*np.pi, 400)
    ax_vec.plot(theta, np.ones_like(theta), 'k--', lw=0.5, alpha=0.3)
    latest = phases[frame % len(t_frames), -1]
    for phi in phases[frame % len(t_frames)]:
        c = 'crimson' if np.pi/2 < phi < 3*np.pi/2 else 'royalblue'
        ax_vec.arrow(0, 0, np.cos(phi), np.sin(phi), head_width=0.03, head_length=0.05,
                     fc=c, ec=c, lw=0.8, alpha=0.5)
    ax_vec.arrow(0, 0, np.cos(latest), np.sin(latest), head_width=0.06, head_length=0.08,
                 fc='black', ec='yellow', lw=2)
    ax_vec.set_xlim(-1.25, 1.25)
    ax_vec.set_ylim(-1.25, 1.25)
    ax_vec.set_aspect('equal')
    ax_vec.set_title(f'{n} hands — Newest operator injects wobble that erodes early phase-lock')

    # Ouroboros tracking
    ax_ouro.clear()
    if len(peaks) > 0:
        lowest = D[peaks[0]]
        early_efficacy.append(lowest)
        if len(early_efficacy) > 150:
            early_efficacy.pop(0)
    ax_ouro.plot(early_efficacy, color='crimson', lw=1.8, label='Efficacy of lowest-t bloom')
    ax_ouro.axhline(0.25, color='k', linestyle='--', alpha=0.6, label='Einselection threshold (illustrative)')
    ax_ouro.set_ylim(0, 1.6)
    ax_ouro.set_xlabel('Step (new asynchronous operator added)')
    ax_ouro.set_ylabel('D(t) of lowest-t bloom')
    ax_ouro.set_title('Ouroboros: Early bloom efficacy collapses; new higher-t blooms emerge')
    ax_ouro.legend(loc='upper right', fontsize=8)
    ax_ouro.grid(True, alpha=0.3)

    new_bloom_count.append(max(0, len(peaks) - 2))
    if len(new_bloom_count) > 150:
        new_bloom_count.pop(0)

ani = FuncAnimation(fig, animate, frames=320, interval=45, blit=False, repeat=True)
ani.save('ouroboros_turnover_early_blooms_destroyed.mp4', writer='ffmpeg', fps=18, dpi=160)
print("✅ Saved: ouroboros_turnover_early_blooms_destroyed.mp4 — early bloom collapses while new higher blooms appear.")
plt.show()