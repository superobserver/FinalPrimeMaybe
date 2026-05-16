#!/usr/bin/env python3
"""
Animated Time-Series of the Log(p) Stack
Shows the cosine waves sliding along t and building the resonance S(t)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder   # your module

# ====================== CONFIG ======================
n_primes = 10000                # first N holes (increase for more accuracy)
t_min, t_max, dt = 10.0, 20.0, 0.005   # sweep range and step
known_zeros = [14.134725]      # first zero (add more as needed)
# ===================================================

# Get deterministic holes from the algebraic ideal (all 24 classes)
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
primes = []
max_m = 20000                  # sufficient for ~1000+ primes
for k in classes:
    amp = elder.generate_amplitude_map(k, max_m)
    holes = [m for m in range(len(amp)) if amp[m] == 0]
    primes.extend(90 * m + k for m in holes)
primes = sorted(set(primes))[:n_primes]

print(f"Using first {len(primes)} deterministic holes from the algebraic ideal.")

# Pre-compute the waves on a fine t-grid
t_values = np.arange(t_min, t_max, dt)
waves = np.zeros((len(primes), len(t_values)))   # individual cosines
for i, p in enumerate(primes):
    waves[i] = np.cos(t_values * np.log(p)) / np.sqrt(p)

S = np.sum(waves, axis=0)   # total resonance

# ====================== ANIMATION ======================
fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
fig.suptitle('Sliding the Stack of log(p) Waves — Resonance S(t) from Algebraic Ideal Holes', fontsize=14)

# Individual waves (first 8, faint)
colors = plt.cm.viridis(np.linspace(0, 1, 8))
lines = []
for i in range(8):
    line, = ax.plot(t_values, waves[i], lw=0.8, alpha=0.3, color=colors[i],
                    label=f'p={primes[i]}' if i == 0 else "")
    lines.append(line)

# Total sum S(t) — thick line
sum_line, = ax.plot(t_values, S, 'b-', lw=2.5, label='S(t) = sum of all waves')

# Vertical sweep line
vline = ax.axvline(t_min, color='red', lw=2, linestyle='--', alpha=0.8)

# Known zero marker
for z in known_zeros:
    ax.axvline(z, color='red', linestyle=':', alpha=0.6, label='Known zero' if z == known_zeros[0] else "")

ax.set_xlabel('t (imaginary part along the number line)')
ax.set_ylabel('Wave amplitude')
ax.set_title('Each log(p) cosine slides at speed log(p) — coherent alignment creates bright troughs')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9)

def animate(frame):
    current_t = t_min + frame * dt
    vline.set_xdata([current_t])
    # Update sum line (cumulative up to current t for visual effect, but full S shown for clarity)
    # (full S is pre-computed; we just move the line)
    return [vline] + lines + [sum_line]

ani = FuncAnimation(fig, animate, frames=len(t_values), interval=30, blit=True, repeat=True)

# Save animation (uncomment to write MP4; requires ffmpeg)
# ani.save('logp_stack_animation.mp4', writer='ffmpeg', fps=60, dpi=200)

plt.tight_layout()
plt.show()