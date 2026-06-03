#!/usr/bin/env python3
"""
Dual Clock Visualization — Index-Line Sieve Clock vs Imaginary-Axis Resonance Clock
Self-dual quantum clockwork of the algebraic ideal.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
MAX_N = 80000
N_OPERATORS = 24                     # number of clock hands / rulers
AMP_FACTOR = 1.5
T_MIN, T_MAX, DT = 13.5, 68.0, 0.018
INDEX_MAX = 500                      # how far to show on the index line
# ===================================================

def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

# Get correct deterministic holes (true primes only)
holes = []
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
for k in classes:
    amp = elder.generate_amplitude_map(k, MAX_N)
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 5 and is_prime(p):
                holes.append(p)
                if len(holes) >= N_OPERATORS:
                    break
    if len(holes) >= N_OPERATORS:
        break
holes = np.array(sorted(set(holes)))[:N_OPERATORS]
print("True prime operators (frequency operators):", holes)

# ====================== COMPUTE PATHS ======================
t_frames = np.arange(T_MIN, T_MAX, DT)
freqs = np.log(holes)
base_lengths = 1.0 / np.sqrt(holes)
phases = np.outer(t_frames, freqs) % (2 * np.pi)
R_complex = np.sum(base_lengths[None, :] * np.exp(1j * phases), axis=1)
V = np.cumsum(R_complex) * (t_frames[1] - t_frames[0])

# ====================== FIGURE ======================
fig = plt.figure(figsize=(20, 10), dpi=200)
gs = fig.add_gridspec(1, 2)

# LEFT: Index-line sieve clock
ax_index = fig.add_subplot(gs[0, 0], projection='polar')
ax_index.set_title('Index-Line Sieve Clock\n(Marking rods strike composites at 12 noon)', fontsize=14)
theta = np.linspace(0, 2*np.pi, 500)
ax_index.plot(theta, np.ones(500), 'k--', lw=1, alpha=0.3)
marking_lines = [ax_index.plot([], [], 'r-', lw=2.5)[0] for _ in holes]
marking_tips = [ax_index.scatter([], [], s=80, color='red') for _ in holes]
ax_index.set_rmax(1.3)
noon_line, = ax_index.plot([0, 0], [0, 1.3], 'gold', lw=4, solid_capstyle='round')

# RIGHT: Imaginary-axis resonance clock + golden vector trace
ax_res = fig.add_subplot(gs[0, 1], projection='polar')
ax_res.set_title('Imaginary-Axis Resonance Clock\n(Golden vector R(t) trace + murmuration)', fontsize=14)
ax_res.plot(theta, np.ones(500), 'k--', lw=1, alpha=0.3)
golden_line, = ax_res.plot([], [], 'gold', lw=6, solid_capstyle='round', zorder=5)
golden_tip = ax_res.scatter([], [], s=220, color='gold', zorder=6)
ax_res.set_rmax((1.0 / np.sqrt(holes)).max() * AMP_FACTOR * 1.2)

# Animation
def animate(frame):
    t = t_frames[frame % len(t_frames)]
    # LEFT: Index-line marking rods (phase advance with period p)
    for i, p in enumerate(holes):
        phase_index = (frame * 2 * np.pi / p) % (2 * np.pi)
        marking_lines[i].set_data([0, phase_index], [0, 1.0])
        marking_tips[i].set_offsets([[phase_index, 1.0]])
    
    # RIGHT: Resonance clock
    angles = (t * np.log(holes)) % (2 * np.pi)
    for i, theta in enumerate(angles):
        # (optional live hand update if you want; here we focus on golden vector)
        pass
    r_angle = np.angle(R_complex[frame % len(t_frames)])
    r_mag = min(ax_res.get_rmax()*0.95, 0.6 + 3.5 * abs(R_complex[frame % len(t_frames)])**1.4)
    golden_line.set_data([0, r_angle], [0, r_mag])
    golden_tip.set_offsets([[r_angle, r_mag]])
    
    return marking_lines + marking_tips + [golden_line, golden_tip]

ani = FuncAnimation(fig, animate, frames=len(t_frames)*2, interval=25, blit=False, repeat=True)

plt.suptitle("Self-Dual Quantum Clockwork of the Algebraic Ideal\n"
             "Left: Index-line marking (holes when no rod at noon) — Right: Resonance golden vector trace", 
             fontsize=16, y=0.98)

# ====================== VIDEO OUTPUT ======================
ani.save('dual_clock_self_dual_clockwork.mp4', writer='ffmpeg', fps=35, dpi=240)
print("✅ Video saved as 'dual_clock_self_dual_clockwork.mp4'")

plt.tight_layout()
plt.show()

print("Animation complete.")
print("• Left: Index-line sieve clock — marking rods strike at noon")
print("• Right: Imaginary-axis resonance clock — golden vector R(t) trace")
print("• The two clocks are self-dual: same quantized phase harmonics govern both")