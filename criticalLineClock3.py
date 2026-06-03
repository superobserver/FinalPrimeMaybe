#!/usr/bin/env python3
"""
Eddy Visualization: Tight Negative Clustering at a True Zero vs Dispersed Flow
Demonstrates self-reinforcing accumulation of negative momentum.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Your exact data at the first true zero
primes = np.array([7,11,13,17,19,23,29,31,37,41,43,47])
phases_zero = np.radians(np.array([135.9,142.0,277.3,134.5,224.6,19.3,207.0,261.0,44.3,127.5,166.0,238.1]))
t_zero = 14.134725

# Same primes at a nearby random t (dispersed)
np.random.seed(42)
t_random = t_zero + 0.7
freqs = np.log(primes)
phases_random = (t_random * freqs) % (2 * np.pi)

def plot_clock(ax, phases, title, D_val):
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(theta, np.ones_like(theta), 'k--', lw=0.5, alpha=0.3)
    for phi in phases:
        color = 'crimson' if np.pi/2 < phi < 3*np.pi/2 else 'royalblue'
        ax.arrow(0, 0, np.cos(phi), np.sin(phi), head_width=0.05, head_length=0.07,
                 fc=color, ec=color, lw=1.6, alpha=0.85)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\nD = {D_val:.3f}')

def half_plane_mask(phases):
    return (np.pi/2 < phases) & (phases < 3*np.pi/2)

# Compute D for both
lengths = 1.0 / np.sqrt(primes)
neg_z = half_plane_mask(phases_zero)
R_neg_z = np.sum(lengths[neg_z] * np.exp(1j * phases_zero[neg_z]))
R_pos_z = np.sum(lengths[~neg_z] * np.exp(1j * phases_zero[~neg_z]))
D_z = np.abs(R_neg_z) - np.abs(R_pos_z)

neg_r = half_plane_mask(phases_random)
R_neg_r = np.sum(lengths[neg_r] * np.exp(1j * phases_random[neg_r]))
R_pos_r = np.sum(lengths[~neg_r] * np.exp(1j * phases_random[~neg_r]))
D_r = np.abs(R_neg_r) - np.abs(R_pos_r)

fig = plt.figure(figsize=(14, 6), dpi=180)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])

ax1 = fig.add_subplot(gs[0], projection='polar')
plot_clock(ax1, phases_zero, 'Eddy at True Zero (t = 14.13)', D_z)

ax2 = fig.add_subplot(gs[1], projection='polar')
plot_clock(ax2, phases_random, 'Dispersed Flow at Random t', D_r)

# Polar density of phases at the zero (showing the eddy)
ax3 = fig.add_subplot(gs[2], projection='polar')
theta = np.linspace(0, 2*np.pi, 400)
kde = gaussian_kde(phases_zero, bw_method=0.25)
density = kde(theta)
ax3.plot(theta, density, color='crimson', lw=2)
ax3.fill_between(theta, density, alpha=0.3, color='crimson')
ax3.set_title('Phase Density at Zero\n(sharp negative eddy)')
ax3.set_ylim(0, np.max(density)*1.1)

plt.suptitle('Self-Reinforcing Eddy at a True Zeta Zero\n'
             'Tight negative-momentum clustering (left) vs dispersed flow (right) — '
             'the geometric signature of anti-chaining regularity',
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('zeta_eddy_vs_dispersed.png', dpi=200)
print("✅ Saved: zeta_eddy_vs_dispersed.png — the persistent negative eddy at the true zero.")
plt.show()