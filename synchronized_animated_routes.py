#!/usr/bin/env python3
"""
Animated Synchronized Routes: Classical Zeta vs. Exact Laplace Route (Class k=11)
Fixed scaling — both routes now fully visible in their frames.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmath
import mpmath

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ================== CONFIG ==================
k = 11
MAX_EPOCHS = 40
MAX_T = 1200
STEP = 2
SIGMA = 0.01
INTERVAL_MS = 30
SAVE_ANIMATION = False
FILENAME = "zeta_vs_laplace_animated_k11_fixed.mp4"
# ===========================================

print(f"Loading exact grating data for class k={k}...")
grating_data = elder.get_grating_data(k, max_x=MAX_EPOCHS)
print(f"Loaded {len(grating_data)} exact combs.")

def exact_laplace(s):
    total = 0j
    for _, y, p, q in grating_data:
        for period in (p, q):
            denom = 1 - cmath.exp(-s * period)
            if abs(denom) > 1e-12:
                total += cmath.exp(-s * y) / denom
    return total

# Pre-compute both routes
print("Pre-computing synchronized routes...")
t_values = np.arange(0, MAX_T + 1, STEP)

zeta_real, zeta_imag = [], []
laplace_real, laplace_imag = [], []

for t in t_values:
    # Classical zeta route
    z = mpmath.zeta(0.5 + 1j * t)
    zeta_real.append(float(z.real))
    zeta_imag.append(float(z.imag))
    
    # Our exact Laplace route
    s = complex(SIGMA, t)
    val = exact_laplace(s)
    laplace_real.append(val.real)
    laplace_imag.append(val.imag)

# Compute sensible limits with padding
max_z = 1.2 * max(max(abs(np.array(zeta_real))), max(abs(np.array(zeta_imag))))
max_l = 1.2 * max(max(abs(np.array(laplace_real))), max(abs(np.array(laplace_imag))))

print(f"Zeta route limits: ±{max_z:.1f}")
print(f"Laplace route limits: ±{max_l:.1f}")

# ================== ANIMATION SETUP ==================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

# Left: Zeta route
ax1.set_xlim(-max_z, max_z)
ax1.set_ylim(-max_z, max_z)
ax1.set_xlabel(r'Real part of $\zeta(1/2 + it)$')
ax1.set_ylabel(r'Imaginary part of $\zeta(1/2 + it)$')
ax1.set_title('Classical Riemann Zeta Route')
ax1.grid(True, alpha=0.3)
line_zeta, = ax1.plot([], [], 'k-', lw=1.2)
scatter_zeta = ax1.scatter([], [], c=[], cmap='viridis', s=12)
ax1.plot(0, 0, 'ro', markersize=8, label='Origin')
ax1.legend()

# Right: Laplace route
ax2.set_xlim(-max_l, max_l)
ax2.set_ylim(-max_l, max_l)
ax2.set_xlabel(r'Real part of $\mathcal{L}\{P_{11}(t)\}(0.01 + it)$')
ax2.set_ylabel(r'Imaginary part of $\mathcal{L}\{P_{11}(t)\}(0.01 + it)$')
ax2.set_title('Exact Laplace Route from Dirac Combs (Class 11)')
ax2.grid(True, alpha=0.3)
line_laplace, = ax2.plot([], [], 'k-', lw=1.2)
scatter_laplace = ax2.scatter([], [], c=[], cmap='viridis', s=12)
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Imaginary axis (all poles)')
ax2.plot(0, 0, 'ro', markersize=8)
ax2.legend()

# Colorbar and title
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
norm = plt.Normalize(0, MAX_T)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
fig.colorbar(sm, cax=cbar_ax).set_label(r'time parameter $t$')

title = fig.suptitle('', fontsize=16)

def init():
    line_zeta.set_data([], [])
    line_laplace.set_data([], [])
    scatter_zeta.set_offsets(np.empty((0, 2)))
    scatter_laplace.set_offsets(np.empty((0, 2)))
    return line_zeta, line_laplace, scatter_zeta, scatter_laplace

def animate(i):
    t_slice = t_values[:i+1]
    # Zeta
    line_zeta.set_data(zeta_real[:i+1], zeta_imag[:i+1])
    scatter_zeta.set_offsets(np.column_stack((zeta_real[:i+1], zeta_imag[:i+1])))
    scatter_zeta.set_array(t_slice)
    # Laplace
    line_laplace.set_data(laplace_real[:i+1], laplace_imag[:i+1])
    scatter_laplace.set_offsets(np.column_stack((laplace_real[:i+1], laplace_imag[:i+1])))
    scatter_laplace.set_array(t_slice)
    
    title.set_text(f'Synchronized Routes — t = {t_values[i]:.0f} / {MAX_T}')
    return line_zeta, line_laplace, scatter_zeta, scatter_laplace

ani = animation.FuncAnimation(fig, animate, frames=len(t_values), init_func=init,
                              interval=INTERVAL_MS, blit=True, repeat=True)

print("Animation ready — displaying live...")
plt.tight_layout(rect=[0, 0, 0.9, 0.96])

if SAVE_ANIMATION:
    print(f"Saving high-quality animation as {FILENAME}...")
    ani.save(FILENAME, writer='ffmpeg', fps=30, dpi=200)
    print("Animation saved successfully!")

plt.show()