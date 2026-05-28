#!/usr/bin/env python3
"""
Deinterlacing: Sharp Class-Specific Resonances vs Global Signal
Visualizes how each class channel reveals its own clean troughs.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
max_m = 15000                 # sufficient for clean troughs up to t≈100
t_min, t_max, dt = 0.0, 80.0, 0.01
known_zeros = [14.134725, 21.022039, 25.010857, 30.424876, 32.935062,
               37.586178, 40.918719, 43.327073, 48.005151, 49.773832]
test_classes = [7, 11, 13, 17, 19, 37]   # change or expand as desired
# ===================================================

classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]

# Precompute primes per class (and global)
primes_per_class = {}
primes_global = []
for k in classes:
    amp = elder.generate_amplitude_map(k, max_m)
    holes = [m for m in range(len(amp)) if amp[m] == 0]
    primes = sorted([90 * m + k for m in holes if 90 * m + k > 1])
    primes_per_class[k] = primes
    primes_global.extend(primes)
primes_global = sorted(set(primes_global))

print(f"Global holes: {len(primes_global)}")

def compute_S(t_values, primes_list):
    S = np.zeros(len(t_values))
    for i, t in enumerate(t_values):
        S[i] = sum(np.cos(t * np.log(p)) / np.sqrt(p) for p in primes_list if p > 1)
    return S

t_values = np.arange(t_min, t_max, dt)
S_global = compute_S(t_values, primes_global)

# Plot
fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
ax.plot(t_values, S_global, 'k-', lw=2.5, label='Global S(t) (all 24 classes)', alpha=0.9)

colors = plt.cm.tab10(np.linspace(0, 1, len(test_classes)))
for i, k in enumerate(test_classes):
    S_k = compute_S(t_values, primes_per_class[k])
    ax.plot(t_values, S_k, color=colors[i], lw=1.2, label=f'Class {k} (S_k(t))')

for z in known_zeros:
    ax.axvline(z, color='red', linestyle='--', alpha=0.6, label='Known global zero' if z == known_zeros[0] else "")

ax.set_xlabel('t (imaginary part)')
ax.set_ylabel('Resonance Signal Strength')
ax.set_title('Deinterlaced Class Channels vs Global Resonance\n'
             'Sharp, noise-free troughs appear only in the classes that dominate each zero')
ax.grid(True, alpha=0.3)
ax.legend(ncol=3, fontsize=9)
plt.tight_layout()
plt.savefig('deinterlaced_class_channels_vs_global.png', dpi=300, bbox_inches='tight')
plt.show()