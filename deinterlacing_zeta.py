#!/usr/bin/env python3
"""
Deinterlacing: Variable Class Momentum at Each Zeta Zero
Shows how each of the 24 classes contributes differently ("momentum") at every zero.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
max_m = 200000
known_zeros = [14.134725, 21.022039, 25.010857, 30.424876, 32.935062, 37.586178]
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
# ===================================================

# Precompute holes → primes per class
primes_per_class = {}
for k in classes:
    amp = elder.generate_amplitude_map(k, max_m)
    holes = [m for m in range(len(amp)) if amp[m] == 0]
    primes = [90 * m + k for m in holes if 90 * m + k > 1]
    primes_per_class[k] = sorted(set(primes))

def S_k(t, primes_k):
    return np.sum(np.cos(t * np.log(primes_k)) / np.sqrt(primes_k))

# Compute momentum matrix: rows = zeros, columns = classes
momentum = np.zeros((len(known_zeros), len(classes)))
for i, t in enumerate(known_zeros):
    for j, k in enumerate(classes):
        momentum[i, j] = S_k(t, primes_per_class[k])

# Plot stacked momentum (absolute contribution)
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
bottom = np.zeros(len(known_zeros))
class_labels = [str(k) for k in classes]
colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))

for j in range(len(classes)):
    ax.bar(known_zeros, momentum[:, j], bottom=bottom, label=class_labels[j],
           color=colors[j], width=0.8, alpha=0.85)
    bottom += momentum[:, j]

ax.set_xlabel('t (zeta zero)')
ax.set_ylabel('Momentum S_k(t)')
ax.set_title('Variable Class Contributions (“Momentum”) at Each Zeta Zero\n'
             'Deinterlaced from the Algebraic Ideal (24 independent channels)')
ax.legend(ncol=6, fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('class_momentum_at_zeta_zeros.png', dpi=300, bbox_inches='tight')
plt.show()

# Print numerical table
print("Class Momentum Table (S_k(t_n))")
for i, t in enumerate(known_zeros):
    print(f"\nt ≈ {t:.6f} | Global S(t) ≈ {np.sum(momentum[i]):.6f}")
    sorted_idx = np.argsort(np.abs(momentum[i]))[::-1]
    for j in sorted_idx[:8]:  # top 8 only
        pct = momentum[i, j] / np.sum(momentum[i]) * 100
        print(f"  Class {classes[j]:2d}: {momentum[i,j]:8.4f} ({pct:5.1f}%)")