#!/usr/bin/env python3
"""
Single-Class vs Global Resonance
Deinterlaces the global signal into 24 independent channels and visualizes one class vs global.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
max_m = 100000                 # increase for sharper higher-t troughs
t_min, t_max, dt = 0.0, 100.0, 0.02
test_class = 7                 # change to any coprime k
known_zeros = [14.134725, 21.022039, 25.010857, 30.424876, 32.935062,
               37.586178, 40.918719, 43.327073]
# ===================================================

classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]

# Global (all classes)
primes_global = []
for k in classes:
    amp = elder.generate_amplitude_map(k, max_m)
    holes = [m for m in range(len(amp)) if amp[m] == 0]
    primes_global.extend(90 * m + k for m in holes)
primes_global = sorted(set(primes_global))

# Single class only
amp_single = elder.generate_amplitude_map(test_class, max_m)
holes_single = [m for m in range(len(amp_single)) if amp_single[m] == 0]
primes_single = sorted(90 * m + test_class for m in holes_single)

print(f"Global holes: {len(primes_global)}")
print(f"Class {test_class} holes: {len(primes_single)}")

# t-grid
t_values = np.arange(t_min, t_max, dt)

# Compute both signals
S_global = np.zeros(len(t_values))
S_single = np.zeros(len(t_values))

for i, t in enumerate(t_values):
    S_global[i] = sum(np.cos(t * np.log(p)) / np.sqrt(p) for p in primes_global if p > 1)
    S_single[i] = sum(np.cos(t * np.log(p)) / np.sqrt(p) for p in primes_single if p > 1)

# Plot
fig, axs = plt.subplots(2, 1, figsize=(14, 9), dpi=300, sharex=True)

axs[0].plot(t_values, S_global, 'b-', lw=1.8, label='Global resonance (all 24 classes)')
for z in known_zeros:
    axs[0].axvline(z, color='red', linestyle='--', alpha=0.7, label='Known zeta zero' if z == known_zeros[0] else "")
axs[0].set_ylabel('Signal Strength')
axs[0].set_title('Global Resonance (All Classes)')
axs[0].grid(True, alpha=0.3)
axs[0].legend()

axs[1].plot(t_values, S_single, 'orange', lw=1.8, label=f'Class {test_class} resonance')
for z in known_zeros:
    axs[1].axvline(z, color='red', linestyle='--', alpha=0.7)
axs[1].set_xlabel('t (imaginary part)')
axs[1].set_ylabel('Signal Strength')
axs[1].set_title(f'Single-Class Resonance (Class {test_class})')
axs[1].grid(True, alpha=0.3)
axs[1].legend()

plt.suptitle('Deinterlacing the Global Resonance\n'
             'Single-class partial trace vs full zeta-zero signal')
plt.tight_layout()
plt.savefig(f'resonance_single_class_{test_class}_vs_global.png', dpi=300, bbox_inches='tight')
plt.show()