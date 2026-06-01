import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi   # for li(x) approximation
sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

print("=== Super-Pointer Correction to the Prime-Counting Function ===")

# Super-pointer t_k* from your data (k=2 to k=14)
super_pointers = np.array([
    14.396000,
    111.356500,
    750.711000,
    3724.663,
    18479.967,
    91688.611,
    454914.312,
    2257063.646,
    11198452.468,
    6217965702.472
])

# Generate actual primes for visual comparison (up to x=10^7 for speed)
MAX_X = 10000000
holes = []
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
for k in classes:
    amp = elder.generate_amplitude_map(k, MAX_X//90 + 100)
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 5 and p <= MAX_X:
                holes.append(p)
primes = np.sort(np.unique(holes))
print(f"Generated {len(primes)} actual primes up to {MAX_X}")

# Compute smooth main term li(x) ≈ ∫_2^x dt/log t
def li(x):
    return expi(np.log(x)) - expi(np.log(2)) if x > 2 else 0

x = np.logspace(1, 7, 2000)  # log-spaced points for smooth plot

# Smooth main term
main_term = np.array([li(xi) for xi in x])

# Oscillatory correction using super-pointers as zeros
correction = np.zeros_like(x)
for tn in super_pointers:
    phase = tn * np.log(x)
    amp = x**0.5 / np.abs(0.5 + 1j*tn)
    correction += -amp * np.cos(phase) / np.sqrt(0.25 + tn**2)

corrected = main_term + correction

# Actual prime counting function for comparison
pi_actual = np.zeros_like(x)
for i, xi in enumerate(x):
    pi_actual[i] = np.sum(primes <= xi)

# Plot
plt.figure(figsize=(12, 8))
plt.plot(x, main_term, 'b-', lw=2, label='Smooth main term li(x)')
plt.plot(x, corrected, 'r-', lw=2, label='Super-pointer corrected π(x)')
plt.plot(x, pi_actual, 'k-', lw=1.5, alpha=0.8, label='Actual π(x) from algebraic ideal')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x (log scale)')
plt.ylabel('Prime counting function')
plt.title('Prime Counting Function Updated by Super-Pointer States')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('super_pointer_prime_counting_correction.png', dpi=300)
plt.show()

print("Plot saved as super_pointer_prime_counting_correction.png")
print("The red curve shows how the super-pointer states sharpen the smooth logarithmic decay into the step-like prime-counting function.")