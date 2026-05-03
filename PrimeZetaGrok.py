import numpy as np
import matplotlib.pyplot as plt

def get_primes(n):
    """Simple sieve to get primes up to n."""
    primes = []
    sieve = [True] * (n + 1)
    for p in range(2, n + 1):
        if sieve[p]:
            primes.append(p)
            for i in range(p * p, n + 1, p):
                sieve[i] = False
    return primes

def scan_zeta_zeros(primes, t_start=0, t_end=35, step=0.1):
    """
    Scans through values of t to find resonance among log(p).
    We use the cosine sum: sum(cos(t * log(p)) / sqrt(p))
    """
    t_values = np.arange(t_start, t_end, step)
    signal_strength = []

    for t in t_values:
        # This is the "Collective Behavior" sum
        # We divide by sqrt(p) to match the zeta function's critical line behavior
        resonance = sum(np.cos(t * np.log(p)) / np.sqrt(p) for p in primes)
        signal_strength.append(resonance)
        
    return t_values, signal_strength

# 1. Get our "Survivors" (e.g., primes up to 1000)
my_primes = get_primes(1000)

# 2. Run the scan
t_axis, signal = scan_zeta_zeros(my_primes, t_start=10, t_end=35, step=0.05)

# 3. Visualize the "Resonance"
plt.figure(figsize=(12, 6))
plt.plot(t_axis, signal, label='Collective Prime Resonance', color='blue')
plt.axhline(0, color='black', lw=0.5)

# Marking the known actual Zeta Zeros
known_zeros = [14.13, 21.02, 25.01, 30.42, 32.93]
for z in known_zeros:
    plt.axvline(z, color='red', linestyle='--', alpha=0.6, label=f'Actual Zero (~{z})' if z == 14.13 else "")

plt.title("Tuning into the Zeta Zeros using Prime 'Gaps'")
plt.xlabel("Test Frequency (t)")
plt.ylabel("Signal Strength (Constructive/Destructive Interference)")
plt.legend()
plt.show()
