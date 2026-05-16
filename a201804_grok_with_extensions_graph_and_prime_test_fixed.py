#!/usr/bin/env python
import math
import matplotlib.pyplot as plt
import numpy as np

def is_prime(n):
    """Check if n is prime and return divisors if non-prime."""
    if n < 2:
        return False, [i for i in range(1, n + 1) if n % i == 0]
    if n == 2:
        return True, []
    if n % 2 == 0:
        return False, [i for i in range(1, n + 1) if n % i == 0]
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False, [i for i in range(1, n + 1) if n % i == 0]
    return True, []

def drLD(x, l, m, z, listvar):
    """Composite generating function for A201804."""
    y = 90 * (x * x) - l * x + m
    if 0 <= y < len(listvar):
        listvar[y] += 1
    p = z + 90 * (x - 1)
    for n in range(1, int(((len(listvar) - 1 - y) / p) + 1)):
        next_y = y + p * n
        if 0 <= next_y < len(listvar):
            listvar[next_y] += 1

def compute_amplitudes_and_graph(h, silo_width=1000):
    """Compute amplitudes, plot last 100 addresses, and test last 100 zeros for primality."""
    # Compute epoch
    n_max = 90 * h * h - 12 * h + 1

    # Initialize amplitude map
    A224854 = [0] * (n_max + 1)

    # Base operators for k=11 (Table 8, Page 25)
    base_operators = [
        (120, 34, 7, 53),  # op 1
        (132, 48, 19, 29),  # op 2
        (120, 38, 17, 43),  # op 3
        (90, 11, 13, 77),   # op 4
        (78, -1, 11, 91),   # op 5
        (108, 32, 31, 41),  # op 6
        (90, 17, 23, 67),   # op 7
        (72, 14, 49, 59),   # op 8
        (60, 4, 37, 83),    # op 9
        (60, 8, 47, 73),    # op 10
        (48, 6, 61, 71),    # op 11
        (12, 0, 79, 89)     # op 12
    ]

    # Extend operators
    operators = []
    max_x = int((n_max / 90) ** 0.5) + 2
    max_prime = int(math.sqrt(90 * n_max)) + 1
    for l, m, p, q in base_operators:
        operators.append((l, m, p, q))
        i = 1
        while True:
            new_p = p + 90 * i
            new_q = q + 90 * i
            if new_p > max_prime or new_q > max_prime:
                break
            if is_prime(new_p)[0] and is_prime(new_q)[0]:
                operators.append((l, m, new_p, new_q))
            i += 1

    # Mark composites
    for l, m, p, q in operators:
        for x in range(1, max_x):
            drLD(x, l, m, p, A224854)
            drLD(x, l, m, q, A224854)

    # Compute holes and primes
    A224854a = [i for i, x in enumerate(A224854) if x == 0]
    primes = [90 * i + 11 for i in A224854a]

    # Test primality of last 100 zeros
    non_prime_zeros = []
    last_100_zeros = A224854a[-100:] if len(A224854a) >= 100 else A224854a
    for n in last_100_zeros:
        num = 90 * n + 11
        is_prime_result, divisors = is_prime(num)
        if not is_prime_result:
            non_prime_zeros.append((n, num, divisors))

    # Validation against original output
    validation_amplitudes = [
        (4, 31), (6, 31), (8, 31), (11, 63), (14, 31)
    ]
    validation_result = {
        "matches": True,
        "mismatches": []
    }
    for n, expected_amp in validation_amplitudes:
        if n < len(A224854) and A224854[n] != expected_amp:
            validation_result["matches"] = False
            validation_result["mismatches"].append((n, A224854[n], expected_amp))

    # Compute metrics
    total_energy = sum(A224854)
    zero_count = len(A224854a)
    zero_to_amplitude_ratio = zero_count / total_energy if total_energy > 0 else 0
    overlaps = sum(1 for x in A224854 if x > 1)
    frequency_primes = set(p for _, _, p, q in operators).union(q for _, _, p, q in operators)
    reciprocal_sum = sum(1.0 / p for p in frequency_primes if p != 0)

    # Compute silo variances
    silo_variances = []
    for start in range(0, n_max + 1, silo_width):
        end = min(start + silo_width - 1, n_max)
        silo = A224854[start:end + 1]
        silo_variances.append(np.var(silo) if silo else 0)

    # Plot last 100 amplitudes
    last_100_indices = list(range(n_max - 99, n_max + 1))
    last_100_amplitudes = A224854[n_max - 99:n_max + 1]
    plt.figure(figsize=(12, 6))
    plt.plot(last_100_indices, last_100_amplitudes, marker='o', linestyle='-', color='b')
    plt.title(f'Amplitudes for Last 100 Addresses (n={n_max-99} to {n_max})')
    plt.xlabel('Address (n)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig('amplitudes_last_100.png')
    plt.close()

    return {
        "n_max": n_max,
        "holes": zero_count,
        "primes": primes[:10],
        "total_energy": total_energy,
        "zero_to_amplitude_ratio": zero_to_amplitude_ratio,
        "reciprocal_sum": reciprocal_sum,
        "silo_variances": silo_variances[:10],
        "overlaps": overlaps,
        "validation": validation_result,
        "non_prime_zeros": non_prime_zeros
    }

# Run for h=30
h = 5
result = compute_amplitudes_and_graph(h)
print(f"Epoch n_max: {result['n_max']}")
print(f"Number of holes: {result['holes']}")
print(f"Primes (90n + 11): {result['primes']}...")
print(f"Total energy: {result['total_energy']}")
print(f"Zero-to-amplitude ratio: {result['zero_to_amplitude_ratio']:.6f}")
print(f"Sum of reciprocals of prime frequencies: {result['reciprocal_sum']:.6f}")
print(f"Silo variances (first 10): {result['silo_variances']}")
print(f"Number of overlaps: {result['overlaps']}")
print(f"Validation result: {result['validation']}")
print(f"Non-prime zeros in last 100 (n, 90n+11, divisors): {result['non_prime_zeros']}")
print("Graph of last 100 amplitudes saved as 'amplitudes_last_100.png'")
