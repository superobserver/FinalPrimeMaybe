import numpy as np
import matplotlib.pyplot as plt

# First 12 known zeros (imaginary parts)
t_n = np.array([14.134725, 21.022039, 25.010857, 30.424876, 32.935062,
                37.586178, 40.918719, 43.327073, 48.005150, 49.773832,
                52.970321, 56.446247])

# Predicted epoch scaling
x_n = t_n * np.log(t_n) / (360 * np.pi)
N_primes = 90 * x_n**2

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
ax1.plot(t_n, x_n, 'b-', lw=2.5, label=r'Epochs $x_n \sim \frac{t_n \log t_n}{360\pi}$')
ax1.set_xlabel('Zero height $t_n$')
ax1.set_ylabel('Epochs required (log scale)', color='blue')
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(t_n, N_primes, 'r--', lw=2, label='Primes required $\sim 90 x_n^2$')
ax2.set_ylabel('Number of primes required', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Resistance to Sharpening: Epoch/Prime Cost vs. Zero Height\n'
          'The algebraic ideal predicts exponential growth in computational effort')
fig.legend(loc='upper right')
plt.tight_layout()
plt.savefig('resistance_to_sharpening.png', dpi=300)
plt.show()