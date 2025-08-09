import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)
y = np.sin(x)
noise = np.random.normal(0, 0.1, size=x.shape)
y_noisy = y + noise

plt.plot(x, y, label='Original sin(x)')
plt.scatter(x, y_noisy, color='red', label='Noisy data', s=10)
plt.title("Sine Function and Noisy Data")
plt.legend()
plt.show()
