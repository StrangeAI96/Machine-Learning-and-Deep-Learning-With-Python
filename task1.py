# Exercise 1 - Data Visualization and Monotonic Array Check

This notebook contains two tasks:

**Task 1:** Plot a sine function and generate noise-corrupted data from it to visualize the effect of noise on a basic signal.

**Task 2:** Check whether an array is monotonic (increasing or decreasing) without using built-in Python functions.

This exercise is designed to strengthen Python programming and data analysis skills.

**Author: Arash Ganjouri**
"""

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

def is_monotonic (array):
  increasing = True
  decreasing = True

  for i in range(len(array) - 1):
    if array[i] < array[i + 1]:
      decreasing = False
    elif array[i + 1] < array[i]:
      increasing = False

  if increasing or decreasing:
    print("Monotonic")
  else:
    print("Not Monotonic")

# Test:
array1 = [1,2,3,3,3,3,3,4,5,6,7]      # increasing
array2 = [7,6,5,4,3,3,3,3,3,2,1]      # decreasing
array3 = [1,5,4,3,3,3,3,3,2,6,7]      # not monotonic

is_monotonic(array1)
is_monotonic(array2)
is_monotonic(array3)
