import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4 * np.pi, 200)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("Sine Wave — Plot Test")
plt.grid(True)
plt.show()
