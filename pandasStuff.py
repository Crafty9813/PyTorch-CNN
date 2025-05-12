import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



x = np.linspace(0, 10, 100) #get a set of x values for plotting points, more is better
y = np.sin(x)
plt.plot(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("y = sinx")
plt.show()