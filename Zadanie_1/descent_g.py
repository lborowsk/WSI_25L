import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from gradientFunctions import gradient_g
from gradientDescent import gradient_descent

# Defining the function
def g(x1, x2):
    return (x1 - 2)**4 + (x2 + 3)**4 + 2 * (x1 - 2)**2 * (x2 + 3)**2

# Preparing the grid for the 3D plot
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = g(X1, X2)

# Gradient descent parameters
start_point = [4, 4]
learning_rate = 0.001
tolerance = 0.001
max_iter = 1000

# Running the algorithm
minimum, history = gradient_descent(gradient_g, start_point, learning_rate, tolerance, max_iter)

# Converting history to a NumPy array
history = np.array(history)

# Creating the figure and 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.8)

# Starting point
ax.scatter(start_point[0], start_point[1], g(start_point[0], start_point[1]), 
           color='black', marker='o', s=50, label="Starting Point")

# Final minimum point
min_scatter = ax.scatter([], [], [], color='red', marker='x', s=100, label="Minimum")

# Creating the gradient path line
path, = ax.plot([], [], [], color='r', marker='o', markersize=3, label="Gradient Path")

# Axis labels and title
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("g(x1, x2)")
ax.set_title("Gradient Descent Path Animation for g(x1, x2)")
ax.legend()

# Update function for animation
def update(i):
    path.set_data(history[:i+1, 0], history[:i+1, 1])  # Set x1, x2
    path.set_3d_properties(g(history[:i+1, 0], history[:i+1, 1]))  # Set g(x1, x2)
    
    if i == len(history) - 1:
        min_scatter._offsets3d = (history[i, 0:1], history[i, 1:2], g(history[i, 0:1], history[i, 1:2]))
    
    return path, min_scatter

# Creating the animation
ani = animation.FuncAnimation(fig, update, frames=len(history), interval=50, blit=False)

# Saving as GIF
ani.save("gradient_descent_3D.gif", writer="pillow", fps=20)

plt.show()
