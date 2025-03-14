import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gradientFunctions import gradient_f
from gradientDescent import gradient_descent

# Define the function
def f(x):
    return 10*x**4 + 3*x**3 - 30*x**2 + 10*x

# Gradient descent parameters
start_point = [2]
learning_rate = 0.01
tolerance = 0.001
max_iter = 1000

# Running the algorithm
minimum, history = gradient_descent(gradient_f, start_point, learning_rate, tolerance, max_iter)
history_x = np.array(history)[:, 0]  
history_f = f(history_x)
print(history_x[-1], history_f[-1])

# Preparing the plot
x = np.linspace(-3, 3, 400)
y = f(x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, label=r"$f(x) = 10x^4 + 3x^3 - 30x^2 + 10x$", color="b")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Gradient Descent Path Animation")
ax.legend()
ax.grid()

# Creating point markers
point, = ax.plot([], [], 'ro-', markersize=3, label="Gradient Path")
final_min, = ax.plot([], [], 'rx', markersize=10, label="Minimum")

# Update function for the animation
def update(i):
    point.set_data(history_x[:i+1], history_f[:i+1])
    if i == len(history_x) - 1:
        final_min.set_data([history_x[i]], [history_f[i]])
    return point, final_min


# Creating the animation
ani = animation.FuncAnimation(fig, update, frames=len(history_x), interval=50, blit=True)

# Save as GIF
ani.save("Zadanie_1/gradient_descent.gif", writer="pillow", fps=20)


# Save the last frame as JPG
update(len(history_x) - 1)  # Ensure the last frame is displayed
plt.savefig("gradient_descent_final.jpg")

plt.show()
