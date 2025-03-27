import numpy as np
import matplotlib.pyplot as plt

# Function f(x)
def f(x):
    return 10*x**4 + 3*x**3 - 30*x**2 + 10*x

# Function g(x1, x2)
def g(x1, x2):
    return (x1 - 2)**4 + (x2 + 3)**4 + 2 * (x1 - 2)**2 * (x2 + 3)**2

# Plot f(x)
x = np.linspace(-3, 3, 400)
y = f(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$f(x) = 10x^4 + 3x^3 - 30x^2 + 10x$", color="b")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Funkcja f(x)")
plt.legend()
plt.grid()
plt.show()

# Plot g(x1, x2)
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = g(X1, X2)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.8)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("g(x1, x2)")
ax.set_title("Funkcja g(x1, x2)")
plt.show()
