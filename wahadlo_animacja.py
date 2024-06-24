import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Parametry systemu
m = 0.1  # masa wahadła [kg]
L = 1.0  # długość pręta [m]
g = 9.81 # przyspieszenie grawitacyjne [m/s^2]
b = 0.1  # współczynnik tłumienia [N*m*s]

# Parametry sterowania PD
Kp = 10.0  # wzmocnienie proporcjonalne
Kd = 5.0   # wzmocnienie różniczkowe

# Równania ruchu z kontrolą PD
def pendulum_ode_pd(t, y, m, L, g, b, Kp, Kd):
    theta, omega = y
    u = -Kp * (theta - np.pi) - Kd * omega  # kontrola PD
    dydt = [omega, (m * g * L * np.sin(theta) - b * omega + u) / (m * L**2)]
    return dydt

# Warunki początkowe
theta0 = np.pi + 1  # kąt początkowy [rad]
omega0 = 0.0          # prędkość kątowa początkowa [rad/s]
y0 = [theta0, omega0]

# Czas symulacji
t_span = [0, 10]
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Rozwiązanie równań różniczkowych z kontrolą PD
sol = solve_ivp(pendulum_ode_pd, t_span, y0, t_eval=t_eval, args=(m, L, g, b, Kp, Kd))

# Animacja
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    theta = sol.y[0, frame]
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    line.set_data([0, x], [0, y])
    return line,

ani = FuncAnimation(fig, update, frames=len(sol.t), init_func=init, blit=True, interval=20)

plt.title('Animacja ruchu wahadła odwróconego z')
plt.show()
