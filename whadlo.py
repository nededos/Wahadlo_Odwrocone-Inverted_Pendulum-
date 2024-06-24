import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def pendulum_eq(t, y, A, l, omega, g=9.81):
    phi, phi_dot = y
    phi_ddot = -(g / l) * np.sin(phi) - (A ** 2 * omega ** 2 / (2 * l ** 2)) * np.sin(phi) * np.cos(phi)
    return [phi_dot, phi_ddot]


def effective_potential(phi, A, l, omega, m, g=9.81):
    return -m * g * l * np.cos(phi) + (m * A ** 2 * omega ** 2 / 4) * np.sin(phi) ** 2


def simulate_and_plot(A, l, f, m, initial_phi_degrees, initial_angular_velocity_degrees, simulation_time):
    omega = 2 * np.pi * f
    t_span = (0, simulation_time)
    t_eval = np.linspace(*t_span, 300)
    initial_phi_radians = np.radians(initial_phi_degrees)
    initial_angular_velocity_radians = np.radians(initial_angular_velocity_degrees)
    y0 = [initial_phi_radians, initial_angular_velocity_radians]

    sol = solve_ivp(pendulum_eq, t_span, y0, args=(A, l, omega), t_eval=t_eval, method='RK45')

    phi_degrees = np.degrees(sol.y[0])
    phi_dot_degrees_per_sec = np.degrees(sol.y[1])

    # Efektywny potencjał
    phi_range_degrees = np.linspace(-360, 360, 400)
    phi_range_radians = np.radians(phi_range_degrees)
    U = effective_potential(phi_range_radians, A, l, omega, m)

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.plot(sol.t, phi_degrees, label='Kąt (stopnie)')
    plt.plot(sol.t, phi_dot_degrees_per_sec, label='Prędkość kątowa (stopnie/s)')
    plt.title('Dynamika wahadła')
    plt.xlabel('Czas (s)')
    plt.ylabel('Wartości kątowe')
    plt.legend()
    plt.grid(True)

    plt.subplot(122)
    plt.plot(phi_range_degrees, U, label='Efektywny potencjał')
    plt.title('Efektywny potencjał wahadła')
    plt.xlabel('Kąt φ (stopnie)')
    plt.ylabel('Potencjał U (J)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Parametry wahadła
l = 0.245  # długość wahadła, m
A = 0.016  # amplituda pionowych drgań punktu zawieszenia, m
f = 10.5  # częstotliwość drgań punktu zawieszenia, Hz
m = 0.005  # masa wahadła, kg
initial_phi_degrees = 178  # początkowy kąt, stopnie (nieco odchylony od 180 dla obserwacji drgań)
initial_angular_velocity_degrees = 0  # początkowa prędkość kątowa, stopnie/s
simulation_time = 50  # czas symulacji, s

simulate_and_plot(A, l, f, m, initial_phi_degrees, initial_angular_velocity_degrees, simulation_time)


def calculate_critical_frequency(l, A, g=9.81):
    return 1 / (2 * np.pi) * np.sqrt(2 * g * l / A ** 2)


f_critical = calculate_critical_frequency(l, A)
print(f"Graniczna częstotliwość: {f_critical:.2f} Hz")