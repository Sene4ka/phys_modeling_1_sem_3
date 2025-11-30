import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal


def solve_schrodinger_well(U, x_range, N=1000, mass=1.0, hbar=1.0):
    x = np.linspace(x_range[0], x_range[1], N)
    dx = x[1] - x[0]

    # Внутренние точки решетки (граничные условия ψ=0)
    x_inner = x[1:-1]
    U_values = U(x_inner)

    # Коэффициент перед лапласианом
    k = hbar ** 2 / (2 * mass * dx ** 2)

    diagonal = 2 * k + U_values
    off_diag = -k * np.ones(N - 3)

    energies, wavefunctions = eigh_tridiagonal(diagonal, off_diag)

    # Добавляем нули на границах
    wf_full = np.zeros((N, len(energies)))
    wf_full[1:-1, :] = wavefunctions

    # Нормировка
    for i in range(len(energies)):
        wf_full[:, i] /= np.sqrt(np.trapezoid(np.abs(wf_full[:, i]) ** 2, x))

    return energies, wf_full, x


BIG = 1e12


def make_rectangular_well(a):
    def U(x):
        # Используем поэлементное сравнение для массивов
        return np.where((x >= 0) & (x <= a), 0.0, BIG)

    return U


def make_triangular_well(a, slope=20.0):
    def U(x):
        # Используем поэлементное сравнение для массивов
        condition = (x >= 0) & (x <= a)
        return np.where(condition, slope * np.abs(x - a / 2), BIG)

    return U


def make_w_well(a, depth=10.0, width=0.2, sigma=0.01):
    def U(x):
        condition = (x >= 0) & (x <= a)
        center = a / 2
        left = center - width
        right = center + width

        # Вычисляем потенциал только для точек внутри ямы
        potential = np.zeros_like(x)
        mask = condition
        potential[mask] = -depth * (
                np.exp(-(x[mask] - left) ** 2 / sigma) +
                np.exp(-(x[mask] - right) ** 2 / sigma)
        )
        potential[~mask] = BIG

        return potential

    return U


def make_harmonic_well(a, k=10.0):
    def U(x):
        condition = (x >= 0) & (x <= a)
        center = a / 2
        return np.where(condition, 0.5 * k * (x - center) ** 2, BIG)

    return U


def plot_potential_and_wavefunctions(U, x_range, num_states=5, show_density=False):
    energies, psi, x = solve_schrodinger_well(U, x_range)
    num_states = min(num_states, len(energies))

    plt.figure(figsize=(11, 6))

    # нарисовать потенциал (заполненную область)
    Uvals = U(x)
    plt.fill_between(x, Uvals, np.min(Uvals) - 0.1 * abs(np.min(Uvals) + 1e-12),
                     color='lightgray', alpha=0.5, label='U(x)')

    # автоматический масштаб для волновых функций:
    if num_states > 1:
        dE = np.mean(np.diff(energies[:num_states]))
    else:
        dE = max(1.0, abs(energies[0]) * 0.2)

    psi_scale = max(0.2 * dE, 1e-6)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i in range(num_states):
        if show_density:
            profile = np.abs(psi[:, i]) ** 2
            profile = profile / np.max(profile) * psi_scale + energies[i]
            label = f"E={energies[i]:.3f} (ρ)"
            plt.plot(x, profile, color=colors[i % len(colors)], label=label)
        else:
            profile = psi[:, i]
            profile = profile / np.max(np.abs(profile)) * psi_scale + energies[i]
            label = f"E={energies[i]:.3f}"
            plt.plot(x, profile, color=colors[i % len(colors)], label=label)

        # горизонтальная линия уровня энергии
        plt.hlines(energies[i], x[0], x[-1], colors=colors[i % len(colors)],
                   linestyles='--', alpha=0.6)

    plt.plot(x, Uvals, 'k-', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('Энергия / U(x)')
    plt.title('Потенциал и собственные состояния')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.ylim(np.min(Uvals) - 0.5 * abs(np.min(Uvals) + 1e-12),
             energies[num_states - 1] + 0.6 * abs(energies[num_states - 1] + 1e-12))
    plt.show()


def transmission_coefficient(U, x_range, E, mass=1.0, hbar=1.0, N=2000):
    x = np.linspace(*x_range, N)
    dx = x[1] - x[0]

    M = np.eye(2, dtype=complex)
    eps = 1e-12

    for i in range(N - 1):
        U_avg = 0.5 * (U(x[i]) + U(x[i + 1]))
        dE = E - U_avg

        if dE > 0:
            k = np.sqrt(2 * mass * dE) / hbar
            k = max(k, eps)

            M_i = np.array([
                [np.cos(k * dx), 1j * np.sin(k * dx) / k],
                [1j * k * np.sin(k * dx), np.cos(k * dx)]
            ], dtype=complex)

        else:
            κ = np.sqrt(2 * mass * (-dE)) / hbar
            κ = max(κ, eps)

            M_i = np.array([
                [np.cosh(κ * dx), np.sinh(κ * dx) / κ],
                [κ * np.sinh(κ * dx), np.cosh(κ * dx)]
            ], dtype=complex)

        M = M_i @ M

        norm = np.max(np.abs(M))
        if norm > 1e50:
            M /= norm

    M11 = M[0, 0]
    M11_abs = max(abs(M11), eps)

    k_in = np.sqrt(max(E, 0))
    k_in = max(k_in, eps)

    T = 1.0 / (M11_abs ** 2)
    T = min(max(T, 0.0), 1.0)

    return T


def plot_transmission(U, x_range, E_range, mass=1.0):
    transmission = []
    for E in E_range:
        T = transmission_coefficient(U, x_range, E, mass)
        transmission.append(T)

    plt.figure(figsize=(10, 6))
    plt.plot(E_range, transmission, 'b-', linewidth=2)
    plt.xlabel('Энергия E')
    plt.ylabel('Коэффициент прохождения T')
    plt.title('Зависимость коэффициента прохождения от энергии')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.show()


def make_rectangular_barrier(a, height=5.0, width=0.3):
    center = a / 2

    def U(x):
        return np.where(np.abs(x - center) < width / 2, height, 0.0)

    return U


def make_triangular_barrier(a, height=5.0):
    center = a / 2
    width = a / 2

    def U(x):
        condition = np.abs(x - center) < width
        return np.where(condition, height * (1 - np.abs(x - center) / width), 0.0)

    return U


def make_gaussian_barrier(a, height=5.0, sigma=0.1):
    center = a / 2

    def U(x):
        return height * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))

    return U


if __name__ == "__main__":
    a = 5
    x_range = (0, a)

    plot_potential_and_wavefunctions(make_rectangular_well(a), x_range)
    plot_potential_and_wavefunctions(make_triangular_well(a), x_range)
    plot_potential_and_wavefunctions(make_w_well(a), x_range)
    plot_potential_and_wavefunctions(make_harmonic_well(a), x_range)

    E_range = np.linspace(0.1, 10, 200)

    plot_transmission(make_rectangular_barrier(a, height=5), x_range, E_range)
    plot_transmission(make_triangular_barrier(a), x_range, E_range)
    plot_transmission(make_gaussian_barrier(a, height=5), x_range, E_range)