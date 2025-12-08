import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal


def solve_schrodinger_well(U, x_range, N=2000, hbar = 1.0, mass = 1.0):
    x = np.linspace(x_range[0], x_range[1], N)
    dx = x[1] - x[0]

    x_inner = x[1:-1]
    n = len(x_inner)
    if n < 2:
        raise ValueError()

    U_values = np.asarray(U(x_inner), dtype=float)
    if np.any(np.isinf(U_values)):
        raise ValueError()

    kcoef = hbar ** 2 / (2.0 * mass * dx ** 2)  # = 1/(2 dx²)

    diagonal = 2.0 * kcoef + U_values
    off_diag = -kcoef * np.ones(n - 1)

    energies, vecs = eigh_tridiagonal(diagonal, off_diag)

    wf_full = np.zeros((N, len(energies)))
    wf_full[1:-1, :] = vecs

    for i in range(len(energies)):
        prob = np.abs(wf_full[:, i]) ** 2
        norm = np.trapezoid(prob, x)
        if norm > 0:
            wf_full[:, i] /= np.sqrt(norm)

    return energies, wf_full, x


def rectangular_well(a, depth=None, width=None):
    if width is None:
        width = a

    center = a / 2
    left = center - width/2
    right = center + width/2

    def U(x):
        x_arr = np.asarray(x)
        inside = (x_arr >= left) & (x_arr <= right)
        if depth is None:
            out = np.full_like(x_arr, np.inf, dtype=float)
            out[inside] = 0.0
            return out.item() if np.isscalar(x) else out
        else:
            out = np.full_like(x_arr, float(depth))
            out[inside] = 0.0
            return out.item() if np.isscalar(x) else out

    return U


def triangular_well(a, depth=20.0):
    center = a / 2
    slope = (2 * depth) / a

    def U(x):
        x_arr = np.asarray(x)
        cond = (0 <= x_arr) & (x_arr <= a)
        out = np.full_like(x_arr, depth)
        out[cond] = slope * np.abs(x_arr[cond] - center)
        return out.item() if np.isscalar(x) else out

    return U


def w_well(a, depth=20.0, positions=None, sigma=0.1):
    if positions is None:
        positions = [a/3, 2*a/3]
    pos = np.array(positions)
    sigma = float(sigma)

    def U(x):
        x_arr = np.asarray(x)
        out = np.full_like(x_arr, depth)
        cond = (0 <= x_arr) & (x_arr <= a)
        if np.any(cond):
            xx = x_arr[cond]
            S = np.zeros_like(xx)
            for p in pos:
                S += np.exp(-0.5 * ((xx - p)/sigma)**2)
            Smax = np.max(S) if np.max(S) > 0 else 1
            out[cond] = (1 - S/Smax) * depth
        return out.item() if np.isscalar(x) else out

    return U

def harmonic_well(a, depth=20.0):
    center = a/2
    k = depth / (center**2)

    def U(x):
        x_arr = np.asarray(x)
        out = np.full_like(x_arr, depth)
        cond = (0 <= x_arr) & (x_arr <= a)
        out[cond] = k * (x_arr[cond] - center)**2
        return out.item() if np.isscalar(x) else out

    return U

def rectangular_barrier(a, height=1.0, width=10.0):
    center = a / 2
    left = center - width / 2
    right = center + width / 2

    def U(x):
        x_arr = np.asarray(x)
        potential = np.zeros_like(x_arr)
        potential[(x_arr >= left) & (x_arr <= right)] = height
        return potential.item() if np.isscalar(x) else potential

    return U


def triangular_barrier(a, height=1.0, width=20.0):
    center = a / 2
    half_width = width / 2
    left = center - half_width
    right = center + half_width

    def U(x):
        x_arr = np.asarray(x)
        potential = np.zeros_like(x_arr)
        # Треугольная функция: максимум в центре, 0 на краях
        mask = (x_arr >= left) & (x_arr <= right)
        x_masked = x_arr[mask]
        # Линейный рост от краев к центру
        slope = height / half_width
        potential[mask] = height - slope * np.abs(x_masked - center)
        return potential.item() if np.isscalar(x) else potential

    return U


def gaussian_barrier(a, height=1.0, sigma=5.0):
    center = a / 2

    def U(x):
        x_arr = np.asarray(x)
        potential = height * np.exp(-(x_arr - center) ** 2 / (2 * sigma ** 2))
        return potential.item() if np.isscalar(x) else potential

    return U


def analytical_transmission_rectangular(E, V0, a_barrier):
    if E <= 0:
        return 0.0

    k = np.sqrt(2 * max(E, 1e-10))

    if E < V0:
        kappa = np.sqrt(2 * max(V0 - E, 1e-10))
        # Избегаем переполнения при больших kappa*a
        arg = min(kappa * a_barrier, 100)
        denom = 1 + ((k ** 2 + kappa ** 2) ** 2 / (4 * k ** 2 * kappa ** 2)) * np.sinh(arg) ** 2
    else:
        K = np.sqrt(2 * max(E - V0, 1e-10))
        denom = 1 + ((k ** 2 - K ** 2) ** 2 / (4 * k ** 2 * K ** 2)) * np.sin(K * a_barrier) ** 2

    return 1.0 / denom if denom > 0 else 0.0


def transmission_coefficient_matrix(U, x_range, E, N=4000, mass=1.0, hbar=1.0, max_exp_arg=200.0):
    if E <= 0:
        return 0.0

    x0, x1 = x_range
    x = np.linspace(x0, x1, N + 1)
    dx = x[1] - x[0]

    xm = 0.5 * (x[:-1] + x[1:])
    Uvals = np.asarray(U(xm), dtype=float)

    k = np.sqrt(2.0 * mass * (E - Uvals).astype(complex)) / hbar

    k_left = np.sqrt(2.0 * mass * E) / hbar
    k_right = k_left

    M = np.eye(2, dtype=complex)

    def P(kj, dx):
        arg = 1j * kj * dx

        im = np.imag(arg)
        if abs(im) > max_exp_arg:
            arg = np.real(arg) + 1j * np.sign(im) * max_exp_arg
        return np.array([
            [np.exp(arg),           0.0],
            [0.0,            np.exp(-arg)]
        ], dtype=complex)

    def I(kj, knext):
        eps = 1e-16
        kn = knext if abs(knext) > eps else (knext + eps)
        denom = 2.0 * kn

        return np.array([
            [(kn + kj) / denom, (kn - kj) / denom],
            [(kn - kj) / denom, (kn + kj) / denom]
        ], dtype=complex)

    for j in range(len(k)):
        Pj = P(k[j], dx)
        if j < len(k) - 1:
            M = I(k[j], k[j + 1]) @ Pj @ M
        else:
            M = Pj @ M

    M11, M12 = M[0, 0], M[0, 1]
    M21, M22 = M[1, 0], M[1, 1]

    if abs(M22) < 1e-20:
        r = -M21 / (M22 + 1e-20)
    else:
        r = -M21 / M22

    A_right = M11 + M12 * r

    T = np.abs(A_right)**2 * (np.real(k_right) / np.real(k_left))

    if not np.isfinite(T):
        return 0.0
    return float(np.clip(T, 0.0, 1.0))


def plot_potential_and_wavefunctions(U, x_range, num_states=6, title=None):
    energies, psi, x = solve_schrodinger_well(U, x_range)
    num_states = min(num_states, len(energies))

    Uvals = np.asarray(U(x), float)

    plt.figure(figsize=(10, 7))

    # Потенциал
    plt.fill_between(x, Uvals, np.min(Uvals) - 0.1, color='lightgray', alpha=0.5, label='Потенциал')
    plt.plot(x, Uvals, 'k-', linewidth=2)

    # Масштаб для волновых функций
    if num_states > 1:
        dE = np.mean(np.diff(energies[:num_states]))
    else:
        dE = max(0.1, abs(energies[0]) * 0.2) if len(energies) > 0 else 0.1

    psi_scale = 0.2 * dE

    colors = plt.cm.viridis(np.linspace(0, 0.8, num_states))

    for i in range(num_states):
        # Волновая функция
        profile = psi[:, i] / np.max(np.abs(psi[:, i])) * psi_scale + energies[i]
        plt.plot(x, profile, color=colors[i], linewidth=1.5, label=f'E{i}={energies[i]:.4f} Ha({(energies[i]*27.2114):.4f} eV)')
        plt.axhline(energies[i], color=colors[i], linestyle='--', alpha=0.5)

    plt.xlabel('x (a.u.)')
    plt.ylabel('Энергия (Ha)')
    plt.title(title or 'Потенциал и собственные состояния')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Энергии первых {num_states} уровней:")
    for i in range(num_states):
        print(f"  E{i} = {energies[i]:.6f} Ha ({energies[i] * 27.2114:.4f} эВ)")

    return energies[:num_states]


def plot_infinite_well(a, num_states=6):
    # Создаем потенциал
    U = rectangular_well(a, depth=None)

    # Область расчета должна быть немного шире ямы
    x0 = 0.0
    x1 = a

    energies, psi, x = solve_schrodinger_well(U, (x0, x1))
    num_states = min(num_states, len(energies))

    plt.figure(figsize=(10, 7))

    # Рисуем потенциал
    Uvals = np.asarray(U(x), float)

    # Для отображения обрезаем очень большие значения
    max_display = max(np.max(energies[:num_states]) * 3, 30)
    Uvals_display = np.clip(Uvals, None, max_display)

    plt.fill_between(x, Uvals_display, np.min(Uvals_display) - 0.1,
                     color='lightgray', alpha=0.5, label='Потенциал')
    plt.plot(x, Uvals_display, 'k-', linewidth=2)

    # Добавляем вертикальные линии для стенок
    center = a / 2
    left = center - a / 2
    right = center + a / 2
    plt.axvline(x=left, color='black', linewidth=3, alpha=0.7, linestyle='-', ymin=0)
    plt.axvline(x=right, color='black', linewidth=3, alpha=0.7, linestyle='-', ymin=0)

    # Волновые функции
    if num_states > 1:
        dE = np.mean(np.diff(energies[:num_states]))
    else:
        dE = max(0.1, abs(energies[0]) * 0.2) if len(energies) > 0 else 0.1

    psi_scale = 0.15 * dE
    colors = plt.cm.viridis(np.linspace(0, 0.8, num_states))

    for i in range(num_states):
        # Нормируем и масштабируем волновую функцию
        psi_norm = psi[:, i] / np.max(np.abs(psi[:, i]))
        profile = psi_norm * psi_scale + energies[i]
        plt.plot(x, profile, color=colors[i], linewidth=1.5,
                 label=f'E{i}={energies[i]:.4f} Ha({(energies[i]*27.2114):.4f} eV)')
        plt.axhline(energies[i], color=colors[i], linestyle='--', alpha=0.3)

    plt.xlabel('x (a.u.)')
    plt.ylabel('Энергия (Ha)')
    plt.title(f'Бесконечная прямоугольная яма (ширина={a:.1f} a.u.)')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Энергии первых {num_states} уровней:")
    for i in range(num_states):
        print(f"  E{i} = {energies[i]:.6f} Ha ({energies[i] * 27.2114:.4f} эВ)")

    return energies[:num_states]


def plot_barrier_potentials(barriers, a=200.0, names=None, N=5000):
    if names is None:
        names = [f"Барьер {i + 1}" for i in range(len(barriers))]

    x = np.linspace(0, a, N)

    plt.figure(figsize=(12, 8))

    colors = ['blue', 'green', 'red']

    for i, (barrier_func, name, color) in enumerate(zip(barriers, names, colors)):
        plt.plot(x, barrier_func(x), color=color, linewidth=2, label=name)

    plt.xlabel('x (a.u.)')
    plt.ylabel('Потенциал U(x) (Ha)')
    plt.title('Сравнение форм потенциальных барьеров')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_transmission_coefficient(barrier_func, x_range, barrier_params=None,
                                  compare_analytical=False, N_points=1000,
                                  E_range=(0.1, 3.0), title=None):
    # Диапазон энергий
    E_min, E_max = E_range
    E_values = np.linspace(E_min, E_max, N_points)

    # Численное решение
    T_numerical = []
    for E in E_values:
        try:
            T = transmission_coefficient_matrix(barrier_func, x_range, E, N=5000)
            T_numerical.append(T)
        except Exception as e:
            print(f"Ошибка при E={E}: {e}")
            T_numerical.append(0.0)

    plt.figure(figsize=(12, 8))

    # Численное решение
    plt.plot(E_values, T_numerical, 'b-', linewidth=2, label='Численное решение')

    # Аналитическое решение
    if compare_analytical and barrier_params is not None:
        V0, a_barrier = barrier_params
        T_analytical = [analytical_transmission_rectangular(E, V0, a_barrier)
                        for E in E_values]
        plt.plot(E_values, T_analytical, 'r--', linewidth=2, label='Аналитическое решение')
        plt.axvline(x=V0, color='black', linestyle='--', alpha=0.7,
                    label=f'Высота барьера: {V0} Ha')
    elif compare_analytical:
        print("Для сравнения с аналитическим решением нужны параметры barrier_params=(V0, a_barrier)")

    plt.xlabel('Энергия E (Ha)')
    plt.ylabel('Коэффициент прохождения T')

    if title:
        plt.title(title)
    else:
        if compare_analytical:
            plt.title('Коэффициент прохождения: численное и аналитическое решения')
        else:
            plt.title('Коэффициент прохождения')

    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return E_values, T_numerical


if __name__ == "__main__":
    # Параметры в атомных единицах
    a = 10.0  # 10 a0 ≈ 0.529 нм

    dp = 5

    # 1. Бесконечная прямоугольная яма
    plot_infinite_well(a)

    # 2. Конечная прямоугольная яма
    plot_potential_and_wavefunctions(
        rectangular_well(a, depth=dp),
        (-a / 4, a * 1.25),
        title="Конечная прямоугольная яма"
    )

    # 3. Треугольная яма
    plot_potential_and_wavefunctions(
        triangular_well(a, depth=dp),
        (-a / 4, a * 1.25),
        title="Треугольная яма"
    )

    # 4. W-образная яма
    plot_potential_and_wavefunctions(
        w_well(a, depth=dp, sigma=1),
        (-a / 4, a * 1.25),
        title="W-образная яма"
    )

    # 5. Параболическая яма
    plot_potential_and_wavefunctions(
        harmonic_well(a, depth=dp),
        (-a / 4, a * 1.25),
        title="Параболическая яма"
    )

    # Для барьеров используем большую область
    a_barrier = 200.0
    x_range = (0.0, a_barrier)

    # Параметры барьеров
    barrier_height = 1.0  # Ha
    barrier_width = 10.0  # a0

    # Создаем все барьеры
    barriers = [
        rectangular_barrier(a_barrier, height=barrier_height, width=barrier_width),
        triangular_barrier(a_barrier, height=barrier_height, width=barrier_width * 2),
        gaussian_barrier(a_barrier, height=barrier_height, sigma=barrier_width / 2)
    ]

    barrier_names = [
        f"Прямоугольный (h={barrier_height} Ha, w={barrier_width} a0)",
        f"Треугольный (h={barrier_height} Ha, w={barrier_width * 2} a0)",
        f"Гауссов (h={barrier_height} Ha, σ={barrier_width / 2:.1f} a0)"
    ]

    # 1. Отрисовка всех барьеров
    plot_barrier_potentials(barriers, a_barrier, barrier_names)

    # 2. Расчет коэффициентов прохождения для каждого барьера

    # 2.1 Прямоугольный барьер (с сравнением с аналитическим)
    plot_transmission_coefficient(
        barriers[0],
        x_range,
        barrier_params=(barrier_height, barrier_width),
        compare_analytical=True,
        title=f'Прямоугольный барьер: коэффициент прохождения\n(h={barrier_height} Ha, w={barrier_width} a0)'
    )

    # 2.2 Треугольный барьер
    plot_transmission_coefficient(
        barriers[1],
        x_range,
        compare_analytical=False,
        title=f'Треугольный барьер: коэффициент прохождения\n(h={barrier_height} Ha, w={barrier_width * 2} a0)'
    )

    # 2.3 Гауссов барьер
    plot_transmission_coefficient(
        barriers[2],
        x_range,
        compare_analytical=False,
        title=f'Гауссов барьер: коэффициент прохождения\n(h={barrier_height} Ha, σ={barrier_width / 2:.1f} a0)'
    )