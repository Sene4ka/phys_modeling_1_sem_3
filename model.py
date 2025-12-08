import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal


def solve_schrodinger_well(U, x_range, N=2000, hbar=1.0, mass=1.0):
    """
    Решает стационарное уравнение Шредингера для одномерной потенциальной ямы
    методом конечных разностей.

    Параметры:
    ----------
    U : callable
        Функция потенциала U(x), возвращающая значение потенциала в точке x
    x_range : tuple
        Диапазон интегрирования (x_min, x_max)
    N : int
        Количество точек дискретизации
    hbar : float
        Приведенная постоянная Планка (по умолчанию 1 в атомных единицах)
    mass : float
        Масса частицы (по умолчанию 1 в атомных единицах)

    Возвращает:
    -----------
    energies : ndarray
        Массив собственных значений энергии (отсортированных по возрастанию)
    wf_full : ndarray
        Матрица волновых функций, где wf_full[:, i] соответствует энергии energies[i]
    x : ndarray
        Массив координатных точек сетки

    Алгоритм:
    ---------
    1. Создается равномерная сетка и вычисляется шаг dx
    2. Уравнение Шредингера дискретизируется методом конечных разностей
    3. Формируется трехдиагональная матрица Гамильтона
    4. Решается задача на собственные значения для трехдиагональной матрицы
    5. Волновые функции нормируются на единицу
    """
    x = np.linspace(x_range[0], x_range[1], N)
    dx = x[1] - x[0]

    # Используем внутренние точки (граничные точки исключаются для условий Дирихле)
    x_inner = x[1:-1]
    n = len(x_inner)
    if n < 2:
        raise ValueError("Слишком мало внутренних точек для дискретизации")

    U_values = np.asarray(U(x_inner), dtype=float)
    if np.any(np.isinf(U_values)):
        raise ValueError("Бесконечные значения потенциала внутри области решения")

    # Коэффициент для кинетической энергии в дискретной форме
    kcoef = hbar ** 2 / (2.0 * mass * dx ** 2)  # = ħ²/(2mΔx²)

    # Формирование трехдиагональной матрицы Гамильтона:
    # Диагональные элементы: 2kcoef + U(x)
    # Внедиагональные элементы: -kcoef
    diagonal = 2.0 * kcoef + U_values
    off_diag = -kcoef * np.ones(n - 1)

    # Решение задачи на собственные значения для трехдиагональной матрицы
    energies, vecs = eigh_tridiagonal(diagonal, off_diag)

    # Расширение волновых функций на всю сетку (включая граничные точки)
    wf_full = np.zeros((N, len(energies)))
    wf_full[1:-1, :] = vecs

    # Нормировка волновых функций: ∫|ψ(x)|²dx = 1
    for i in range(len(energies)):
        prob = np.abs(wf_full[:, i]) ** 2
        norm = np.trapezoid(prob, x)  # Численное интегрирование по правилу трапеций
        if norm > 0:
            wf_full[:, i] /= np.sqrt(norm)

    return energies, wf_full, x


def plot_potential_and_wavefunctions(U, x_range, num_states=6, title=None):
    """
    Визуализирует потенциал и несколько первых собственных состояний.

    Параметры:
    ----------
    U : callable
        Функция потенциала
    x_range : tuple
        Диапазон координат для расчета
    num_states : int
        Количество отображаемых собственных состояний
    title : str, optional
        Заголовок графика

    Возвращает:
    -----------
    energies : ndarray
        Энергии отображенных состояний
    """
    # Решаем уравнение Шредингера
    energies, psi, x = solve_schrodinger_well(U, x_range)
    num_states = min(num_states, len(energies))

    # Вычисляем значения потенциала на сетке
    Uvals = np.asarray(U(x), float)

    plt.figure(figsize=(10, 7))

    # Отрисовка потенциала (заливка серым цветом)
    plt.fill_between(x, Uvals, np.min(Uvals) - 0.1,
                     color='lightgray', alpha=0.5, label='Потенциал')
    plt.plot(x, Uvals, 'k-', linewidth=2)

    # Определение масштаба для волновых функций на основе разности энергий
    if num_states > 1:
        dE = np.mean(np.diff(energies[:num_states]))
    else:
        dE = max(0.1, abs(energies[0]) * 0.2) if len(energies) > 0 else 0.1

    psi_scale = 0.2 * dE  # Масштабный коэффициент для амплитуд волновых функций

    # Цветовая палитра для различных состояний
    colors = plt.cm.viridis(np.linspace(0, 0.8, num_states))

    # Отрисовка волновых функций и горизонтальных линий уровней энергии
    for i in range(num_states):
        # Масштабирование и смещение волновой функции к уровню энергии
        profile = psi[:, i] / np.max(np.abs(psi[:, i])) * psi_scale + energies[i]
        plt.plot(x, profile, color=colors[i], linewidth=1.5,
                 label=f'E{i}={energies[i]:.4f} Ha({(energies[i] * 27.2114):.4f} eV)')
        plt.axhline(energies[i], color=colors[i], linestyle='--', alpha=0.5)

    plt.xlabel('x (a.u.)')
    plt.ylabel('Энергия (Ha)')
    plt.title(title or 'Потенциал и собственные состояния')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Вывод численных значений энергий в консоль
    print(f"Энергии первых {num_states} уровней:")
    for i in range(num_states):
        print(f"  E{i} = {energies[i]:.6f} Ha ({energies[i] * 27.2114:.4f} эВ)")

    return energies[:num_states]


def plot_infinite_well(a, num_states=6):
    """
    Специальная функция для визуализации бесконечной прямоугольной ямы.

    Параметры:
    ----------
    a : float
        Ширина ямы
    num_states : int
        Количество отображаемых собственных состояний

    Особенности:
    ------------
    - Добавляет вертикальные линии для обозначения бесконечных стенок
    - Ограничивает отображение больших значений потенциала
    - Настраивает визуализацию для лучшего восприятия
    """
    # Создаем потенциал бесконечной ямы
    U = rectangular_well(a, depth=None)

    # Область расчета совпадает с шириной ямы
    x0 = 0.0
    x1 = a

    # Решение уравнения Шредингера
    energies, psi, x = solve_schrodinger_well(U, (x0, x1))
    num_states = min(num_states, len(energies))

    plt.figure(figsize=(10, 7))

    # Вычисление и ограничение значений потенциала для отображения
    Uvals = np.asarray(U(x), float)
    max_display = max(np.max(energies[:num_states]) * 3, 30)
    Uvals_display = np.clip(Uvals, None, max_display)

    # Отрисовка потенциала
    plt.fill_between(x, Uvals_display, np.min(Uvals_display) - 0.1,
                     color='lightgray', alpha=0.5, label='Потенциал')
    plt.plot(x, Uvals_display, 'k-', linewidth=2)

    # Добавление вертикальных линий для стенок ямы
    center = a / 2
    left = center - a / 2
    right = center + a / 2
    plt.axvline(x=left, color='black', linewidth=3, alpha=0.7, linestyle='-', ymin=0)
    plt.axvline(x=right, color='black', linewidth=3, alpha=0.7, linestyle='-', ymin=0)

    # Определение масштаба для волновых функций
    if num_states > 1:
        dE = np.mean(np.diff(energies[:num_states]))
    else:
        dE = max(0.1, abs(energies[0]) * 0.2) if len(energies) > 0 else 0.1

    psi_scale = 0.15 * dE
    colors = plt.cm.viridis(np.linspace(0, 0.8, num_states))

    # Отрисовка волновых функций
    for i in range(num_states):
        psi_norm = psi[:, i] / np.max(np.abs(psi[:, i]))
        profile = psi_norm * psi_scale + energies[i]
        plt.plot(x, profile, color=colors[i], linewidth=1.5,
                 label=f'E{i}={energies[i]:.4f} Ha({(energies[i] * 27.2114):.4f} eV)')
        plt.axhline(energies[i], color=colors[i], linestyle='--', alpha=0.3)

    plt.xlabel('x (a.u.)')
    plt.ylabel('Энергия (Ha)')
    plt.title(f'Бесконечная прямоугольная яма (ширина={a:.1f} a.u.)')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Вывод численных значений энергий
    print(f"Энергии первых {num_states} уровней:")
    for i in range(num_states):
        print(f"  E{i} = {energies[i]:.6f} Ha ({energies[i] * 27.2114:.4f} эВ)")

    return energies[:num_states]


def rectangular_well(a, depth=None, width=None):
    """
    Создает функцию потенциала для прямоугольной ямы.

    Параметры:
    ----------
    a : float
        Ширина расчетной области
    depth : float, optional
        Глубина ямы. Если None, создается бесконечная яма
    width : float, optional
        Ширина ямы. Если None, равна ширине области

    Возвращает:
    -----------
    U : callable
        Функция потенциала, возвращающая значение в точке x

    Особенности:
    ------------
    - При depth=None создает бесконечную яму (U=∞ вне ямы, U=0 внутри)
    - При depth заданном создает конечную яму (U=depth вне ямы, U=0 внутри)
    """
    if width is None:
        width = a

    center = a / 2
    left = center - width / 2
    right = center + width / 2

    def U(x):
        x_arr = np.asarray(x)
        inside = (x_arr >= left) & (x_arr <= right)
        if depth is None:
            # Бесконечная яма: ∞ вне ямы, 0 внутри
            out = np.full_like(x_arr, np.inf, dtype=float)
            out[inside] = 0.0
            return out.item() if np.isscalar(x) else out
        else:
            # Конечная яма: depth вне ямы, 0 внутри
            out = np.full_like(x_arr, float(depth))
            out[inside] = 0.0
            return out.item() if np.isscalar(x) else out

    return U


def triangular_well(a, depth=20.0):
    """
    Создает функцию потенциала для треугольной ямы.

    Параметры:
    ----------
    a : float
        Ширина расчетной области
    depth : float
        Глубина ямы в центре

    Описание:
    ---------
    Создает V-образный потенциал с минимумом в центре области
    и линейным ростом к краям.
    """
    center = a / 2
    slope = (2 * depth) / a  # Наклон: от 0 в центре до depth на краях

    def U(x):
        x_arr = np.asarray(x)
        cond = (0 <= x_arr) & (x_arr <= a)
        out = np.full_like(x_arr, depth)  # По умолчанию значение depth
        out[cond] = slope * np.abs(x_arr[cond] - center)  # Линейный рост от центра
        return out.item() if np.isscalar(x) else out

    return U


def w_well(a, depth=20.0, positions=None, sigma=0.1):
    """
    Создает функцию потенциала для W-образной ямы.

    Параметры:
    ----------
    a : float
        Ширина расчетной области
    depth : float
        Максимальная глубина ямы
    positions : list, optional
        Положения минимумов потенциала
    sigma : float
        Ширина гауссовых ямок

    Описание:
    ---------
    Создает потенциал с двумя минимумами (W-форма) путем
    суперпозиции двух гауссовых ямок.
    """
    if positions is None:
        positions = [a / 3, 2 * a / 3]  # Равномерное распределение минимумов
    pos = np.array(positions)
    sigma = float(sigma)

    def U(x):
        x_arr = np.asarray(x)
        out = np.full_like(x_arr, depth)
        cond = (0 <= x_arr) & (x_arr <= a)
        if np.any(cond):
            xx = x_arr[cond]
            S = np.zeros_like(xx)
            # Сумма двух гауссовых функций для создания двух минимумов
            for p in pos:
                S += np.exp(-0.5 * ((xx - p) / sigma) ** 2)
            Smax = np.max(S) if np.max(S) > 0 else 1
            # Нормализация для создания W-формы
            out[cond] = (1 - S / Smax) * depth
        return out.item() if np.isscalar(x) else out

    return U


def harmonic_well(a, depth=20.0):
    """
    Создает функцию потенциала для параболической (гармонической) ямы.

    Параметры:
    ----------
    a : float
        Ширина расчетной области
    depth : float
        Глубина ямы на краях области

    Описание:
    ---------
    Создает квадратичный потенциал вида U(x) = k*(x-center)²,
    где k подбирается так, чтобы U(0) = U(a) = depth.
    """
    center = a / 2
    k = depth / (center ** 2)  # Коэффициент жесткости

    def U(x):
        x_arr = np.asarray(x)
        out = np.full_like(x_arr, depth)
        cond = (0 <= x_arr) & (x_arr <= a)
        out[cond] = k * (x_arr[cond] - center) ** 2
        return out.item() if np.isscalar(x) else out

    return U


def transmission_coefficient_matrix(U, E, x_range=None, X=None,
                                    mass=1.0, hbar=1.0, max_exp_arg=200.0):
    """
    Вычисление коэффициента прохождения через потенциальный барьер
    методом трансфер-матриц.

    Параметры:
    ----------
    U : callable
        Функция потенциала U(x)
    E : float
        Энергия частицы
    x_range : tuple, optional
        Диапазон (x0, x1) для равномерной сетки
    X : array_like, optional
        Массив точек сетки (N+1 точек, N сегментов).
        Если задан, то x_range игнорируется
    mass : float
        Масса частицы
    hbar : float
        Приведенная постоянная Планка
    max_exp_arg : float
        Максимальное значение аргумента экспоненты для предотвращения переполнения

    Возвращает:
    -----------
    T : float
        Коэффициент прохождения T(E) ∈ [0, 1]

    Алгоритм:
    ---------
    1. Разбиение области на сегменты с постоянным потенциалом
    2. Для каждого сегмента вычисление волнового числа k = √[2m(E-U)]/ħ
    3. Построение матриц распространения (P) и интерфейса (I)
    4. Перемножение матриц для получения полной трансфер-матрицы
    5. Вычисление коэффициента прохождения из элементов трансфер-матрицы

    Примечание:
    -----------
    Метод работает как для E > U (осцилляции), так и для E < U (экспоненциальное затухание)
    """
    if E <= 0:
        return 0.0

    # Определяем сетку (равномерную или пользовательскую)
    if X is not None:
        # Используем пользовательскую сетку
        x = np.asarray(X)
        if len(x) < 2:
            raise ValueError("Сетка X должна содержать как минимум 2 точки")
    else:
        # Используем равномерную сетку
        if x_range is None:
            raise ValueError("Необходимо задать либо x_range, либо X")
        x0, x1 = x_range
        N_segments = 4000  # по умолчанию
        x = np.linspace(x0, x1, N_segments + 1)

    # Вычисляем длины сегментов
    dx = x[1:] - x[:-1]  # массив длин N сегментов

    # Вычисляем потенциал в средних точках сегментов
    xm = 0.5 * (x[:-1] + x[1:])
    Uvals = np.asarray(U(xm), dtype=float)

    # Волновые числа в сегментах (комплексные для учета туннелирования)
    k = np.sqrt(2.0 * mass * (E - Uvals).astype(complex)) / hbar

    # Инициализация трансфер-матрицы (единичная матрица)
    M = np.eye(2, dtype=complex)

    def P(kj, dxj):
        """Матрица распространения внутри сегмента."""
        arg = 1j * kj * dxj
        im = np.imag(arg)
        # Ограничение аргумента для предотвращения переполнения
        if abs(im) > max_exp_arg:
            arg = np.real(arg) + 1j * np.sign(im) * max_exp_arg
        return np.array([
            [np.exp(arg), 0.0],
            [0.0, np.exp(-arg)]
        ], dtype=complex)

    def I(kj, knext):
        """Матрица интерфейса на границе сегментов."""
        eps = 1e-16
        kn = knext if abs(knext) > eps else (knext + eps)
        denom = 2.0 * kn
        # Матрица, обеспечивающая непрерывность ψ и dψ/dx на границе
        return np.array([
            [(kn + kj) / denom, (kn - kj) / denom],
            [(kn - kj) / denom, (kn + kj) / denom]
        ], dtype=complex)

    # Построение полной трансфер-матрицы
    for j in range(len(k)):
        Pj = P(k[j], dx[j])  # матрица распространения в j-ом сегменте

        if j < len(k) - 1:
            # Граница между сегментами j и j+1
            M = I(k[j], k[j + 1]) @ Pj @ M
        else:
            # Последний сегмент, нет границы после него
            M = Pj @ M

    # Элементы трансфер-матрицы
    M11, M12 = M[0, 0], M[0, 1]
    M21, M22 = M[1, 0], M[1, 1]

    # Коэффициент отражения r = B₀/A₀
    if abs(M22) < 1e-20:
        r = -M21 / (M22 + 1e-20)
    else:
        r = -M21 / M22

    # Амплитуда прошедшей волны t = A_N/A₀
    t = M11 + M12 * r

    # Волновые числа в первом и последнем сегментах
    k_left = k[0]
    k_right = k[-1]

    # Коэффициент прохождения: T = |t|² * (k_right/k_left)
    # (отношение потоков вероятности)
    T = np.abs(t) ** 2 * (np.real(k_right) / np.real(k_left))

    # Ограничение и проверка
    if not np.isfinite(T):
        return 0.0
    return float(np.clip(T, 0.0, 1.0))


def analytical_transmission_rectangular(E, V0, a_barrier):
    """
    Аналитическое выражение для коэффициента прохождения
    через прямоугольный потенциальный барьер.

    Параметры:
    ----------
    E : float
        Энергия частицы
    V0 : float
        Высота барьера
    a_barrier : float
        Ширина барьера

    Возвращает:
    -----------
    T : float
        Коэффициент прохождения T(E)

    Формулы:
    --------
    Для E < V0 (туннелирование):
        T = 1 / [1 + V0²/(4E(V0-E)) * sinh²(κa)]
        где κ = √[2m(V0-E)]/ħ

    Для E > V0 (надбарьерное прохождение):
        T = 1 / [1 + V0²/(4E(E-V0)) * sin²(Ka)]
        где K = √[2m(E-V0)]/ħ
    """
    if E <= 0:
        return 0.0

    k = np.sqrt(2 * max(E, 1e-10))

    if E < V0:
        # Туннелирование (экспоненциальная зависимость)
        kappa = np.sqrt(2 * max(V0 - E, 1e-10))
        # Избегаем переполнения при больших kappa*a
        arg = min(kappa * a_barrier, 100)
        denom = 1 + ((k ** 2 + kappa ** 2) ** 2 / (4 * k ** 2 * kappa ** 2)) * np.sinh(arg) ** 2
    else:
        # Надбарьерное прохождение (осцилляции)
        K = np.sqrt(2 * max(E - V0, 1e-10))
        denom = 1 + ((k ** 2 - K ** 2) ** 2 / (4 * k ** 2 * K ** 2)) * np.sin(K * a_barrier) ** 2

    return 1.0 / denom if denom > 0 else 0.0


def plot_transmission_coefficient(barrier_func, x_range=None, X=None, barrier_params=None,
                                  compare_analytical=False, N_points=1000,
                                  E_range=(0.1, 3.0), title=None):
    """
    Построение графика зависимости коэффициента прохождения от энергии.

    Параметры:
    ----------
    barrier_func : callable
        Функция потенциала U(x)
    x_range : tuple, optional
        Диапазон (x0, x1) для равномерной сетки
    X : array_like, optional
        Пользовательская сетка точек (N+1 точек)
    barrier_params : tuple, optional
        Параметры барьера (V0, a) для аналитического сравнения
    compare_analytical : bool
        Сравнивать ли с аналитическим решением (только для прямоугольного барьера)
    N_points : int
        Количество точек по энергии
    E_range : tuple
        Диапазон энергий (E_min, E_max)
    title : str, optional
        Заголовок графика
    """
    # Проверка входных данных
    if X is None and x_range is None:
        raise ValueError("Необходимо задать либо x_range, либо X")

    # Диапазон энергий
    E_min, E_max = E_range
    E_values = np.linspace(E_min, E_max, N_points)

    # Численное решение для каждой энергии
    T_numerical = []
    for E in E_values:
        try:
            if X is not None:
                # Используем пользовательскую сетку
                T = transmission_coefficient_matrix(barrier_func, E=E, X=X)
            else:
                # Используем равномерную сетку
                T = transmission_coefficient_matrix(barrier_func, E=E, x_range=x_range)
            T_numerical.append(T)
        except Exception as e:
            print(f"Ошибка при E={E}: {e}")
            T_numerical.append(0.0)

    plt.figure(figsize=(12, 8))

    # Численное решение
    plt.plot(E_values, T_numerical, 'b-', linewidth=2, label='Численное решение')

    # Аналитическое решение (только для прямоугольного барьера)
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


def plot_barrier_potentials(barriers, a=200.0, names=None, N=5000):
    """
    Визуализация различных форм потенциальных барьеров для сравнения.

    Параметры:
    ----------
    barriers : list
        Список функций потенциалов
    a : float
        Ширина расчетной области
    names : list, optional
        Список названий барьеров
    N : int
        Количество точек для построения графиков
    """
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


def rectangular_barrier(a, height=1.0, width=10.0):
    """
    Создает функцию прямоугольного потенциального барьера.

    Параметры:
    ----------
    a : float
        Ширина расчетной области
    height : float
        Высота барьера
    width : float
        Ширина барьера

    Возвращает:
    -----------
    U : callable
        Функция потенциала: height внутри барьера, 0 снаружи
    """
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
    """
    Создает функцию треугольного потенциального барьера.

    Параметры:
    ----------
    a : float
        Ширина расчетной области
    height : float
        Максимальная высота барьера (в центре)
    width : float
        Ширина основания барьера

    Описание:
    ---------
    Создает симметричный треугольный барьер с максимумом в центре
    и линейным спадом к краям.
    """
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
    """
    Создает функцию гауссова потенциального барьера.

    Параметры:
    ----------
    a : float
        Ширина расчетной области
    height : float
        Максимальная высота барьера (в центре)
    sigma : float
        Стандартное отклонение (ширина) гауссовой кривой

    Возвращает:
    -----------
    U : callable
        Гауссов потенциал: height * exp(-(x-center)²/(2σ²))
    """
    center = a / 2

    def U(x):
        x_arr = np.asarray(x)
        potential = height * np.exp(-(x_arr - center) ** 2 / (2 * sigma ** 2))
        return potential.item() if np.isscalar(x) else potential

    return U


def create_3segment_grid_for_rectangular_barrier(a_barrier, barrier_width):
    """
    Создает минимальную сетку (3 сегмента) для прямоугольного барьера.

    Параметры:
    ----------
    a_barrier : float
        Общая ширина области (x от 0 до a_barrier)
    barrier_width : float
        Ширина прямоугольного барьера
    padding : float
        Отступ слева и справа от барьера (не используется в текущей реализации)

    Возвращает:
    -----------
    X : np.array
        Сетка из 4 точек, определяющая 3 сегмента:
        1. Область слева от барьера
        2. Сам барьер
        3. Область справа от барьера

    Особенности:
    ------------
    Использование минимальной сетки значительно ускоряет вычисления
    для прямоугольного барьера без потери точности.
    """
    # Барьер находится в центре области
    center = a_barrier / 2
    left = center - barrier_width / 2
    right = center + barrier_width / 2

    # Проверяем, чтобы границы не выходили за пределы области
    left = max(left, 0)
    right = min(right, a_barrier)

    # Создаем сетку из 3 сегментов:
    # 1. Слева от барьера
    # 2. Сам барьер
    # 3. Справа от барьера
    X = np.array([0.0, left, right, a_barrier])
    return X


if __name__ == "__main__":
    # Параметры в атомных единицах
    a = 10.0  # 10 a.u. ≈ 0.529 нм

    dp = 5  # Глубина конечных потенциальных ям

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
    barrier_height = 1.0  # Ha
    barrier_width = 10.0  # a.u.
    center = a_barrier / 2

    # Создаем все барьеры
    barriers = [
        rectangular_barrier(a_barrier, height=barrier_height, width=barrier_width),
        triangular_barrier(a_barrier, height=barrier_height, width=barrier_width * 2),
        gaussian_barrier(a_barrier, height=barrier_height, sigma=barrier_width / 2)
    ]

    barrier_names = [
        f"Прямоугольный (h={barrier_height} Ha, w={barrier_width} a.u.)",
        f"Треугольный (h={barrier_height} Ha, w={barrier_width * 2} a.u.)",
        f"Гауссов (h={barrier_height} Ha, σ={barrier_width / 2:.1f} a.u.)"
    ]

    # 1. Отрисовка всех барьеров
    plot_barrier_potentials(barriers, a_barrier, barrier_names)

    # 2. Расчет коэффициентов прохождения для каждого барьера

    # 2.1 Прямоугольный барьер (с сравнением с аналитическим)
    # Создаем сетку из 3 сегментов для прямоугольного барьера
    X_rect = create_3segment_grid_for_rectangular_barrier(a_barrier, barrier_width)

    plot_transmission_coefficient(
        barriers[0],
        X=X_rect,  # Используем сетку из 3 сегментов
        barrier_params=(barrier_height, barrier_width),
        compare_analytical=True,
        title=f'Прямоугольный барьер (3 сегмента): коэффициент прохождения\n(h={barrier_height} Ha, w={barrier_width} a.u.)'
    )

    # 2.2 Треугольный барьер (используем равномерную сетку)
    plot_transmission_coefficient(
        barriers[1],
        x_range=(0, a_barrier),
        compare_analytical=False,
        title=f'Треугольный барьер: коэффициент прохождения\n(h={barrier_height} Ha, w={barrier_width * 2} a.u.)'
    )

    # 2.3 Гауссов барьер (используем равномерную сетку)
    plot_transmission_coefficient(
        barriers[2],
        x_range=(0, a_barrier),
        compare_analytical=False,
        title=f'Гауссов барьер: коэффициент прохождения\n(h={barrier_height} Ha, σ={barrier_width / 2:.1f} a.u.)'
    )