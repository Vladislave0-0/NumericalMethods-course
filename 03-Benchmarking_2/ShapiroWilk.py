import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

T = 1.0
delta = 0.0001
M = 1000
N = int(T / delta)

xi = np.random.normal(0, 1, size=(N, M))
dB = np.sqrt(delta) * xi
B = np.vstack([np.zeros(M), np.cumsum(dB, axis=0)])
time_axis = np.linspace(0, T, N + 1)

gamma = 20
# Берем индексы через равные промежутки
check_indices = np.linspace(N // gamma, N, gamma, dtype=int)
check_times = time_axis[check_indices]

print(f"{'Время t':<10} | {'E[Bt]':<10} | {'D[Bt] (теор)':<12} | {'D[Bt] (выб)':<10} | {'p-value':<10}")
print("-" * 65)

p_values = []

for idx, t_val in zip(check_indices, check_times):
    sample = B[idx, :]
    
    mean_val = np.mean(sample)
    var_theoretical = t_val
    var_empirical = np.var(sample, ddof=1)
    
    # Тест Шапиро-Уилка
    _, p_val = stats.shapiro(sample)
    p_values.append(p_val)
    
    print(f"{t_val:<10.2f} | {mean_val:<10.4f} | {var_theoretical:<12.2f} | {var_empirical:<10.4f} | {p_val:<10.4f}")

plt.figure(figsize=(10, 5))
plt.plot(check_times, p_values, 'o--', color='blue', label='p-value (Shapiro-Wilk)')
plt.axhline(0.05, color='red', linestyle=':', label='alpha = 0.05')
plt.title(f"Результаты теста Шапиро-Уилка в {gamma} точках времени")
plt.xlabel("Время t")
plt.ylabel("p-value")
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("ShapiroWilk_results.png")
