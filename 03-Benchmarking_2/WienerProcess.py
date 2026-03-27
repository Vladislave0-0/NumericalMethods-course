import numpy as np
import matplotlib.pyplot as plt

T = 1.0            # Конечное время
delta = 0.0001     # Шаг дискретизации
M = 1000           # Количество траекторий
N = int(T / delta) # Общее количество шагов

# Создаем матрицу случайных чисел xi ~ N(0, 1)
xi = np.random.normal(0, 1, size=(N, M))

# Вычисляем приращения: dB = sqrt(delta) * xi
db = np.sqrt(delta) * xi

b_zero = np.zeros((1, M))
B = np.concatenate([b_zero, np.cumsum(db, axis=0)], axis=0)

t = np.linspace(0, T, N + 1)
final_values = B[-1, :]
mean_final = np.mean(final_values)
var_final = np.var(final_values, ddof=1)

print(f"Результаты симуляции {M} траекторий в момент T = {T}:")
print(f"  - Математическое ожидание: {mean_final:.4f}  (ожидалось 0)")
print(f"  - Выборочная дисперсия:    {var_final:.4f}  (ожидалось 1)")

plt.figure(figsize=(12, 6))
plt.plot(t, B[:, :1000], lw=0.5, alpha=0.7)

plt.title(f"Симуляция {M} траекторий Винеровского процесса ($delta$={delta})")
plt.xlabel("$t$")
plt.ylabel("$B_t$")
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig("wiener_process.png", dpi=300, bbox_inches='tight')
print("\nГрафик сохранен в './wiener_process.png'")
