import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('benchmark_errors.csv')

plt.figure(figsize=(10, 6))

plt.loglog(df['N'], df['StandardError'], 'r--', label=r'SE ($\frac{1}{\sqrt{N}}$)', linewidth=2)
plt.loglog(df['N'], df['AbsoluteError'], 'bo', markersize=4, alpha=0.6, label='AE')

plt.xlabel('Количество точек N')
plt.ylabel('Стандартная ошибка')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.savefig('error_plot.png', dpi=300)
