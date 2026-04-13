import numpy as np
from scipy import stats

def get_normal_sample(size):
    return np.random.standard_normal(size)

def calculate_ks_p_value(sample):
    result = stats.kstest(sample, 'norm')
    return result.pvalue

def perform_full_validation(iterations=100, sample_size=1000):
    collected_p_values = []
    
    # Собираем p-values от множества независимых выборок
    for _ in range(iterations):
        sample = get_normal_sample(sample_size)
        p_val = calculate_ks_p_value(sample)
        collected_p_values.append(p_val)
    
    # Проверяем, распределены ли p-values равномерно на [0, 1]
    final_stat, final_p_value = stats.kstest(collected_p_values, 'uniform')
    
    return final_p_value, collected_p_values

if __name__ == "__main__":
    np.random.seed(42)
    
    p_result, all_p = perform_full_validation(iterations=100, sample_size=1000)
    
    print(f"Результат двухуровневого теста: {p_result:.5f}")
    
    alpha = 0.05
    if p_result > alpha:
        print(f"Генератор прошел проверку (p > {alpha}).")
    else:
        print(f"Генератор не прошел проверку (p <= {alpha}).")
