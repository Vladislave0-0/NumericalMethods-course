import numpy as np
from scipy import stats
from scipy.special import erfc

def generate_random_bits(n):
    return np.random.randint(0, 2, n)

def run_fourier_test(bits):
    # Преобразование в -1 и 1
    n = len(bits)
    x = 2 * bits - 1

    # БПФ
    f_transform = np.fft.fft(x)
    magnitudes = np.abs(f_transform[:n // 2])

    # Порог по стандарту NIST
    threshold = np.sqrt(np.log(1/0.05) * n)

    # Подсчет пиков ниже порога
    n0 = 0.95 * (n / 2)
    n1 = np.sum(magnitudes < threshold)

    # Статистика отклонения
    dist = (n1 - n0) / np.sqrt(n * 0.95 * 0.05 / 4)
    p_value = erfc(np.abs(dist) / np.sqrt(2))

    return p_value

def run_autocorrelation_test(bits, lag=1):
    # Считаем количество несовпадающих бит
    n = len(bits)
    diffs = np.sum(bits[:n-lag] ^ bits[lag:])

    # Ожидаемое распределение - биномиальное
    mu = (n - lag) * 0.5
    sigma = np.sqrt((n - lag) * 0.25)
    z = (diffs - mu) / sigma
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))

    return p_value

def run_gaps_test(bits):
    # Индексы всех единиц
    ones_idx = np.where(bits == 1)[0]
    if len(ones_idx) < 2: return 1.0
    
    # Длины промежутков (количество нулей между единицами)
    gaps = np.diff(ones_idx) - 1
    
    # Группируем по количеству совпадений от 1 до 5
    max_gap = 5
    observed, _ = np.histogram(gaps, bins=list(range(max_gap + 1)) + [np.inf])
    
    # P(gap=k) = (1/2)^(k+1)
    expected_probs = [0.5**(k+1) for k in range(max_gap)]
    expected_probs.append(1 - sum(expected_probs))
    expected_freqs = np.array(expected_probs) * len(gaps)
    
    chi_stat, p_value = stats.chisquare(observed, f_exp=expected_freqs)

    return p_value

def two_level_validation(test_func, iterations=50, size=1024):
    p_values = []
    for _ in range(iterations):
        bits = generate_random_bits(size)
        p_values.append(test_func(bits))
    
    # Проверка набора p-values на равномерность (использую KS-тест)
    _, final_p = stats.kstest(p_values, 'uniform')
    return final_p

if __name__ == "__main__":
    np.random.seed(42)
    print("Результаты двухуровневого тестирования битов:")
    
    p_fourier = two_level_validation(run_fourier_test, iterations=100, size=4096)
    print(f"1. Тест Фурье: p = {p_fourier:.5f}")
    
    p_auto = two_level_validation(run_autocorrelation_test, iterations=100, size=1000)
    print(f"2. Автокорреляция: p = {p_auto:.5f}")
    
    p_gaps = two_level_validation(run_gaps_test, iterations=100, size=10000)
    print(f"3. Тест промежутков: p = {p_gaps:.5f}")
