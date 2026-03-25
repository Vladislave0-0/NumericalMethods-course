import numpy as np

def relative_error(measured, actual):
    if actual == 0:
        return 0.0
    return abs(measured - actual) / abs(actual) * 100

def fast_variance(data):
    n = len(data)
    if n < 1: return 0.0
    sum_x = np.sum(data)
    sum_x2 = np.sum(data**2)
    var = (sum_x2 - (sum_x**2) / n) / n
    return var

def two_pass_variance(data):
    n = len(data)
    if n < 1: return 0.0
    mean = np.mean(data)
    sum_diff2 = np.sum((data - mean)**2)
    return sum_diff2 / n

def one_pass_variance(data):
    n = 0
    mean = 0.0
    M2 = 0.0
    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
    return M2 / n

def run_experiment(mu, sigma):
    np.random.seed(42)
    raw_data = np.random.normal(mu, sigma, 1000)
    
    data_high = raw_data.astype(np.float64)
        
    reference_var = one_pass_variance(data_high)
    
    results = {}
    for dtype in [np.float32, np.float64]:
        data = raw_data.astype(dtype)
        results[dtype.__name__] = {
            'Быстрый': fast_variance(data),
            'Двухпроходной': two_pass_variance(data),
            'Однопроходной': one_pass_variance(data)
        }
    
    return reference_var, results

samples = [(1, 1), (10, 0.1), (100, 0.01)]

print(f"{'Выборка (mu, sigma)':<20} | {'Тип':<10} | {'Метод':<15} | {'Отн. ошибка (%)':<15}")
print("-" * 70)

for mu, sigma in samples:
    ref, res_all = run_experiment(mu, sigma)
    
    for dtype_name, methods in res_all.items():
        first_in_dtype = True
        for method_name, val in methods.items():
            err = relative_error(val, ref)
            
            mu_sig_str = f"({mu:g}, {sigma:g})" if (first_in_dtype and dtype_name == 'float32') else ""
            dt_str = dtype_name if first_in_dtype else ""
            
            print(f"{mu_sig_str:<20} | {dt_str:<10} | {method_name:<15} | {err:15.10f}")
            first_in_dtype = False
        print("-" * 70)
