import timeit
import functools
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy.stats import zscore

def remove_outliers(x, y, threshold=1):
    z_scores = zscore(y)
    mask = np.abs(z_scores) < threshold
    return x[mask], y[mask]

def fit_curves(x, y):
    def constant(x, a):
        return a * np.ones_like(x)
    
    def logarithmic(x, a, b):
        return a * np.log(x) + b
    
    def linear(x, a, b):
        return a * x + b
    
    def linearithmic(x, a, b):
        return a * x * np.log(x) + b
    
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    def exponential(x, a, b, c):
        return a * np.exp(b * x) + c

    functions = [constant, logarithmic, linear, linearithmic, quadratic, exponential]
    labels = ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 'O(n^2)', 'O(2^n)']

    best_fit = None
    best_rmse = float('inf')

    for func, label in zip(functions, labels):
        try:
            popt, _ = curve_fit(func, x, y)
            y_fit = func(x, *popt)
            rmse = np.sqrt(np.mean((y - y_fit) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_fit = label
        except RuntimeError:
            continue

    return best_fit

def analyze_time_complexity(func_name, input_sizes, func, num_iterations=10):
    times = []
    for size in input_sizes:
        stmt = f'{func_name}({size})'
        setup = f'from __main__ import {func_name}'
        timeit.timeit(stmt, setup=setup, number=20)
        time = timeit.timeit(stmt, setup=setup, number=num_iterations) / num_iterations
        times.append(time)
    return times

def foo(n):
    for i in range(n):
        pass

def bar(n):
    for i in range(n):
        for j in range(n):
            pass

def baz(n):
    while n > 1:
        n = n // 2

def plot_data(input_sizes, times, title):
    plt.figure(figsize=(10, 6))
    plt.plot(input_sizes, times, 'o-', label='Timing Data')
    plt.xlabel('Input Size')
    plt.ylabel('Time (s)')
    plt.title(title)
    plt.legend()
    plt.show()

input_sizes = [i for i in range(100, 1000, 1)]

# Analyze foo
times = analyze_time_complexity('foo', input_sizes, lambda n: n)
plot_data(input_sizes, times, 'Timing Data for foo (before removing outliers)')
input_sizes, times = remove_outliers(np.array(input_sizes), np.array(times))
plot_data(input_sizes, times, 'Timing Data for foo (after removing outliers)')
best_fit = fit_curves(input_sizes, times)
print(f"The best fit for foo is: {best_fit}")

# # Analyze bar
# times = analyze_time_complexity('bar', input_sizes, lambda n: n)
# plot_data(input_sizes, times, 'Timing Data for bar (before removing outliers)')
# input_sizes, times = remove_outliers(np.array(input_sizes), np.array(times))
# plot_data(input_sizes, times, 'Timing Data for bar (after removing outliers)')
# best_fit = fit_curves(input_sizes, times)
# print(f"The best fit for bar is: {best_fit}")

# # Analyze baz
# times = analyze_time_complexity('baz', input_sizes, lambda n: n)
# plot_data(input_sizes, times, 'Timing Data for baz (before removing outliers)')
# input_sizes, times = remove_outliers(np.array(input_sizes), np.array(times))
# plot_data(input_sizes, times, 'Timing Data for baz (after removing outliers)')
# best_fit = fit_curves(input_sizes, times)
# print(f"The best fit for baz is: {best_fit}")