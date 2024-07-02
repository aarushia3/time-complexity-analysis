import timeit
import functools
import matplotlib.pyplot as plt

# This function measures how much time a function takes
def measure_time(func_name, *args, **kwargs):
    func = globals().get(func_name)
    
    if not callable(func):
        raise ValueError(f"No function named '{func_name}' found")

    def wrapper():
        return func(*args, **kwargs)
    
    execution_time = timeit.timeit(wrapper, number=10)
    print(f"Execution Time: {execution_time/10}")
    return execution_time/10

def analyze_time_complexity(func_name, input_sizes, *args_template, **kwargs_template):
    times = []
    for size in input_sizes:
        args = [arg(size) if callable(arg) else arg for arg in args_template]
        kwargs = {k: (v(size) if callable(v) else v) for k, v in kwargs_template.items()}
        
        elapsed_time = measure_time(func_name, *args, **kwargs)
        times.append(elapsed_time)
        print(f"Size: {size}, Time: {elapsed_time:.6f} seconds")
    
    plt.plot(input_sizes, times, marker='o')
    plt.xlabel('Input Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Time Complexity of {func_name}')
    plt.grid(True)
    plt.show()

def foo(n):
    return sum(range(n))

def bar(n):
    return [i * i for i in range(n**2)]

input_sizes = [i for i in range(1, 1000, 5)]
analyze_time_complexity('foo', input_sizes, lambda n: n)
analyze_time_complexity('bar', input_sizes, lambda n: n)