from typing import NamedTuple
from random import randint
from time import time
from math import ceil, sin, cos, pi, degrees
import matplotlib.pyplot as plt
import numpy as np

N = 64
n = 8
w_frequency = 1200
x_t = list(range(N))
text = []

class Signal(NamedTuple):
    '''
    Parameters of a random signal:
        A is an amplitude;
        phi is an angle;
        w_frequency is a max frequency;
        name is a label.
    '''
    A: int
    phi: int
    w_freq: int
    name: str

Signal1 = Signal(randint(1, 100), randint(1, 100), w_frequency, 'no.1')


def calculation_time(func):
    '''
    A decorator, which prints time passed since the function executed.
    '''
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        if func.__name__ != 'wrapper':
            print(f"Time to count {func.__name__}: {(time() - start)*10**6:}"
                  f".{3} us.")
        if isinstance(result, float):
            result = ceil(result)
            if result > 10000:
                print(f"{func.__name__} = {result/1000} * 10^3")
            else:
                print(f"{func.__name__} = {result}")
        elif isinstance(result, list) or isinstance(result, tuple):
            if func.__name__ not in ['generate_harmonics',
                                     'generate_random_signal']:
                print(f"{func.__name__} = {result}\n")
        return result
    return wrapper


def plotting(func):
    '''
    Builds a plot and saves it in a current directory.
    '''
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        result = func(*args, **kwargs)
        plt.figure(wrapper.count)
        if func.__name__ == 'generate_harmonics':
            plt.title(f"Harmonics no.{wrapper.count}")
            for i in range(len(result)):
                plt.plot(x_t, result[i])
            plt.savefig(f'lab3_harm{wrapper.count}.png')
            plt.close(wrapper.count)
        elif func.__name__ == 'generate_random_signal':
            plt.title(f"Random signal no.{wrapper.count}")
            plt.plot(x_t, result)
            plt.savefig(f'lab3_signal{wrapper.count}.png')
            plt.close(wrapper.count)
        elif func.__name__ == 'DFT':
            x0_t = np.array(x_t, dtype=complex)
            plt.title(f"DFT")
            plt.plot(x0_t, result.real, 'red', x0_t, result.imag)
            plt.savefig(f'lab3_dft.png')
            plt.close(wrapper.count)
        return result
    wrapper.count = 0
    return wrapper


@plotting
def generate_harmonics(n: int,
                       A: int,
                       phi: int,
                       w_freq: int) -> list:
    '''A function to generate n-harmonics.'''
    harmonic = []
    for i in range(n):
        harmonic.append(list(map(
                        lambda x: A*sin(w_freq/(i+1)*x_t.index(x) + phi),
                        x_t)))
    return harmonic


@plotting
def generate_random_signal(harmonic: list) -> list:
    '''
    A function, which generates a random signal
    from n-harmonic.
    '''
    tmp = list(zip(*harmonic))
    x_t = []
    for element in tmp:
        x_t.append(sum(element))
    return x_t

def DFT(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd, X_even + factor[N / 2:] * X_odd])

a1 = generate_harmonics(n, Signal1.A, Signal1.phi, Signal1.w_freq)
s1 = generate_random_signal(a1)
fur = DFT(s1)
fastfur = FFT(s1)