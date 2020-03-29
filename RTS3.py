from typing import NamedTuple
from random import randint
from time import time
from math import ceil, sin, cos, pi, degrees
import matplotlib.pyplot as plt
import numpy as np


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


N = 100
n = 6
w_frequency = 1200
x_t = list(range(N))
text = []

Signal1 = Signal(randint(1, 100), randint(1, 100), w_frequency, 'no.1')


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

@plotting
def DFT(N: int, signal: list):
    def w_count(N: int) -> list:
        res = []
        for p in range(N):
            tmp = []
            for k in range(N):
                angle = degrees(2*pi*p*k/N)
                if angle == 0 or angle % 180 == 0:
                    w_pk = cos(2*pi*p*k/N)
                elif angle % 90 == 0:
                    w_pk = -1j*round(sin(2*pi*p*k/N), 2)
                else:
                    w_pk = round(cos(2*pi*p*k/N), 2) - \
                           1j*round(sin(2*pi*p*k/N), 2)
                tmp.append(w_pk)
            res.append(tmp)
        return res
    w_pks = np.array(w_count(N))
    result = np.array(signal).dot(w_pks)
    return result


def check(signal: list):
    '''
    Function that uses NumPy to calculate DFT. Creates a plot.
    '''
    signal = np.array(signal)
    result = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1])
    plt.title(f"Numy DFT")
    plt.plot(x_t, result.real,'red', x_t, result.imag)
    plt.savefig('lab3_np.png')


a1 = generate_harmonics(n, Signal1.A, Signal1.phi, Signal1.w_freq)
s1 = generate_random_signal(a1)
fur = DFT(N, s1)
ch = check(s1)