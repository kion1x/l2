from math import sin, cos, pi
from threading import Thread

import numpy as np
import random
import matplotlib.pyplot as plt


def plot(function):
    plt.figure(figsize=(25, 20))
    plt.plot(function, color="green")
    plt.grid(True)
    plt.savefig("task.png")

def test(new_N):
    n = 6
    w = 1200
    num = w / (n - 1)

    # frequency generation
    W = lambda n, w: w - n * num
    w_values = [W(n, w) for n in range(n)]
    x = np.zeros(new_N)

    # Signal X
    for j in range(n):
        amplitude = random.choice([i for i in range(-10, 10) if i != 0])
        phi = random.randint(-360, 360)
        for t in range(new_N):
            x[t] += amplitude * sin(w_values[j] * t + phi)

    # coefficients table w[p][k] (new_N/2)
    w_coeff = np.zeros(shape=(new_N // 2, new_N // 2))
    for p in range(new_N // 2):
        for k in range(new_N // 2):
            w_coeff[p][k] = cos(4 * pi / new_N * p * k) + sin(4 * pi / new_N * p * k)

    # new table w[p] (new_N)
    new_coeff = np.zeros(new_N)
    for p in range(new_N):
        new_coeff[p] = cos(2 * pi / new_N * p) + sin(2 * pi / new_N * p)

    F_odd = np.zeros(new_N // 2)
    F_nodd = np.zeros(new_N // 2)

    # final function
    F = np.zeros(new_N)

    # odd and even parts separately
    for p in range(new_N // 2):
        for k in range(new_N // 2):
            F_odd[p] += x[2 * k] * w_coeff[p][k]
            F_nodd[p] += x[2 * k + 1] * w_coeff[p][k]

    F_1half = np.zeros(new_N//2)
    F_2half = np.zeros(new_N//2)

    # function for the first thread to calculate the first half of N
    def T1(new_N):
        print(f"N = {new_N}\nThread 1 started!")
        for p in range(0, new_N//2):
            F_1half[p] += F_odd[p] + new_coeff[p] * F_nodd[p]
        print(f"Thread 1 finished!")

    # function for the second thread to calculate the second half of N (different formula)
    def T2(new_N):
        print(f"Thread 2 started!")
        for p in range(0, new_N//2):
            F_2half[p] += F_odd[p] - new_coeff[p + (new_N//2)] * F_nodd[p]
        print(f"Thread 2 finished!")
        print("\n")

    t1 = Thread(target=T1, args=(new_N,))
    t2 = Thread(target=T2, args=(new_N,))

    t1.start()
    t2.start()

    F = np.append(F_1half, F_2half)

    plot(F)

# calculating fft for different N
N_variants = [256, 910, 1054, 2048, 3000]
for n in N_variants:
    test(n)
