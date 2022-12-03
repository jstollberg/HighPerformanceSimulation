# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 16:25:55 2022

@author: jonat
"""
import numpy as np
import matplotlib.pyplot as plt

matrix_sizes = [10, 100, 500, 1000, 2000, 4000, 5200]
local_work_sizes = [1, 10, 20, 50, 1000]

time_sequential = [0.01, 0.07, 0.29, 1.17, 4.53, 18.24, 30.82]
time_cpu = [0.09, 0.50, 0.35, 1.23, 4.69, 16.77, 28.41]
time_gpu = [[0.05, 0.04, 0.10, 0.18, 0.40, 1.12, 0.80],
            [0.03, 0.07, 0.30, 2.56, 1.76, 6.26, 5.79],
            [0.03, 0.04, 0.13, 0.21, 0.35, 0.99, 1.10],
            [0.06, 0.04, 0.10, 0.18, 0.36, 0.96, 0.84],
            [0.03, 0.04, 0.09, 0.14, 0.28, 0.95, None],
            [0.03, 0.04, 0.17, 2.33, 1.99, 4.30, None]]

# convert to numpy arrays
time_sequential = np.array(time_sequential)
time_cup = np.array(time_cpu)
time_gpu = np.array(time_gpu)

# first get all values with automatic opencl lws
time_gpu_auto = time_gpu[0,:]

# make time plot for auto lws
fig1, ax1 = plt.subplots(dpi=600)
ax1.plot(matrix_sizes, time_sequential, label="sequentiell", marker="o")
# ax1.plot(matrix_sizes, time_cpu, label="Stream", marker="o")
ax1.plot(matrix_sizes, time_gpu_auto, label="OpenCL", marker="o")
ax1.set(xlabel="Problemgröße m", ylabel="Zeit [ms]")
# ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.legend(loc="upper left")
ax1.grid()

# speedup
speedup = time_sequential/time_gpu_auto
fig2, ax2 = plt.subplots(dpi=600)
ax2.plot(matrix_sizes, speedup, marker="o")
ax2.set(xlabel="Problemgröße m", ylabel="Speedup")
ax2.grid()

# lws vs time and speedup
times = time_gpu[:,5]
times = times[1::]
fig3, ax3 = plt.subplots(dpi=600)
ax3.plot(local_work_sizes, times, marker="o")
ax3.grid()

fig34, ax4 = plt.subplots(dpi=600)
X, Y = np.meshgrid(matrix_sizes, local_work_sizes)
Z = time_gpu[1::,:]
cp = ax4.contourf(X, Y, Z)
# fig.colorbar(cp) # Add a colorbar to a plot


# fig1.savefig("auto_lws.pdf")
# fig2.savefig("auto_speedup.pdf")

# plt.show()


