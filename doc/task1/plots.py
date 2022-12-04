# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 16:25:55 2022

@author: jonat
"""
import numpy as np
import matplotlib.pyplot as plt

matrix_sizes = [10, 1000, 2000, 4000, 8000, 10000, 15000]

matrix_sizes_2 = [1000, 2000, 4000, 5000, 6000, 8000, 10000]
local_work_sizes = [5, 10, 20, 25, 50, 100]

time_sequential = [0.01, 7.54, 4.56, 18.29, 73.59, 115.02, 259.41]
time_cpu = [0.73, 9.62, 4.36, 23.11, 69.50, 105.83, 237.30]
time_gpu = [[0.04, 0.14, 0.31, 1.51, 3.48, 1.95, 3.48]]
time_gpu_2 = [[0.08, 0.12, 0.33, 0.64, 0.55, 1.21, 1.48],
              [0.08, 0.12, 0.32, 0.63, 0.53, 1.06, 1.14],
              [0.08, 0.12, 0.37, 0.66, 0.55, 1.14, 1.01],
              [0.09, 0.13, 0.40, 0.63, 0.58, 1.21, 1.01],
              [0.12, 0.15, 0.41, 0.82, 0.61, 1.41, 1.04],
              [0.12, 0.19, 1.73, 0.70, 0.63, 1.44, 1.06]]

# convert to numpy arrays
time_sequential = np.array(time_sequential)
time_cup = np.array(time_cpu)
time_gpu = np.array(time_gpu)
time_gpu_2 = np.array(time_gpu_2)

# first get all values with automatic opencl lws
time_gpu_auto = time_gpu[0,:]

# make time plot for auto lws
fig1, ax1 = plt.subplots(dpi=600)
ax1.plot(matrix_sizes, time_sequential, label="sequentiell", marker="o")
ax1.plot(matrix_sizes, time_cpu, label="Stream", marker="o")
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
times = time_gpu_2[:,6]
fig3, ax3 = plt.subplots(dpi=600)
ax3.plot(local_work_sizes, times, marker="o")
ax3.set(xlabel="local size", ylabel="Zeit [ms]")
ax3.set(xlim=[0,105])
ax3.grid()

fig4, ax4 = plt.subplots(dpi=600)
X, Y = np.meshgrid(matrix_sizes_2, local_work_sizes)
Z = time_gpu_2
cp = ax4.contourf(X, Y, Z)
ax4.set(xlabel="Problemgröße m", ylabel="local size")
fig4.colorbar(cp) # Add a colorbar to a plot


# fig1.savefig("auto_lws.pdf")
# fig2.savefig("auto_speedup.pdf")
fig3.savefig("lws_variation.pdf")
fig4.savefig("contour.pdf")

# plt.show()


