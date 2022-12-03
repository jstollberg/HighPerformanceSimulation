# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 16:25:55 2022

@author: jonat
"""
import numpy as np
import matplotlib.pyplot as plt

matrix_sizes = [10, 1000, 2000, 4000, 8000, 10000, 15000]
local_work_sizes = [-1, 10, 20, 50, 100]

time_sequential = np.random.rand(7,7)
time_cpu = np.random.rand(7,7)
time_gpu = np.random.rand(7,7)

# convert to numpy arrays
time_sequential = np.array(time_sequential)
time_cup = np.array(time_cpu)
time_gpu = np.array(time_gpu)

# first get all values with automatic opencl lws
time_sequential_auto = time_sequential[0,:]
time_cpu_auto = time_cpu[0,:]
time_gpu_auto = time_gpu[0,:]

# make runtime plot for auto lws
fig1, ax1 = plt.subplots()
ax1.plot(matrix_sizes, time_sequential_auto, label="sequential")
ax1.plot(matrix_sizes, time_cpu_auto, label="stream")
ax1.plot(matrix_sizes, time_gpu_auto, label="OpenCL")
ax1.set(xlabel="problem size m", ylabel="runtime [s]", xlim=[0,16000])
ax1.legend(loc="upper left")
ax1.grid()

# fig1.savefig("auto_lws.pdf")
plt.show()


