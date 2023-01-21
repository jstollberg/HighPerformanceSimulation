import numpy as np
import matplotlib.pyplot as plt

# results for 16 processes
matrix_dims = [1000000, 4000000, 9000000, 16000000, 25000000, 36000000, 49000000, 64000000]
times_mpi = [0.0, 1.0, 4.0, 8.0, 15.0, 26.0, 42.0, 61.0]
times_serial = [2.0, 15.0, 50.0, 121.0, 210.0, 364.0, 650.0, 865.0]
times_stream =  [1.0, 1.0, 3.0, 5.0, 9.0, 17.0, 26.0, 31.0]

speedup_procs = [4, 16, 25, 64]
speedup_time_serial = 210.0
speedup_times_mpi = [55.0, 15.0, 9.0, 5.0]
speedup_times_stream = [37.0, 8.0, 6.0, 2.0]

speedup_time_serial = 121
speedup_times_mpi = [29.0, 9.0, 7.0, 2.0]
speedup_times_stream = [20.0, 5.0, 3.0, 2.0]

# convert to numpy arrays
times_mpi = np.array(times_mpi)
times_serial = np.array(times_serial)
times_stream = np.array(times_stream)

# make time plot for auto lws
fig1, ax1 = plt.subplots(dpi=600)
ax1.plot(matrix_dims, times_mpi, label="MPI", marker="o")
ax1.plot(matrix_dims, times_serial, label="sequentiell", marker="o")
ax1.plot(matrix_dims, times_stream, label="Stream", marker="o")
ax1.set(xlabel="Matrixdimension n", ylabel="Zeit [s]")
ax1.set_yscale("log")
ax1.legend(loc="upper left")
ax1.grid()

# compute speedup
speedup_mpi = []
for i in speedup_times_mpi:
    speedup_mpi.append(speedup_time_serial/i)
speedup_stream = []
for i in speedup_times_stream:
    speedup_stream.append(speedup_time_serial/i)
    
# visualize speedup
fig2, ax2 = plt.subplots(dpi=600)
ax2.plot(speedup_procs, speedup_mpi, label="MPI", marker="o")
ax2.plot(speedup_procs, speedup_stream, label="Stream", marker="o")
ax2.set(xlabel="Prozesse", ylabel="Speed-Up")
# ax2.set_yscale("log")
ax2.legend(loc="upper left")
ax2.grid()

# save plots
# fig1.savefig("16_procs.pdf")
# fig2.savefig("speedup.pdf")