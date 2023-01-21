#!/bin/bash
#SBATCH -J MPICannon
#SBATCH -A kurs00060
#SBATCH -p kurs00060
#SBATCH --reservation=kurs00060
##SBATCH --mail-type=ALL
##SBATCH --mail-user=jonathan.stollberg@stud.tu-darmstadt.de
#SBATCH -e ./MPICannon.err.%j
#SBATCH -o ./MPICannon.out.%j
#SBATCH --mem-per-cpu=7000
#SBATCH --time=00:29:00
#SBATCH -n 1
#SBATCH -c 16

# first remove all modules and load required modules
module purge
module load intel/2020.1
module load java/16.0.1
module load gcc openmpi

# set MPJ_HOME
export MPJ_HOME=./mpj
export PATH=$MPJ_HOME/bin:$PATH

# call to the parallel program
srun mpjrun.sh -Xms5g -np 16 HlS_Task2_nonblocking.jar 1
srun mpjrun.sh -Xms5g -np 16 HlS_Task2_nonblocking.jar 2
srun mpjrun.sh -Xms5g -np 16 HlS_Task2_nonblocking.jar 3
srun mpjrun.sh -Xms5g -np 16 HlS_Task2_nonblocking.jar 4
srun mpjrun.sh -Xms5g -np 16 HlS_Task2_nonblocking.jar 5
srun mpjrun.sh -Xms5g -np 16 HlS_Task2_nonblocking.jar 6
srun mpjrun.sh -Xms5g -np 16 HlS_Task2_nonblocking.jar 7
srun mpjrun.sh -Xms5g -np 16 HlS_Task2_nonblocking.jar 8

