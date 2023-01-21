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
#SBATCH -c 4

# first remove all modules and load required modules
module purge
module load intel/2020.1
module load java/16.0.1
module load gcc openmpi

# set MPJ_HOME
export MPJ_HOME=./mpj
export PATH=$MPJ_HOME/bin:$PATH

# call to the parallel program
srun mpjrun.sh -Xms5g -np 4 HlS_Task2_speedup_25000000.jar 1
