#!/bin/bash
#SBATCH -J uebung1
#SBATCH -A kurs00060
#SBATCH -p kurs00060
#SBATCH --reservation=kurs00060
#  #SBATCH --mail-type=ALL
#  #SBATCH --mail-user=marcel.heis@tu-darmstadt.de
#SBATCH -e /work/home/kurse/kurs00060/lb82noru/ue1/out/ue1/log/ue1.err.%j
#SBATCH -o /work/home/kurse/kurs00060/lb82noru/ue1/out/ue1/log/ue1.out.%j

#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:30
#SBATCH -n 1
#SBATCH --gres=gpu:a100:1

 # first remove all modules and load required modules
module purge
module load java/16.0.1
module load cuda

 # call to the parallel program
srun java -jar HighPerformanceSimulation.jar