#!/bin/bash
#SBATCH -J uebung1
#SBATCH -A kurs00060
#SBATCH -p kurs00060
#SBATCH --reservation=kurs00060
#SBATCH -e /work/home/kurse/kurs00060/js51xywu/ue1/out/ue1.err.%j
#SBATCH -o /work/home/kurse/kurs00060/js51xywu/ue1/out/ue1.out.%j
#SBATCH -t 00:15:00

#SBATCH --mem-per-cpu=30000
#SBATCH -n 1
#SBATCH --gres=gpu:a100:1

echo "This is Job  - $SLURM_JOB_ID"

 # first remove all modules and load required modules
module purge
module load java/16.0.1
module load cuda

 # call to the parallel program
srun java -jar HighPerformanceSimulation.jar -Xms20g