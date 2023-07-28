#!/usr/bin/bash

#$ -S /bin/bash 
#$ -pe smp 1
#$ -cwd
#$ -l h_rt=0:25:00
#$ -r y

. ../venv/bin/activate

export OMP_NUM_THREADS=${NSLOTS:-1}
python "$@" --seed $SGE_TASK_ID
