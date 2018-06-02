#!/bin/bash
#BSUB -x
#BSUB -n 8                                          # number of MPI tasks
#BSUB -gpu "num=4:mode=exclusive_process:mps=no:j_exclusive=yes"
#BSUB -J ppkMHD_mpi_kokkos_cuda                       # Job name
#BSUB -o ppkMHD_mpi_kokkos_cuda.%J.out                # stdout filename
#BSUB -q computet1                                  # queue name
#BSUB -a p8aff(5,8,1,balance)                       # 5 threads/pask, so that only 2 tasks/CPU
#BSUB -R 'span[ptile=4]'                            # tile : number of MPI task/node (1 MPI task <--> 1 GPU)
#BSUB -W 00:05


module load at/10.0 gcc/4.8/ompi/2.1 cuda/9.0

NUMBER_OF_GPUS_PER_NODES=4

EXE_NAME=euler_kokkos
SETTINGS=settings.ini

# Nominal behavior: each mpi task binded to a different GPU
# Note that option "-gpu" is necessary when using IBM Spectrum MPI
# and not necessary when using cuda-aware regular OpenMPI
mpirun -gpu --report-bindings ./$EXE_NAME $SETTINGS --ndevices=$NUMBER_OF_GPUS_PER_NODES
