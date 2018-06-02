#!/bin/bash
#BSUB -x
#BSUB -J ppkMHD_mpi_kokkos_openmp                  # Job name
#BSUB -n 4                                         # total number of MPI task
#BSUB -o ppkMHD_mpi_kokkos_openmp.%J.out           # stdout filename
#BSUB -q computet1                                 # queue name
#BSUB -a p8aff(10,8,1,balance)                     # 10 OpenMP thread/task, SMT=8
#BSUB -R 'span[ptile=2]'                           # tile : number of MPI task/node
#BSUB -W 00:05


module load at/10.0 gcc/4.8/ompi/2.1

EXE_NAME=euler_kokkos
SETTINGS=settings.ini

# report bindings for cross-checking
mpirun --report-bindings ./$EXE_NAME ./$SETTINGS
