# Euler_kokkos

## What is it ?

Provide performance portable Kokkos implementation for compressible hydrodynamics.

## Dependencies

* [Kokkos](https://github.com/kokkos/kokkos)
* [cmake](https://cmake.org/) with version >= 3.1
   
## Build

A few example builds

### Build without MPI / With Kokkos-openmp

* Create a build directory, configure and make
```shell
mkdir build; cd build
cmake -DUSE_MPI=OFF ..
make -j 4
```

Add variable CXX on the cmake command line to change the compiler (clang++, icpc, pgcc, ....)

### Build without MPI / With Kokkos-openmp for Intel KNL

* Create a build directory, configure and make
```shell
export CXX=icpc
mkdir build; cd build
cmake -DUSE_MPI=OFF -DHOST_ARCH=KNL ..
make -j 4
```

### Build without MPI / With Kokkos-cuda

* Create a build directory, configure and make
```shell
mkdir build; cd build
CXX=/path/to/nvcc_wrapper cmake -DUSE_MPI=OFF ..
make -j 4
```

### Build with MPI / With Kokkos-cuda


Please make sure to use a CUDA-aware MPI implementation (OpenMPI or MVAPICH2) built with the proper flags for activating CUDA support.


* Make sure MPI compiler wrapper will use `nvcc_wrapper` from Kokkos
```shell
export CXX=mpicxx 
export OMPI_CXX=/path/to/nvcc_wrapper
```

* Create a build directory, configure and make
```shell
mkdir build; cd build
cmake ..
make -j 4
```

Example command line to run the application (1 GPU used per MPI task)
```shell
mpirun -np 4 ./euler_kokkos ./test_implode_2D_mpi.ini --kokkos-ndevices=4
```

### Developping with vim and youcomplete plugin

Assuming you are using vim (or neovim) text editor and have installed the youcomplete plugin, you can have
semantic autocompletion in a C++ project.

Make sure to have CMake variable CMAKE_EXPORT_COMPILE_COMMANDS set to ON, and symlink the generated file to the top level
source directory.

