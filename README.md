[![DOI](https://zenodo.org/badge/168991445.svg)](https://zenodo.org/badge/latestdoi/168991445) ![C/C++ CI](https://github.com/pkestene/euler_kokkos/workflows/C/C++%20CI/badge.svg)

# Euler_kokkos

## What is it ?

Provide performance portable (multi-architecture) Kokkos implementation for compressible hydrodynamics (second order, Godunov, MUSCL-Hancock) on cartesian grids.

## Dependencies

* [Kokkos](https://github.com/kokkos/kokkos) library will be built by euler_kokkos using the same flags (architecture, optimization, ...).
* [cmake](https://cmake.org/) with version >= 3.X (3.X is chosen to meet Kokkos own requirement for CMake; i.e. it might increase in the future)

Current application is configured with kokkos library as a git submodule. So you'll need to run the following git commands right after cloning euler_kokkos:

```shell
git submodule init
git submodule update
```

Kokkos is built with the same flags as the main application.

## Build

A few example builds, with minimal configuration options.

### Build without MPI / With Kokkos-openmp

* Create a build directory, configure and make

```shell
mkdir build; cd build
cmake -DUSE_MPI=OFF -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_HWLOC=ON ..
make -j 4
```

Add variable CXX on the cmake command line to change the compiler (clang++, icpc, pgcc, ....).

### Build without MPI / With Kokkos-openmp for Intel KNL

* Create a build directory, configure and make

```shell
export CXX=icpc
mkdir build; cd build
cmake -DUSE_MPI=OFF -DKokkos_ARCH_KNL=ON -DKokkos_ENABLE_OPENMP=ON ..
make -j 4
```

### Build without MPI / With Kokkos-cuda

**NOT NEEDED ANYMORE (with Kokkos v3.5)**
To be able to build with CUDA backend, you need to use `nvcc_wrapper` located in
kokkos source (external/kokkos/bin/nvcc_wrapper).

* Create a build directory, configure and make

```shell
mkdir build; cd build
# exporting CXX to nvcc_wrapper is not needed anymore
#export CXX=/path/to/nvcc_wrapper
cmake -DUSE_MPI=OFF -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_MAXWELL50=ON ..
make -j 4
```

`nvcc_wrapper` is a compiler wrapper arroud NVIDIA `nvcc`. It is available from Kokkos sources: `external/kokkos/bin/nvcc_wrapper`. Any Kokkos application target NVIDIA GPUs must be built with `nvcc_wrapper`.

### Build with MPI / With Kokkos-cuda

Please make sure to use a CUDA-aware MPI implementation (OpenMPI or MVAPICH2) built with the proper flags for activating CUDA support.

It may happen that eventhough your MPI implementation is actually cuda-aware, cmake find_package macro for MPI does not detect it to be cuda aware. In that case, you can enforce cuda awareness by turning option `USE_MPI_CUDA_AWARE_ENFORCED` to ON.

You don't need to use mpi compiler wrapper mpicxx, cmake *should* be able to correctly populate `MPI_CXX_INCLUDE_PATH`, `MPI_CXX_LIBRARIES` which are passed to all final targets.

* Create a build directory, configure and make

```shell
mkdir build; cd build
# exporting CXX to nvcc_wrapper is not needed anymore
#export CXX=/path/to/nvcc_wrapper
cmake -DUSE_MPI=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_MAXWELL50=ON ..
make -j 4
```

Example command line to run the application (1 GPU used per MPI task)

```shell
mpirun -np 4 ./euler_kokkos ./test_implode_2D_mpi.ini
```

### Developping with vim or emacs and semantic completion/navigation from ccls

Make sure to have CMake variable `CMAKE_EXPORT_COMPILE_COMMANDS` set to ON, it will generate a file named _compile_commands.json_.
Then you can symlink the generated file in the top level source directory.

Please visit :
* [ccls](https://github.com/MaskRay/ccls)
* [editor configuration for using ccls](https://github.com/MaskRay/ccls/wiki/Editor-Configuration)
* [project setup for using ccls](https://github.com/MaskRay/ccls/wiki/Project-Setup)

## Build Documentation

A Sphinx/html documentation will (hopefully) soon be populated.

To build it:

``` shell
mkdir build
cd build
cmake .. -DBUILD_CODE:BOOL=OFF -DBUILD_DOC:BOOL=ON -DDOC:STRING=html
```

Building documentation requires to have python3 with up-to-date breathe extension.
