# SPDX-FileCopyrightText: 2025 euler_kokkos authors
#
# SPDX-License-Identifier: Unlicense

add_subdirectory(kokkos)

# backtrace does not build anymore with cuda 10.1
#
# add_subdirectory(backward-cpp)

add_subdirectory(io)
add_subdirectory(configMap)

if(EULER_KOKKOS_USE_MPI)
  add_subdirectory(mpiBasic)
endif(EULER_KOKKOS_USE_MPI)
