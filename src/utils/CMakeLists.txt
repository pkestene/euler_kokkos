# SPDX-FileCopyrightText: 2025 euler_kokkos authors
#
# SPDX-License-Identifier: Unlicense

add_subdirectory(config)
add_subdirectory(monitoring)
add_subdirectory(io)
if(EULER_KOKKOS_USE_MPI)
  add_subdirectory(mpiUtils)
endif(EULER_KOKKOS_USE_MPI)
