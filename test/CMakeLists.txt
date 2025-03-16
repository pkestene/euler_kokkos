add_subdirectory(kokkos)
add_subdirectory(io)
add_subdirectory(configMap)

if(EULER_KOKKOS_USE_MPI)
  add_subdirectory(mpiBasic)
endif(EULER_KOKKOS_USE_MPI)
