add_subdirectory(kokkos)

add_subdirectory(io)

if(USE_MPI)
  add_subdirectory(mpiBasic)
endif(USE_MPI)

