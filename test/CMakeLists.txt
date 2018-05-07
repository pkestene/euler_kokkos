add_subdirectory(kokkos)

add_subdirectory(backward-cpp)
add_subdirectory(io)
add_subdirectory(configMap)

if(USE_MPI)
  add_subdirectory(mpiBasic)
endif(USE_MPI)

