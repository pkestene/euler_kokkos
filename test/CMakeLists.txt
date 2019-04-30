add_subdirectory(kokkos)

# backtrace does not build anymore with cuda 10.1
#add_subdirectory(backward-cpp)
add_subdirectory(io)
add_subdirectory(configMap)

if(USE_MPI)
  add_subdirectory(mpiBasic)
endif(USE_MPI)

