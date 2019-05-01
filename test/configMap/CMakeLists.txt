#
# ConfigMap class test (src/util/config)
#

##############################################
add_executable(ConfigMapTest
  ConfigMapTest.cpp)

target_link_libraries(ConfigMapTest
  PRIVATE
  EulerKokkos::config)

##############################################
if(USE_MPI)
  
  configure_file(test_mpi.ini test_mpi.ini COPYONLY)

  add_executable(ConfigMapTest_mpi
    ConfigMapTest_mpi.cpp)

  target_link_libraries(ConfigMapTest_mpi
    PRIVATE
    EulerKokkos::config
    EulerKokkos::mpiUtils)
  
endif(USE_MPI)
