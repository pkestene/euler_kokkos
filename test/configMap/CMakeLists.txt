#
# ConfigMap class test (src/util/config)
#

##############################################
add_executable(ConfigMapTest
  ConfigMapTest.cpp)
target_include_directories(ConfigMapTest
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/utils/config
  )
target_link_libraries(ConfigMapTest config)

##############################################
if(USE_MPI)
  
  configure_file(test_mpi.ini test_mpi.ini COPYONLY)

  add_executable(ConfigMapTest_mpi
    ConfigMapTest_mpi.cpp)
  target_include_directories(ConfigMapTest_mpi
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/utils/config
    ${CMAKE_SOURCE_DIR}/src/utils/io
    ${CMAKE_SOURCE_DIR}/src/utils/mpiUtils
  )
  target_link_libraries(ConfigMapTest_mpi config mpiUtils)
  
endif(USE_MPI)
