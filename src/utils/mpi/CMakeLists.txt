add_library(mpiUtils MpiComm.h MpiComm.cpp GlobalMpiSession.cpp
                     GlobalMpiSession.h ParallelEnv.h ParallelEnv.cpp)

# Set target properties
target_include_directories(
  mpiUtils PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>)

target_link_libraries(mpiUtils PUBLIC MPI::MPI_CXX)

# Add an alias so that library can be used inside the build tree
add_library(EulerKokkos::mpiUtils ALIAS mpiUtils)
