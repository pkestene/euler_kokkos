set(MPI_UTILS_SRCS ParallelEnv.h ParallelEnv.cpp)

if(EULER_KOKKOS_USE_MPI)
  list(
    APPEND
    MPI_UTILS_SRCS
    GlobalMpiSession.cpp
    GlobalMpiSession.h
    MpiComm.h
    MpiComm.cpp
    MpiCommCart.h
    MpiCommCart.cpp)
endif(EULER_KOKKOS_USE_MPI)

add_library(euler_kokkos_mpiUtils ${MPI_UTILS_SRCS})

# Set target properties
target_include_directories(
  euler_kokkos_mpiUtils
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
         $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>)

target_link_libraries(euler_kokkos_mpiUtils PUBLIC Kokkos::kokkos)
if(EULER_KOKKOS_USE_MPI)
  target_link_libraries(euler_kokkos_mpiUtils PUBLIC MPI::MPI_CXX)
endif(EULER_KOKKOS_USE_MPI)

# Add an alias so that library can be used inside the build tree
add_library(EulerKokkos::mpiUtils ALIAS euler_kokkos_mpiUtils)
