#
# Kokkos test example : query device information
#

add_executable(query_device_kokkos query_device_kokkos.cpp)
target_link_libraries(query_device_kokkos PUBLIC Kokkos::kokkos)
if(EULER_KOKKOS_USE_MPI)
  target_link_libraries(query_device_kokkos PUBLIC MPI::MPI_CXX)
endif(EULER_KOKKOS_USE_MPI)
