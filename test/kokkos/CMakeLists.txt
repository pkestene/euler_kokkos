#
# Kokkos test example : query device information
#

add_executable(query_device_kokkos
  query_device_kokkos.cpp)
target_link_libraries(query_device_kokkos kokkos hwloc)


