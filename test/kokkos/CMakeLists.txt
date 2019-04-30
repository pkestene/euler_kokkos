#
# Kokkos test example : query device information
#

add_executable(query_device_kokkos
  query_device_kokkos.cpp)
target_link_libraries(query_device_kokkos PUBLIC kokkos)
if (USE_MPI)
  target_compile_options    (query_device_kokkos PUBLIC ${MPI_CXX_COMPILE_FLAGS})
  target_include_directories(query_device_kokkos PUBLIC ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries     (query_device_kokkos PUBLIC ${MPI_CXX_LIBRARIES})
endif(USE_MPI)
