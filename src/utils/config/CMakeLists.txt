add_library(euler_kokkos_config ConfigMap.cpp inih/ini.cpp inih/INIReader.cpp)

target_include_directories(
  euler_kokkos_config PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/src
                             ${CMAKE_SOURCE_DIR}/src/utils ${CMAKE_CURRENT_SOURCE_DIR}/inih)

if(EULER_KOKKOS_USE_MPI)
  target_link_libraries(euler_kokkos_config PUBLIC MPI::MPI_CXX)
endif(EULER_KOKKOS_USE_MPI)

# Add an alias so that library can be used inside the build tree
add_library(EulerKokkos::config ALIAS euler_kokkos_config)
