find_package(OpenMP)

add_library(euler_kokkos_monitoring OpenMPTimer.cpp SimpleTimer.cpp)

target_include_directories(
  euler_kokkos_monitoring
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

if(OpenMP_CXX_FOUND)
  target_link_libraries(euler_kokkos_monitoring PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_CXX_FOUND)

# Add an alias so that library can be used inside the build tree
add_library(EulerKokkos::monitoring ALIAS euler_kokkos_monitoring)
