add_library(config "")

target_sources(
  config
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ConfigMap.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/inih/ini.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/inih/INIReader.cpp)

target_include_directories(
  config PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/src
                ${CMAKE_SOURCE_DIR}/src/utils ${CMAKE_CURRENT_SOURCE_DIR}/inih)

if(EULER_KOKKOS_USE_MPI)
  target_link_libraries(config PUBLIC MPI::MPI_CXX)
endif(EULER_KOKKOS_USE_MPI)

# Add an alias so that library can be used inside the build tree
add_library(EulerKokkos::config ALIAS config)
