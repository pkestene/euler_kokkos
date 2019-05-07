# config file reader / ConfigMap / monitoring timer
add_subdirectory(utils)

# common solver code
add_subdirectory(shared)

# Hydro solver using 2nd order Muscl-Hancock solver
add_subdirectory(muscl)

#
# euler_kokkos executable
#
add_executable(${PROJECT_NAME} "")

target_sources(${PROJECT_NAME}
  PRIVATE
  main.cpp)

target_include_directories(${PROJECT_NAME}
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/src
  )

target_link_libraries(${PROJECT_NAME} 
  PUBLIC
  EulerKokkos::solver_factory
  EulerKokkos::config kokkos
  dl 
  EulerKokkos::muscl
  EulerKokkos::shared 
  EulerKokkos::io
  EulerKokkos::monitoring)

if (USE_MPI)
  target_link_libraries(${PROJECT_NAME}
    PUBLIC
    EulerKokkos::mpiUtils)
endif(USE_MPI)
