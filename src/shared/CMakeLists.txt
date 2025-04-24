# ##############################################################################
# SHARED ############
# ##############################################################################
add_library(
  euler_kokkos_shared
  BoundariesFunctors.h
  BoundariesFunctorsWedge.h
  enums.h
  euler_kokkos_build_info.h
  euler_kokkos_build_info.cpp
  euler_kokkos_git_info.h
  euler_kokkos_git_info.cpp
  HydroParams.cpp
  HydroParams.h
  HydroState.h
  kokkos_shared.h
  mhd_utils.h
  problems/BlastParams.h
  problems/BrioWuParams.h
  problems/FieldLoopParams.h
  problems/GreshoParams.h
  problems/ImplodeParams.h
  problems/initRiemannConfig2d.h
  problems/IsentropicVortexParams.h
  problems/JetParams.h
  problems/KHParams.h
  problems/OrszagTangParams.h
  problems/RayleighTaylorInstabilityParams.h
  problems/RotorParams.h
  problems/WedgeParams.h
  real_type.h
  RiemannSolvers.h
  RiemannSolvers_MHD.h
  SolverBase.cpp
  SolverBase.h
  solver_utils.h
  utils.cpp
  utils.h)

target_include_directories(euler_kokkos_shared PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
                                                      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>)

target_link_libraries(euler_kokkos_shared PUBLIC Kokkos::kokkos EulerKokkos::config EulerKokkos::io)
if(EULER_KOKKOS_USE_MPI)
  target_link_libraries(euler_kokkos_shared PUBLIC EulerKokkos::mpiUtils)
endif(EULER_KOKKOS_USE_MPI)

# Add an alias so that library can be used inside the build tree
add_library(EulerKokkos::shared ALIAS euler_kokkos_shared)

# ##################################################################################################
# SOLVER FACTORY ####
# ##################################################################################################
add_library(euler_kokkos_solver_factory SolverFactory.cpp SolverFactory.h)
target_include_directories(
  euler_kokkos_solver_factory PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/muscl
                                     ${CMAKE_BINARY_DIR}/src)

target_link_libraries(euler_kokkos_solver_factory PUBLIC Kokkos::kokkos)
if(EULER_KOKKOS_USE_MPI)
  target_link_libraries(euler_kokkos_solver_factory PUBLIC EulerKokkos::mpiUtils)
endif(EULER_KOKKOS_USE_MPI)

# Add an alias so that library can be used inside the build tree
add_library(EulerKokkos::solver_factory ALIAS euler_kokkos_solver_factory)
