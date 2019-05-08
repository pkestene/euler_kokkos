#######################
### SHARED ############
#######################
add_library(shared "")
target_sources (shared
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/problems/BlastParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/problems/ImplodeParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/problems/KHParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/problems/WedgeParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/problems/JetParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/problems/IsentropicVortexParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/problems/RotorParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/problems/FieldLoopParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/problems/initRiemannConfig2d.h
  ${CMAKE_CURRENT_SOURCE_DIR}/problems/GreshoParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundariesFunctors.h
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundariesFunctorsWedge.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroParams.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroState.h
  ${CMAKE_CURRENT_SOURCE_DIR}/kokkos_shared.h
  ${CMAKE_CURRENT_SOURCE_DIR}/real_type.h
  ${CMAKE_CURRENT_SOURCE_DIR}/enums.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverBase.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverBase.h
  ${CMAKE_CURRENT_SOURCE_DIR}/RiemannSolvers.h
  ${CMAKE_CURRENT_SOURCE_DIR}/RiemannSolvers_MHD.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mhd_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_utils.h
  )

target_link_libraries(shared
  PUBLIC
  kokkos
  EulerKokkos::config
  EulerKokkos::io
  )
if(USE_MPI)
  target_link_libraries(shared
  PUBLIC
  EulerKokkos::mpiUtils)
endif(USE_MPI)

#Add an alias so that library can be used inside the build tree
add_library(EulerKokkos::shared ALIAS shared)

#######################
### SOLVER FACTORY ####
#######################
add_library(solver_factory "")
target_sources (solver_factory
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverFactory.h
  )
target_include_directories(solver_factory
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/muscl
)
target_link_libraries(solver_factory
  PUBLIC
  kokkos
  )
if(USE_MPI)
  target_link_libraries(solver_factory
  PUBLIC
  EulerKokkos::mpiUtils)
endif(USE_MPI)

#Add an alias so that library can be used inside the build tree
add_library(EulerKokkos::solver_factory ALIAS solver_factory)
