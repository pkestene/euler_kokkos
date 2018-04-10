#######################
### SHARED ############
#######################
set (shared_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/BlastParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/KHParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/WedgeParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/JetParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IsentropicVortexParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/RotorParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FieldLoopParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/initRiemannConfig2d.h
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

# if (USE_MPI)
#   set (shared_SRCS
#     ${shared_SRCS}
#     ${CMAKE_CURRENT_SOURCE_DIR}/shared/mpiBorderUtils.h
#     )
# endif()

add_library(shared
  ${shared_SRCS}
  )
target_include_directories(shared
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/utils
  ${CMAKE_SOURCE_DIR}/src/utils/time
  ${CMAKE_SOURCE_DIR}/src/utils/io
  ${CMAKE_SOURCE_DIR}/kokkos/core/src
  ${CMAKE_SOURCE_DIR}/kokkos/containers/src
  ${CMAKE_SOURCE_DIR}/kokkos/algorithms/src
  ${CMAKE_BINARY_DIR}/kokkos
  ) 
target_link_libraries(shared
  kokkos
  )

#######################
### SOLVER FACTORY ####
#######################
set (solver_factory_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverFactory.h
  )
add_library(solver_factory
  ${solver_factory_SRCS}
  )
target_include_directories(solver_factory
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/muscl
)
target_link_libraries(solver_factory
  kokkos
  )
