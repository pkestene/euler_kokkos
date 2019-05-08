add_library(muscl "")

target_sources (muscl
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hydro_shared.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroBaseFunctor2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroBaseFunctor3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunFunctors2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunFunctors3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroInitFunctors2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroInitFunctors3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDBaseFunctor2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDBaseFunctor3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDRunFunctors2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDRunFunctors3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDInitFunctors2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDInitFunctors3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMuscl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMuscl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverMHDMuscl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverMHDMuscl.cpp
  )

target_include_directories(muscl
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )

target_link_libraries(muscl
  PUBLIC
  kokkos
  )

if(USE_MPI)
  target_link_libraries(muscl PUBLIC EulerKokkos::mpiUtils)
endif(USE_MPI)

#Add an alias so that library can be used inside the build tree
add_library(EulerKokkos::muscl ALIAS muscl)
