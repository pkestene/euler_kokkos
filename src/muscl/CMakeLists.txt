add_library(
  euler_kokkos_muscl
  hydro_shared.h
  HydroBaseFunctor2D.h
  HydroBaseFunctor3D.h
  HydroRunFunctors2D.h
  HydroRunFunctors3D.h
  HydroInitFunctors2D.h
  HydroInitFunctors3D.h
  MHDBaseFunctor2D.h
  MHDBaseFunctor3D.h
  MHDRunFunctors2D.h
  MHDRunFunctors3D.h
  MHDInitFunctors2D.h
  MHDInitFunctors3D.h
  SolverHydroMuscl.h
  SolverHydroMuscl.cpp
  SolverMHDMuscl.h
  SolverMHDMuscl.cpp)

target_include_directories(euler_kokkos_muscl PUBLIC ${CMAKE_SOURCE_DIR}/src
                                                     ${CMAKE_BINARY_DIR}/src)

target_link_libraries(euler_kokkos_muscl PUBLIC Kokkos::kokkos)

if(EULER_KOKKOS_USE_MPI)
  target_link_libraries(euler_kokkos_muscl PUBLIC EulerKokkos::mpiUtils)
endif(EULER_KOKKOS_USE_MPI)

# Add an alias so that library can be used inside the build tree
add_library(EulerKokkos::muscl ALIAS euler_kokkos_muscl)
