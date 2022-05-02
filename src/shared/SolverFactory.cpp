#include "shared/SolverFactory.h"

#include "shared/SolverBase.h"

#include "muscl/SolverHydroMuscl.h"
#include "muscl/SolverMHDMuscl.h"

namespace euler_kokkos {

// The main solver creation routine
SolverFactory::SolverFactory()
{

  /*
   * Register some possible solvers
   */
  registerSolver("Hydro_Muscl_2D", &muscl::SolverHydroMuscl<2>::create);
  registerSolver("Hydro_Muscl_3D", &muscl::SolverHydroMuscl<3>::create);

  registerSolver("MHD_Muscl_2D",   &muscl::SolverMHDMuscl<2>::create);
  registerSolver("MHD_Muscl_3D",   &muscl::SolverMHDMuscl<3>::create);

} // SolverFactory::SolverFactory

} // namespace euler_kokkos
