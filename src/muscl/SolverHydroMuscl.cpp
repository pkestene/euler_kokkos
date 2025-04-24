#include <string>
#include <cstdio>
#include <cstdbool>

#include "muscl/SolverHydroMuscl.h"
#include "shared/HydroParams.h"

namespace euler_kokkos
{
namespace muscl
{

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbent, reflexive or periodic
// //////////////////////////////////////////////////
template <>
void
SolverHydroMuscl<2>::make_boundaries(DataArray Udata)
{

  bool mhd_enabled = false;

#ifdef EULER_KOKKOS_USE_MPI

  make_boundaries_mpi(Udata, mhd_enabled);

#else

  make_boundaries_serial(Udata, mhd_enabled);

#endif // EULER_KOKKOS_USE_MPI

} // SolverHydroMuscl<2>::make_boundaries

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbent, reflexive or periodic
// //////////////////////////////////////////////////
template <>
void
SolverHydroMuscl<3>::make_boundaries(DataArray Udata)
{

  bool mhd_enabled = false;

#ifdef EULER_KOKKOS_USE_MPI

  make_boundaries_mpi(Udata, mhd_enabled);

#else

  make_boundaries_serial(Udata, mhd_enabled);

#endif // EULER_KOKKOS_USE_MPI

} // SolverHydroMuscl<3>::make_boundaries

// =======================================================
// =======================================================
/**
 */
template <>
void
SolverHydroMuscl<2>::init_four_quadrant(DataArray Udata)
{

  int    configNumber = configMap.getInteger("riemann2d", "config_number", 0);
  real_t xt = configMap.getFloat("riemann2d", "x", 0.8);
  real_t yt = configMap.getFloat("riemann2d", "y", 0.8);

  HydroState2d U0, U1, U2, U3;
  getRiemannConfig2d(configNumber, U0, U1, U2, U3);

  primToCons_2D(U0, params.settings.gamma0);
  primToCons_2D(U1, params.settings.gamma0);
  primToCons_2D(U2, params.settings.gamma0);
  primToCons_2D(U3, params.settings.gamma0);

  InitFourQuadrantFunctor2D::apply(params, Udata, configNumber, U0, U1, U2, U3, xt, yt);

} // SolverHydroMuscl<2>::init_four_quadrant

// =======================================================
// =======================================================
template <>
void
SolverHydroMuscl<2>::init_isentropic_vortex(DataArray Udata)
{

  IsentropicVortexParams iparams(configMap);

  InitIsentropicVortexFunctor2D::apply(params, iparams, Udata);

} // SolverHydroMuscl<2>::init_isentropic_vortex

// =======================================================
// =======================================================
template <>
void
SolverHydroMuscl<2>::init(DataArray Udata)
{

  // test if we are performing a re-start run (default : false)
  bool restartEnabled = configMap.getBool("run", "restart_enabled", false);

  if (restartEnabled)
  { // load data from input data file

    init_restart(Udata);
  }
  else
  { // regular initialization

    /*
     * initialize hydro array at t=0
     */
    if (!m_problem_name.compare("implode"))
    {

      init_implode(Udata);
    }
    else if (!m_problem_name.compare("blast"))
    {

      init_blast(Udata);
    }
    else if (!m_problem_name.compare("kelvin_helmholtz"))
    {

      init_kelvin_helmholtz(Udata);
    }
    else if (!m_problem_name.compare("gresho_vortex"))
    {

      init_gresho_vortex(Udata);
    }
    else if (!m_problem_name.compare("four_quadrant"))
    {

      init_four_quadrant(Udata);
    }
    else if (!m_problem_name.compare("isentropic_vortex"))
    {

      init_isentropic_vortex(Udata);
    }
    else if (!m_problem_name.compare("rayleigh_taylor"))
    {

      init_rayleigh_taylor(Udata, gravity_field);
    }
    else
    {

      std::cout << "Problem : " << m_problem_name << " is not recognized / implemented."
                << std::endl;
      std::cout << "Use default - implode" << std::endl;
      init_implode(Udata);
    }

  } // end regular initialization

} // SolverHydroMuscl::init / 2d

// =======================================================
// =======================================================
template <>
void
SolverHydroMuscl<3>::init(DataArray Udata)
{

  // test if we are performing a re-start run (default : false)
  bool restartEnabled = configMap.getBool("run", "restart_enabled", false);

  if (restartEnabled)
  { // load data from input data file

    init_restart(Udata);
  }
  else
  { // regular initialization

    /*
     * initialize hydro array at t=0
     */
    if (!m_problem_name.compare("implode"))
    {

      init_implode(Udata);
    }
    else if (!m_problem_name.compare("blast"))
    {

      init_blast(Udata);
    }
    else if (!m_problem_name.compare("kelvin_helmholtz"))
    {

      init_kelvin_helmholtz(Udata);
    }
    else if (!m_problem_name.compare("gresho_vortex"))
    {

      init_gresho_vortex(Udata);
    }
    else if (!m_problem_name.compare("rayleigh_taylor"))
    {

      init_rayleigh_taylor(Udata, gravity_field);
    }
    else
    {

      std::cout << "Problem : " << m_problem_name << " is not recognized / implemented."
                << std::endl;
      std::cout << "Use default - implode" << std::endl;
      init_implode(Udata);
    }

  } // end regular initialization

} // SolverHydroMuscl<3>::init

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual computation of Godunov scheme - 2d
// ///////////////////////////////////////////
template <>
void
SolverHydroMuscl<2>::godunov_unsplit_impl(DataArray data_in, DataArray data_out, real_t dt)
{

  // fill ghost cell in data_in
  timers[TIMER_BOUNDARIES]->start();
  make_boundaries(data_in);
  timers[TIMER_BOUNDARIES]->stop();

  // copy data_in into data_out (not necessary)
  // data_out = data_in;
  Kokkos::deep_copy(data_out, data_in);

  // start main computation
  timers[TIMER_NUM_SCHEME]->start();

  // convert conservative variable into primitives ones for the entire domain
  convertToPrimitives(data_in);

  if (params.implementationVersion == 0)
  {

    // compute fluxes (if gravity_enabled is false, the last parameter is not used)
    ComputeAndStoreFluxesFunctor2D::apply(
      params, Q, Fluxes_x, Fluxes_y, dt, m_gravity, gravity_field);

    // actual update
    UpdateFunctor2D::apply(params, data_out, Fluxes_x, Fluxes_y);

    // gravity source term
    if (m_gravity.enabled)
    {
      GravitySourceTermFunctor2D::apply(params, data_in, data_out, gravity_field, dt);
    }
  } // end params.implementationVersion == 0
  else if (params.implementationVersion == 1)
  {

    // call device functor to compute slopes
    ComputeSlopesFunctor2D::apply(params, Q, Slopes_x, Slopes_y);

    // now trace along X axis
    ComputeTraceAndFluxes_Functor2D<XDIR>::apply(
      params, Q, Slopes_x, Slopes_y, Fluxes_x, dt, m_gravity, gravity_field);

    // and update along X axis
    UpdateDirFunctor2D<XDIR>::apply(params, data_out, Fluxes_x);

    // now trace along Y axis
    ComputeTraceAndFluxes_Functor2D<YDIR>::apply(
      params, Q, Slopes_x, Slopes_y, Fluxes_y, dt, m_gravity, gravity_field);

    // and update along Y axis
    UpdateDirFunctor2D<YDIR>::apply(params, data_out, Fluxes_y);

    // gravity source term
    if (m_gravity.enabled)
    {
      GravitySourceTermFunctor2D::apply(params, data_in, data_out, gravity_field, dt);
    }

  } // end params.implementationVersion == 1
  else if (params.implementationVersion == 2)
  {

    // compute fluxes and update
    ComputeAllFluxesAndUpdateFunctor2D::apply(params, Q, data_out, dt, m_gravity, gravity_field);

    // gravity source term
    if (m_gravity.enabled)
    {
      GravitySourceTermFunctor2D::apply(params, data_in, data_out, gravity_field, dt);
    }

  } // end params.implementationVersion == 2

  timers[TIMER_NUM_SCHEME]->stop();

} // SolverHydroMuscl2D::godunov_unsplit_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual computation of Godunov scheme - 3d
// ///////////////////////////////////////////
template <>
void
SolverHydroMuscl<3>::godunov_unsplit_impl(DataArray data_in, DataArray data_out, real_t dt)
{

  // fill ghost cell in data_in
  timers[TIMER_BOUNDARIES]->start();
  make_boundaries(data_in);
  timers[TIMER_BOUNDARIES]->stop();

  // copy data_in into data_out (not necessary)
  // data_out = data_in;
  Kokkos::deep_copy(data_out, data_in);

  // start main computation
  timers[TIMER_NUM_SCHEME]->start();

  // convert conservative variable into primitives ones for the entire domain
  convertToPrimitives(data_in);

  if (params.implementationVersion == 0)
  {

    // compute fluxes
    ComputeAndStoreFluxesFunctor3D::apply(
      params, Q, Fluxes_x, Fluxes_y, Fluxes_z, dt, m_gravity, gravity_field);

    // actual update
    UpdateFunctor3D::apply(params, data_out, Fluxes_x, Fluxes_y, Fluxes_z);

    // gravity source term
    if (m_gravity.enabled)
    {
      GravitySourceTermFunctor3D::apply(params, data_in, data_out, gravity_field, dt);
    }
  }
  else if (params.implementationVersion == 1)
  {

    // call device functor to compute slopes
    ComputeSlopesFunctor3D::apply(params, Q, Slopes_x, Slopes_y, Slopes_z);

    // now trace along X axis
    ComputeTraceAndFluxes_Functor3D<XDIR>::apply(
      params, Q, Slopes_x, Slopes_y, Slopes_z, Fluxes_x, dt, m_gravity, gravity_field);

    // and update along X axis
    UpdateDirFunctor3D<XDIR>::apply(params, data_out, Fluxes_x);

    // now trace along Y axis
    ComputeTraceAndFluxes_Functor3D<YDIR>::apply(
      params, Q, Slopes_x, Slopes_y, Slopes_z, Fluxes_y, dt, m_gravity, gravity_field);

    // and update along Y axis
    UpdateDirFunctor3D<YDIR>::apply(params, data_out, Fluxes_y);

    // now trace along Z axis
    ComputeTraceAndFluxes_Functor3D<ZDIR>::apply(
      params, Q, Slopes_x, Slopes_y, Slopes_z, Fluxes_z, dt, m_gravity, gravity_field);

    // and update along Z axis
    UpdateDirFunctor3D<ZDIR>::apply(params, data_out, Fluxes_z);

    // gravity source term
    if (m_gravity.enabled)
    {
      GravitySourceTermFunctor3D::apply(params, data_in, data_out, gravity_field, dt);
    }

  } // end params.implementationVersion == 1
  else if (params.implementationVersion == 2)
  {

    // compute fluxes and update
    ComputeAllFluxesAndUpdateFunctor3D::apply(params, Q, data_out, dt, m_gravity, gravity_field);

    // gravity source term
    if (m_gravity.enabled)
    {
      GravitySourceTermFunctor3D::apply(params, data_in, data_out, gravity_field, dt);
    }

  } // end params.implementationVersion == 2

  timers[TIMER_NUM_SCHEME]->stop();

} // SolverHydroMuscl<3>::godunov_unsplit_impl

} // namespace muscl

} // namespace euler_kokkos
