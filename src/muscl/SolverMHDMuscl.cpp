#include <string>
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "muscl/SolverMHDMuscl.h"
#include "shared/HydroParams.h"

namespace euler_kokkos
{
namespace muscl
{

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<2>::make_boundaries(DataArray Udata)
{

  bool mhd_enabled = true;

#ifdef USE_MPI

  make_boundaries_mpi(Udata, mhd_enabled);

#else

  make_boundaries_serial(Udata, mhd_enabled);

#endif // USE_MPI

} // SolverMHDMuscl<2>::make_boundaries

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<3>::make_boundaries(DataArray Udata)
{

  bool mhd_enabled = true;

#ifdef USE_MPI

  make_boundaries_mpi(Udata, mhd_enabled);

#else

  make_boundaries_serial(Udata, mhd_enabled);

#endif // USE_MPI

} // SolverMHDMuscl<3>::make_boundaries

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Compute electric field
// ///////////////////////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<2>::computeElectricField(DataArray Udata)
{

  // call device functor
  // ComputeElecFieldFunctor2D::apply(params, Udata, Q, ElecField);

} // SolverMHDMuscl<2>::computeElectricField

template <>
void
SolverMHDMuscl<3>::computeElectricField(DataArray Udata)
{

  // call device functor
  ComputeElecFieldFunctor3D::apply(params, Udata, Q, ElecField);

} // SolverMHDMuscl<3>::computeElectricField


// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Compute magnetic slopes
// ///////////////////////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<3>::computeMagSlopes(DataArray Udata)
{

  // call device functor
  ComputeMagSlopesFunctor3D::apply(params, Udata, DeltaA, DeltaB, DeltaC);

} // SolverMHDMuscl3D::computeMagSlopes

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Compute trace (only used in implementation version 2), i.e.
// fill global array qm_x, qmy, qp_x, qp_y
// ///////////////////////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<2>::computeTrace(DataArray Udata, real_t dt)
{

  // local variables
  real_t dtdx;
  real_t dtdy;

  dtdx = dt / params.dx;
  dtdy = dt / params.dy;

  // call device functor
  ComputeTraceFunctor2D_MHD::apply(
    params, Udata, Q, Qm_x, Qm_y, Qp_x, Qp_y, QEdge_RT, QEdge_RB, QEdge_LT, QEdge_LB, dtdx, dtdy);

} // SolverMHDMuscl<2>::computeTrace

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Compute trace (only used in implementation version 2), i.e.
// fill global array qm_x, qmy, qp_x, qp_y
// ///////////////////////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<3>::computeTrace(DataArray Udata, real_t dt)
{

  // local variables
  real_t dtdx;
  real_t dtdy;
  real_t dtdz;

  dtdx = dt / params.dx;
  dtdy = dt / params.dy;
  dtdz = dt / params.dz;

  // call device functor
  ComputeTraceFunctor3D_MHD::apply(params,
                                   Udata,
                                   Q,
                                   DeltaA,
                                   DeltaB,
                                   DeltaC,
                                   ElecField,
                                   Qm_x,
                                   Qm_y,
                                   Qm_z,
                                   Qp_x,
                                   Qp_y,
                                   Qp_z,
                                   QEdge_RT,
                                   QEdge_RB,
                                   QEdge_LT,
                                   QEdge_LB,
                                   QEdge_RT2,
                                   QEdge_RB2,
                                   QEdge_LT2,
                                   QEdge_LB2,
                                   QEdge_RT3,
                                   QEdge_RB3,
                                   QEdge_LT3,
                                   QEdge_LB3,
                                   dtdx,
                                   dtdy,
                                   dtdz);

} // SolverMHDMuscl<3>::computeTrace

// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////////////
// Compute flux via Riemann solver and store
// //////////////////////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<2>::computeFluxesAndStore(real_t dt)
{

  real_t dtdx = dt / params.dx;
  real_t dtdy = dt / params.dy;

  // call device functor
  ComputeFluxesAndStoreFunctor2D_MHD::apply(
    params, Qm_x, Qm_y, Qp_x, Qp_y, Fluxes_x, Fluxes_y, dtdx, dtdy);

} // SolverMHDMuscl<2>::computeFluxesAndStore

// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////////////
// Compute flux via Riemann solver and store
// //////////////////////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<3>::computeFluxesAndStore(real_t dt)
{

  real_t dtdx = dt / params.dx;
  real_t dtdy = dt / params.dy;
  real_t dtdz = dt / params.dz;

  // call device functor
  ComputeFluxesAndStoreFunctor3D_MHD::apply(
    params, Qm_x, Qm_y, Qm_z, Qp_x, Qp_y, Qp_z, Fluxes_x, Fluxes_y, Fluxes_z, dtdx, dtdy, dtdz);

} // SolverMHDMuscl<3>::computeFluxesAndStore

// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////////////
// Compute flux via Riemann solver and update
// //////////////////////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<2>::computeFluxesAndUpdate(real_t dt, DataArray Udata)
{

  real_t dtdx = dt / params.dx;
  real_t dtdy = dt / params.dy;

  // call device functor
  ComputeFluxesAndUpdateFunctor2D_MHD::apply(params, Qm_x, Qm_y, Qp_x, Qp_y, Udata, dtdx, dtdy);

} // SolverMHDMuscl<2>::computeFluxesAndUpdate

// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////////////
// Compute flux via Riemann solver and update
// //////////////////////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<3>::computeFluxesAndUpdate(real_t dt, DataArray Udata)
{

  real_t dtdx = dt / params.dx;
  real_t dtdy = dt / params.dy;
  real_t dtdz = dt / params.dz;

  // call device functor
  ComputeFluxesAndUpdateFunctor3D_MHD::apply(
    params, Qm_x, Qm_y, Qm_z, Qp_x, Qp_y, Qp_z, Udata, dtdx, dtdy, dtdz);

} // SolverMHDMuscl<3>::computeFluxesAndUpdate

// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////////////
// Compute EMF via 2D Riemann solver and store
// //////////////////////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<2>::computeEmfAndStore(real_t dt)
{

  real_t dtdx = dt / params.dx;
  real_t dtdy = dt / params.dy;

  // call device functor
  ComputeEmfAndStoreFunctor2D::apply(
    params, QEdge_RT, QEdge_RB, QEdge_LT, QEdge_LB, Emf1, dtdx, dtdy);

} // SolverMHSMuscl<2>::computeEmfAndStore

// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////////////
// Compute EMF via 2D Riemann solver and store
// //////////////////////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<3>::computeEmfAndStore(real_t dt)
{

  real_t dtdx = dt / params.dx;
  real_t dtdy = dt / params.dy;
  real_t dtdz = dt / params.dz;

  // call device functor
  ComputeEmfAndStoreFunctor3D::apply(params,
                                     QEdge_RT,
                                     QEdge_RB,
                                     QEdge_LT,
                                     QEdge_LB,
                                     QEdge_RT2,
                                     QEdge_RB2,
                                     QEdge_LT2,
                                     QEdge_LB2,
                                     QEdge_RT3,
                                     QEdge_RB3,
                                     QEdge_LT3,
                                     QEdge_LB3,
                                     Emf,
                                     dtdx,
                                     dtdy,
                                     dtdz);

} // SolverMHDMuscl<3>::computeEmfAndStore

// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////////////
// Compute EMF via 2D Riemann solver and update magnetic field
// //////////////////////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<2>::computeEmfAndUpdate(real_t dt, DataArray Udata)
{

  real_t dtdx = dt / params.dx;
  real_t dtdy = dt / params.dy;

  // call device functor
  ComputeEmfAndUpdateFunctor2D::apply(
    params, QEdge_RT, QEdge_RB, QEdge_LT, QEdge_LB, Udata, dtdx, dtdy);

} // SolverMHSMuscl<2>::computeEmfAndUpdate

// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////////////
// Compute EMF via 2D Riemann solver and update magnetic field
// //////////////////////////////////////////////////////////////////
template <>
void
SolverMHDMuscl<3>::computeEmfAndUpdate(real_t dt, DataArray Udata)
{

  real_t dtdx = dt / params.dx;
  real_t dtdy = dt / params.dy;
  real_t dtdz = dt / params.dz;

  // call device functor
  ComputeEmfAndUpdateFunctor3D::apply(params,
                                      QEdge_RT,
                                      QEdge_RB,
                                      QEdge_LT,
                                      QEdge_LB,
                                      QEdge_RT2,
                                      QEdge_RB2,
                                      QEdge_LT2,
                                      QEdge_LB2,
                                      QEdge_RT3,
                                      QEdge_RB3,
                                      QEdge_LT3,
                                      QEdge_LB3,
                                      Udata,
                                      dtdx,
                                      dtdy,
                                      dtdz);

} // SolverMHSMuscl<2>::computeEmfAndUpdate

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual computation of Godunov scheme - 2d
// ///////////////////////////////////////////
template <>
void
SolverMHDMuscl<2>::godunov_unsplit_impl(DataArray data_in, DataArray data_out, real_t dt)
{

  real_t dtdx;
  real_t dtdy;

  dtdx = dt / params.dx;
  dtdy = dt / params.dy;

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

    // trace computation: fill arrays qm_x, qm_y, qp_x, qp_y
    computeTrace(data_in, dt);

    // Compute flux via Riemann solver and update (time integration)
    computeFluxesAndStore(dt);

    // Compute Emf
    computeEmfAndStore(dt);

    // actual update with fluxes
    UpdateFunctor2D_MHD::apply(params, data_out, Fluxes_x, Fluxes_y, dtdx, dtdy);

    // actual update with emf
    UpdateEmfFunctor2D::apply(params, data_out, Emf1, dtdx, dtdy);
  }
  else if (params.implementationVersion == 1)
  {
    // trace computation: fill arrays qm_x, qm_y, qp_x, qp_y
    computeTrace(data_in, dt);

    // Compute flux via Riemann solver and update (time integration)
    computeFluxesAndUpdate(dt, data_out);

    // Compute Emf and update magnetic field
    computeEmfAndUpdate(dt, data_out);
  }
  else if (params.implementationVersion == 2)
  {
    // compute limited slopes (for reconstruction)
    ComputeSlopesFunctor2D_MHD::apply(params, data_in, Q, Slopes_x, Slopes_y);

    // update (cell-centered) primitive variables, perform 1/2 time step
    ComputeUpdatedPrimVarFunctor2D_MHD::apply(params, data_in, Q, Q2, dtdx, dtdy);

    // compute electric field (v wedge B)
    ComputeElecFieldFunctor2D::apply(params, data_in, Q, ElecField);

    // update at t_{n+1} with hydro flux (across all faces)
    ComputeFluxAndUpdateAlongDirFunctor2D_MHD<DIR_X>::apply(
      params, data_in, data_out, Q, Q2, dtdx, dtdy);

    ComputeFluxAndUpdateAlongDirFunctor2D_MHD<DIR_Y>::apply(
      params, data_in, data_out, Q, Q2, dtdx, dtdy);

    ReconstructEdgeComputeEmfAndUpdateFunctor2D::apply(
      params, data_in, data_out, Q, Q2, Slopes_x, Slopes_y, ElecField, dtdx, dtdy);
  }

  timers[TIMER_NUM_SCHEME]->stop();

} // SolverMHDMuscl2D::godunov_unsplit_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual computation of Godunov scheme - 3d
// ///////////////////////////////////////////
template <>
void
SolverMHDMuscl<3>::godunov_unsplit_impl(DataArray data_in, DataArray data_out, real_t dt)
{

  real_t dtdx;
  real_t dtdy;
  real_t dtdz;

  dtdx = dt / params.dx;
  dtdy = dt / params.dy;
  dtdz = dt / params.dz;

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

    // compute electric field
    computeElectricField(data_in);

    // compute magnetic slopes
    computeMagSlopes(data_in);

    // trace computation: fill arrays qm_x, qm_y, qm_z, qp_x, qp_y, qp_z
    computeTrace(data_in, dt);

    // Compute flux via Riemann solver and update (time integration)
    computeFluxesAndStore(dt);

    // Compute Emf
    computeEmfAndStore(dt);

    // actual update with fluxes
    UpdateFunctor3D_MHD::apply(params, data_out, Fluxes_x, Fluxes_y, Fluxes_z, dtdx, dtdy, dtdz);

    // actual update with emf
    UpdateEmfFunctor3D::apply(params, data_out, Emf, dtdx, dtdy, dtdz);
  }
  else if (params.implementationVersion == 1)
  {
    // compute electric field
    computeElectricField(data_in);

    // compute magnetic slopes
    computeMagSlopes(data_in);

    // trace computation: fill arrays qm_x, qm_y, qm_z, qp_x, qp_y, qp_z
    computeTrace(data_in, dt);

    // Compute flux via Riemann solver and update (time integration)
    computeFluxesAndUpdate(dt, data_out);

    // Compute Emf and update magnetic field
    computeEmfAndUpdate(dt, data_out);
  }
  else if (params.implementationVersion == 2)
  {
    // TODO
  }
  timers[TIMER_NUM_SCHEME]->stop();

} // SolverMHDMuscl<3>::godunov_unsplit_impl

} // namespace muscl

} // namespace euler_kokkos
