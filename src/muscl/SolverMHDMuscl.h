/**
 * Class SolverMHDMuscl implementation.
 *
 * Main class for solving MHD (Euler) with MUSCL-Hancock scheme for 2D/3D.
 */
#ifndef SOLVER_MHD_MUSCL_H_
#define SOLVER_MHD_MUSCL_H_

#include <string>
#include <cstdio>
#include <cstdbool>
#include <iostream>

// shared
#include "shared/SolverBase.h"
#include "shared/HydroParams.h"
#include "shared/kokkos_shared.h"

// the actual computational functors called in HydroRun
#include "muscl/MHDRunFunctors2D.h"
#include "muscl/MHDRunFunctors3D.h"

// Init conditions functors
#include "muscl/MHDInitFunctors2D.h"
#include "muscl/MHDInitFunctors3D.h"

// for IO
#include <utils/io/IO_ReadWrite.h>

// for init condition
#include "shared/problems/BlastParams.h"
#include "shared/problems/ImplodeParams.h"
#include "shared/problems/RotorParams.h"

namespace euler_kokkos
{
namespace muscl
{

/**
 * Main magnehydrodynamics data structure for 2D/3D MUSCL-Hancock scheme.
 */
template <int dim>
class SolverMHDMuscl : public euler_kokkos::SolverBase
{

public:
  //! Decide at compile-time which data array to use for 2d or 3d
  using DataArray = typename std::conditional<dim == 2, DataArray2d, DataArray3d>::type;

  //! Data array typedef for host memory space
  using DataArrayHost = typename std::conditional<dim == 2, DataArray2dHost, DataArray3dHost>::type;

  SolverMHDMuscl(HydroParams & params, ConfigMap & configMap);
  virtual ~SolverMHDMuscl();

  /**
   * Static creation method called by the solver factory.
   */
  static SolverBase *
  create(HydroParams & params, ConfigMap & configMap)
  {
    SolverMHDMuscl<dim> * solver = new SolverMHDMuscl<dim>(params, configMap);

    return solver;
  }

  DataArray     U;     //!< hydrodynamics conservative variables array
  DataArrayHost Uhost; //!< U mirror on host memory space
  DataArray     U2;    //!< hydrodynamics conservative variables array
  DataArray     Q;     //!< hydrodynamics primitive    variables array
  DataArray Q2; //!< hydrodynamics primitive    variables array at t_{n+1/2} (half time step update)

  /// source term using to compute face centered magnetic field component at t_{n+1/2}
  DataArray sFaceMag;

  DataArray Slopes_x; //!< implementation 2 only
  DataArray Slopes_y; //!< implementation 2 only
  DataArray Slopes_z; //!< implementation 2 only

  DataArray Qm_x; //!< hydrodynamics Riemann states array implementation 2
  DataArray Qm_y; //!< hydrodynamics Riemann states array
  DataArray Qm_z; //!< hydrodynamics Riemann states array

  DataArray Qp_x; //!< hydrodynamics Riemann states array
  DataArray Qp_y; //!< hydrodynamics Riemann states array
  DataArray Qp_z; //!< hydrodynamics Riemann states array

  DataArray QEdge_RT;
  DataArray QEdge_RB;
  DataArray QEdge_LT;
  DataArray QEdge_LB;

  DataArray QEdge_RT2;
  DataArray QEdge_RB2;
  DataArray QEdge_LT2;
  DataArray QEdge_LB2;

  DataArray QEdge_RT3;
  DataArray QEdge_RB3;
  DataArray QEdge_LT3;
  DataArray QEdge_LB3;

  DataArray Fluxes_x;
  DataArray Fluxes_y;
  DataArray Fluxes_z;

  // electromotive forces
  DataArrayScalar  Emf1; // 2d
  DataArrayVector3 Emf;  // 3d

  DataArray ElecField; // 2 or 3 components

  DataArrayVector3 DeltaA;
  DataArrayVector3 DeltaB;
  DataArrayVector3 DeltaC;

  /*
   * methods
   */

  // fill boundaries / ghost 2d / 3d
  void
  make_boundaries(DataArray Udata);

  // host routines (initialization)
  void
  init_blast(DataArray Udata);
  void
  init_implode(DataArray Udata);
  void
  init_brio_wu(DataArray Udata);
  void
  init_orszag_tang(DataArray Udata);
  void
  init_kelvin_helmholtz(DataArray Udata); // 2d and 3d
  void
  init_rotor(DataArray Udata);
  void
  init_field_loop(DataArray Udata);

  //! init restart (load data from file)
  void
  init_restart(DataArray Udata);

  //! init wrapper (actual initialization)
  void
  init(DataArray Udata);

  //! compute time step inside an MPI process, at shared memory level.
  double
  compute_dt_local();

  //! perform 1 time step (time integration).
  void
  next_iteration_impl();

  //! numerical scheme
  void
  godunov_unsplit();

  void
  godunov_unsplit_impl(DataArray data_in, DataArray data_out);

  void
  convertToPrimitives(DataArray Udata);

  void
  computeElectricField(DataArray Udata);

  void
  computeMagSlopes(DataArray Udata);

  void
  computeTrace(DataArray Udata, real_t dt);

  void
  computeFluxesAndStore(real_t dt);

  void
  computeFluxesAndUpdate(real_t dt, DataArray Udata);

  void
  computeEmfAndStore(real_t dt);

  void
  computeEmfAndUpdate(real_t dt, DataArray Udata);

  // output
  void
  save_solution_impl();

  int isize, jsize, ksize;

}; // class SolverMHDMuscl

// =======================================================
// ==== CLASS SolverMHDMuscl IMPL ========================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
template <int dim>
SolverMHDMuscl<dim>::SolverMHDMuscl(HydroParams & params, ConfigMap & configMap)
  : SolverBase(params, configMap)
  , U()
  , U2()
  , Q()
  , Q2()
  , sFaceMag()
  , Slopes_x()
  , Slopes_y()
  , Slopes_z()
  , Qm_x()
  , Qm_y()
  , Qm_z()
  , Qp_x()
  , Qp_y()
  , Qp_z()
  , QEdge_RT()
  , QEdge_RB()
  , QEdge_LT()
  , QEdge_LB()
  , QEdge_RT2()
  , QEdge_RB2()
  , QEdge_LT2()
  , QEdge_LB2()
  , QEdge_RT3()
  , QEdge_RB3()
  , QEdge_LT3()
  , QEdge_LB3()
  , Fluxes_x()
  , Fluxes_y()
  , Fluxes_z()
  , Emf1()
  , Emf()
  , ElecField()
  , DeltaA()
  , DeltaB()
  , DeltaC()
  , isize(params.isize)
  , jsize(params.jsize)
  , ksize(params.ksize)
{

  solver_type = SOLVER_MUSCL_HANCOCK;

  m_nCells = dim == 2 ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;
  m_nDofsPerCell = 1;

  int nbvar = params.nbvar;

  long long int total_mem_size = 0;

  /*
   * memory allocation (use sizes with ghosts included).
   *
   * Note that Uhost is not just a view to U, Uhost will be used
   * to save data from multiple other device array.
   * That's why we didn't use create_mirror_view to initialize Uhost.
   */
  if (dim == 2)
  {

    U = DataArray("U", isize, jsize, nbvar);
    Uhost = Kokkos::create_mirror(U);
    U2 = DataArray("U2", isize, jsize, nbvar);
    Q = DataArray("Q", isize, jsize, nbvar);

    total_mem_size += isize * jsize * nbvar * sizeof(real_t) * 3; // 1+1+1 for U+U2+Q

    if (params.implementationVersion == 0 or params.implementationVersion == 1)
    {

      Qm_x = DataArray("Qm_x", isize, jsize, nbvar);
      Qm_y = DataArray("Qm_y", isize, jsize, nbvar);
      Qp_x = DataArray("Qp_x", isize, jsize, nbvar);
      Qp_y = DataArray("Qp_y", isize, jsize, nbvar);

      QEdge_RT = DataArray("QEdge_RT", isize, jsize, nbvar);
      QEdge_RB = DataArray("QEdge_RB", isize, jsize, nbvar);
      QEdge_LT = DataArray("QEdge_LT", isize, jsize, nbvar);
      QEdge_LB = DataArray("QEdge_LB", isize, jsize, nbvar);

      total_mem_size += isize * jsize * nbvar * sizeof(real_t) * 8;
    }
    else if (params.implementationVersion == 2)
    {
      Q2 = DataArray("Q2", isize, jsize, nbvar);
      sFaceMag = DataArray("sFaceMag", isize, jsize, 2);
      Slopes_x = DataArray("Slope_x", isize, jsize, nbvar);
      Slopes_y = DataArray("Slope_y", isize, jsize, nbvar);
      ElecField = DataArray("ElecField", isize, jsize, 1);

      total_mem_size += isize * jsize * nbvar * sizeof(real_t) * (1 + 2);
      total_mem_size += isize * jsize * 1 * sizeof(real_t) * 1;
      total_mem_size += isize * jsize * 2 * sizeof(real_t) * 1;
    }

    if (params.implementationVersion == 0)
    {
      Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
      Fluxes_y = DataArray("Fluxes_y", isize, jsize, nbvar);

      Emf1 = DataArrayScalar("Emf", isize, jsize);
      total_mem_size += isize * jsize * nbvar * sizeof(real_t) * 2;
      total_mem_size += isize * jsize * 1 * sizeof(real_t);
    }
  }
  else
  {

    U = DataArray("U", isize, jsize, ksize, nbvar);
    Uhost = Kokkos::create_mirror(U);
    U2 = DataArray("U2", isize, jsize, ksize, nbvar);
    Q = DataArray("Q", isize, jsize, ksize, nbvar);

    total_mem_size += isize * jsize * ksize * nbvar * sizeof(real_t) * 3; // 1+1+1=3 for U+U2+Q

    if (params.implementationVersion == 0 or params.implementationVersion == 1)
    {

      Qm_x = DataArray("Qm_x", isize, jsize, ksize, nbvar);
      Qm_y = DataArray("Qm_y", isize, jsize, ksize, nbvar);
      Qm_z = DataArray("Qm_z", isize, jsize, ksize, nbvar);

      Qp_x = DataArray("Qp_x", isize, jsize, ksize, nbvar);
      Qp_y = DataArray("Qp_y", isize, jsize, ksize, nbvar);
      Qp_z = DataArray("Qp_z", isize, jsize, ksize, nbvar);

      QEdge_RT = DataArray("QEdge_RT", isize, jsize, ksize, nbvar);
      QEdge_RB = DataArray("QEdge_RB", isize, jsize, ksize, nbvar);
      QEdge_LT = DataArray("QEdge_LT", isize, jsize, ksize, nbvar);
      QEdge_LB = DataArray("QEdge_LB", isize, jsize, ksize, nbvar);

      QEdge_RT2 = DataArray("QEdge_RT2", isize, jsize, ksize, nbvar);
      QEdge_RB2 = DataArray("QEdge_RB2", isize, jsize, ksize, nbvar);
      QEdge_LT2 = DataArray("QEdge_LT2", isize, jsize, ksize, nbvar);
      QEdge_LB2 = DataArray("QEdge_LB2", isize, jsize, ksize, nbvar);

      QEdge_RT3 = DataArray("QEdge_RT3", isize, jsize, ksize, nbvar);
      QEdge_RB3 = DataArray("QEdge_RB3", isize, jsize, ksize, nbvar);
      QEdge_LT3 = DataArray("QEdge_LT3", isize, jsize, ksize, nbvar);
      QEdge_LB3 = DataArray("QEdge_LB3", isize, jsize, ksize, nbvar);

      ElecField = DataArray("ElecField", isize, jsize, ksize, 3);

      DeltaA = DataArrayVector3("DeltaA", isize, jsize, ksize);
      DeltaB = DataArrayVector3("DeltaB", isize, jsize, ksize);
      DeltaC = DataArrayVector3("DeltaC", isize, jsize, ksize);

      total_mem_size += isize * jsize * ksize * nbvar * sizeof(real_t) * 18 +
                        isize * jsize * ksize * 3 * sizeof(real_t) * 4;
    }
    else if (params.implementationVersion == 2)
    {
      Q2 = DataArray("Q2", isize, jsize, ksize, nbvar);
      sFaceMag = DataArray("sFaceMag", isize, jsize, ksize, 3);
      Slopes_x = DataArray("Slope_x", isize, jsize, ksize, nbvar);
      Slopes_y = DataArray("Slope_y", isize, jsize, ksize, nbvar);
      Slopes_z = DataArray("Slope_z", isize, jsize, ksize, nbvar);
      ElecField = DataArray("ElecField", isize, jsize, ksize, 3);

      total_mem_size += isize * jsize * ksize * nbvar * sizeof(real_t) * (1 + 3) +
                        isize * jsize * ksize * 3 * sizeof(real_t) * 2;
    }

    if (params.implementationVersion == 0)
    {
      Fluxes_x = DataArray("Fluxes_x", isize, jsize, ksize, nbvar);
      Fluxes_y = DataArray("Fluxes_y", isize, jsize, ksize, nbvar);
      Fluxes_z = DataArray("Fluxes_z", isize, jsize, ksize, nbvar);

      Emf = DataArrayVector3("Emf", isize, jsize, ksize);

      total_mem_size += isize * jsize * ksize * nbvar * sizeof(real_t) * 3 +
                        isize * jsize * ksize * 3 * sizeof(real_t) * 1;
    }
  } // dim == 2 / 3

  // perform init condition
  init(U);

  // initialize boundaries
  make_boundaries(U);

  // copy U into U2
  Kokkos::deep_copy(U2, U);

  // primitive variables are necessary for computing time step
  convertToPrimitives(U);

  // compute initialize time step
  compute_dt();

  int myRank = 0;
#ifdef USE_MPI
  myRank = params.myRank;
#endif // USE_MPI

  if (myRank == 0)
  {
    std::cout << "##########################" << "\n";
    std::cout << "Solver is " << m_solver_name << "\n";
    std::cout << "Problem (init condition) is " << m_problem_name << "\n";
    std::cout << "##########################" << "\n";

    // print parameters on screen
    params.print();
    std::cout << "##########################" << "\n";
    std::cout << "Memory requested : " << (total_mem_size / 1e6) << " MBytes\n";
    std::cout << "##########################" << "\n";
  }

} // SolverMHDMuscl::SolverMHDMuscl

// =======================================================
// =======================================================
/**
 *
 */
template <int dim>
SolverMHDMuscl<dim>::~SolverMHDMuscl()
{} // SolverMHDMuscl::~SolverMHDMuscl

// =======================================================
// =======================================================
template <int dim>
void
SolverMHDMuscl<dim>::make_boundaries(DataArray Udata)
{

  // this routine is specialized for 2d / 3d

} // SolverMHDMuscl<dim>::make_boundaries

template <>
void
SolverMHDMuscl<2>::make_boundaries(DataArray Udata);

template <>
void
SolverMHDMuscl<3>::make_boundaries(DataArray Udata);

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
template <int dim>
void
SolverMHDMuscl<dim>::init_blast(DataArray Udata)
{

  BlastParams blastParams = BlastParams(configMap);

  // alias to actual device functor
  using InitBlastFunctor =
    typename std::conditional<dim == 2, InitBlastFunctor2D_MHD, InitBlastFunctor3D_MHD>::type;

  // perform init
  InitBlastFunctor::apply(params, blastParams, Udata);

} // SolverMHDMuscl::init_blast

// =======================================================
// =======================================================
/**
 * Orszag-Tang vortex test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html
 */
template <int dim>
void
SolverMHDMuscl<dim>::init_orszag_tang(DataArray Udata)
{

  OrszagTangParams otParams = OrszagTangParams(configMap);

  // alias to actual device functor
  using InitOrszagTangFunctor =
    typename std::conditional<dim == 2, InitOrszagTangFunctor2D, InitOrszagTangFunctor3D>::type;

  InitOrszagTangFunctor::apply(params, otParams, Udata);

} // init_orszag_tang

// =======================================================
// =======================================================
/**
 * Hydrodynamical Kelvin-Helmholtz instability Test.
 *
 * see https://www.astro.princeton.edu/~jstone/Athena/tests/kh/kh.html
 *
 * See also article by Robertson et al:
 * "Computational Eulerian hydrodynamics and Galilean invariance",
 * B.E. Robertson et al, Mon. Not. R. Astron. Soc., 401, 2463-2476, (2010).
 *
 */
template <int dim>
void
SolverMHDMuscl<dim>::init_kelvin_helmholtz(DataArray Udata)
{

  KHParams khParams = KHParams(configMap);

  // alias to actual device functor
  using InitKelvinHelmholtzFunctor = typename std::
    conditional<dim == 2, InitKelvinHelmholtzFunctor2D_MHD, InitKelvinHelmholtzFunctor3D_MHD>::type;

  // perform init
  InitKelvinHelmholtzFunctor::apply(params, khParams, Udata);

} // init_kelvin_helmholtz

// =======================================================
// =======================================================
/**
 * Implosion test.
 *
 */
template <int dim>
void
SolverMHDMuscl<dim>::init_implode(DataArray Udata)
{

  ImplodeParams implodeParams = ImplodeParams(configMap);

  // alias to actual device functor
  using InitImplodeFunctor =
    typename std::conditional<dim == 2, InitImplodeFunctor2D_MHD, InitImplodeFunctor3D_MHD>::type;

  // perform init
  InitImplodeFunctor::apply(params, implodeParams, Udata);

} // SolverMHDMuscl::init_implode

// =======================================================
// =======================================================
/**
 * Brio-We shock tube.
 *
 */
template <int dim>
void
SolverMHDMuscl<dim>::init_brio_wu(DataArray Udata)
{

  BrioWuParams bwParams = BrioWuParams(configMap);

  // alias to actual device functor
  using InitBrioWuFunctor =
    typename std::conditional<dim == 2, InitBrioWuFunctor2D_MHD, InitBrioWuFunctor3D_MHD>::type;

  // perform init
  InitBrioWuFunctor::apply(params, bwParams, Udata);

} // SolverMHDMuscl::init_brio_wu

// =======================================================
// =======================================================
/**
 * Rotor test.
 *
 */
template <int dim>
void
SolverMHDMuscl<dim>::init_rotor(DataArray Udata)
{

  RotorParams rotorParams = RotorParams(configMap);

  // alias to actual device functor
  using InitRotorFunctor =
    typename std::conditional<dim == 2, InitRotorFunctor2D_MHD, InitRotorFunctor3D_MHD>::type;

  // perform init
  InitRotorFunctor::apply(params, rotorParams, Udata);

} // SolverMHDMuscl::init_rotor

// =======================================================
// =======================================================
/**
 * Field loop test.
 *
 */
template <int dim>
void
SolverMHDMuscl<dim>::init_field_loop(DataArray Udata)
{

  FieldLoopParams flParams = FieldLoopParams(configMap);

  // alias to actual device functor
  using InitFieldLoopFunctor = typename std::
    conditional<dim == 2, InitFieldLoopFunctor2D_MHD, InitFieldLoopFunctor3D_MHD>::type;

  // perform init
  InitFieldLoopFunctor::apply(params, flParams, Udata);

} // SolverMHDMuscl::init_field_loop

// =======================================================
// =======================================================
template <int dim>
void
SolverMHDMuscl<dim>::init_restart(DataArray Udata)
{

  int myRank = 0;
#ifdef USE_MPI
  myRank = params.myRank;
#endif // USE_MPI

  // load data
  auto reader = std::make_shared<io::IO_ReadWrite>(params, configMap, m_variables_names);

  // whether or not we are upscaling input data is handled inside "load_data"
  // m_times_saved are read from file
  reader->load_data(Udata, Uhost, m_times_saved, m_t);

  // increment to avoid overriding last output (?)
  // m_times_saved++;

  // do we force total time to be zero ?
  bool resetTotalTime = configMap.getBool("run", "restart_reset_totaltime", false);
  if (resetTotalTime)
    m_t = 0;

  if (myRank == 0)
  {
    std::cout << "### This is a restarted run ! Current time is " << m_t << " ###\n";
  }

} // SolverMHDMuscl<dim>::init_restart

// =======================================================
// =======================================================
template <int dim>
void
SolverMHDMuscl<dim>::init(DataArray Udata)
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
    if (!m_problem_name.compare("blast"))
    {

      init_blast(Udata);
    }
    else if (!m_problem_name.compare("implode"))
    {

      init_implode(U);
    }
    else if (!m_problem_name.compare("brio-wu"))
    {

      init_brio_wu(U);
    }
    else if (!m_problem_name.compare("orszag_tang"))
    {

      init_orszag_tang(U);
    }
    else if (!m_problem_name.compare("kelvin_helmholtz"))
    {

      init_kelvin_helmholtz(U);
    }
    else if (!m_problem_name.compare("rotor"))
    {

      init_rotor(U);
    }
    else if (!m_problem_name.compare("field_loop") || !m_problem_name.compare("field loop"))
    {

      init_field_loop(U);
    }
    else
    {

      std::cout << "Problem : " << m_problem_name << " is not recognized / implemented."
                << std::endl;
      std::cout << "Use default - Orszag-Tang vortex" << std::endl;
      m_problem_name = "orszag_tang";
      init_orszag_tang(Udata);
    }

  } // end regular initialization

} // SolverMHDMuscl::init / 2d / 3D

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
template <int dim>
double
SolverMHDMuscl<dim>::compute_dt_local()
{

  real_t dt;
  real_t invDt = ZERO_F;

  // alias to actual device functor
  using ComputeDtFunctor =
    typename std::conditional<dim == 2, ComputeDtFunctor2D_MHD, ComputeDtFunctor3D_MHD>::type;

  // call device functor
  ComputeDtFunctor::apply(params, Q, invDt);

  dt = params.settings.cfl / invDt;

  return dt;

} // SolverMHDMuscl::compute_dt_local

// =======================================================
// =======================================================
template <int dim>
void
SolverMHDMuscl<dim>::next_iteration_impl()
{

  int myRank = 0;

#ifdef USE_MPI
  myRank = params.myRank;
#endif // USE_MPI

  if (m_iteration % m_nlog == 0)
  {
    if (myRank == 0)
    {
      printf("time step=%7d (dt=% 10.8f t=% 10.8f)\n", m_iteration, m_dt, m_t);
    }
  }

  // output
  if (params.enableOutput)
  {
    if (should_save_solution())
    {

      if (myRank == 0)
      {
        std::cout << "Output results at time t=" << m_t << " step " << m_iteration << " dt=" << m_dt
                  << std::endl;
      }

      save_solution();

    } // end output
  } // end enable output

  // perform one step integration
  godunov_unsplit();

} // SolverMHDMuscl::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
template <int dim>
void
SolverMHDMuscl<dim>::godunov_unsplit()
{

  if (m_iteration % 2 == 0)
  {
    godunov_unsplit_impl(U, U2);
  }
  else
  {
    godunov_unsplit_impl(U2, U);
  }

} // SolverMHDMuscl::godunov_unsplit

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of Godunov scheme
// ///////////////////////////////////////////
template <int dim>
void
SolverMHDMuscl<dim>::godunov_unsplit_impl(DataArray data_in, DataArray data_out)
{

  // 2d / 3d implementation are specialized

} // SolverMHDMuscl<dim>::godunov_unsplit_impl

// 2d
template <>
void
SolverMHDMuscl<2>::godunov_unsplit_impl(DataArray data_in, DataArray data_out);

// 3d
template <>
void
SolverMHDMuscl<3>::godunov_unsplit_impl(DataArray data_in, DataArray data_out);


// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
template <int dim>
void
SolverMHDMuscl<dim>::convertToPrimitives(DataArray Udata)
{

  // alias to actual device functor
  using ConvertToPrimitivesFunctor = typename std::
    conditional<dim == 2, ConvertToPrimitivesFunctor2D_MHD, ConvertToPrimitivesFunctor3D_MHD>::type;

  // call device functor
  ConvertToPrimitivesFunctor::apply(params, Udata, Q);

} // SolverMHDMuscl::convertToPrimitives

// =======================================================
// =======================================================
template <int dim>
void
SolverMHDMuscl<dim>::computeElectricField(DataArray Udata)
{

  // 2d / 3d implementation are specialized

} // SolverMHDMuscl<dim>::computeElectricField

// 2d
template <>
void
SolverMHDMuscl<2>::computeElectricField(DataArray Udata);

// 3d
template <>
void
SolverMHDMuscl<3>::computeElectricField(DataArray Udata);

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Compute magnetic slopes
// ///////////////////////////////////////////////////////////////////
template <int dim>
void
SolverMHDMuscl<dim>::computeMagSlopes(DataArray Udata)
{

  // NA, 3D only

} // SolverMHDMuscl<dim>::computeMagSlopes

// 3d
template <>
void
SolverMHDMuscl<3>::computeMagSlopes(DataArray Udata);


// =======================================================
// =======================================================
template <int dim>
void
SolverMHDMuscl<dim>::save_solution_impl()
{

  timers[TIMER_IO]->start();
  if (m_iteration % 2 == 0)
    save_data(U, Uhost, m_times_saved, m_t);
  else
    save_data(U2, Uhost, m_times_saved, m_t);

  timers[TIMER_IO]->stop();

} // SolverMHDMuscl::save_solution_impl()

} // namespace muscl

} // namespace euler_kokkos

#endif // SOLVER_MHD_MUSCL_H_
