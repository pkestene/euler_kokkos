/**
 * Class SolverHydroMuscl implementation.
 *
 * Main class for solving hydrodynamics (Euler) with MUSCL-Hancock scheme for 2D/3D.
 */
#ifndef SOLVER_HYDRO_MUSCL_H_
#define SOLVER_HYDRO_MUSCL_H_

#include <cstdio>
#include <cstdbool>
#include <iostream>

// shared
#include "shared/SolverBase.h"
#include "shared/HydroParams.h"
#include "shared/kokkos_shared.h"

// the actual computational functors called in HydroRun
#include "muscl/HydroRunFunctors2D.h"
#include "muscl/HydroRunFunctors3D.h"

// Init conditions functors
#include "muscl/HydroInitFunctors2D.h"
#include "muscl/HydroInitFunctors3D.h"

// for IO
#include <utils/io/IO_ReadWrite.h>

// for init condition
#include "shared/problems/BlastParams.h"
#include "shared/problems/ImplodeParams.h"
#include "shared/problems/KHParams.h"
#include "shared/problems/GreshoParams.h"

namespace euler_kokkos
{
namespace muscl
{

/**
 * Main hydrodynamics data structure for 2D/3D MUSCL-Hancock scheme.
 */
template <int dim>
class SolverHydroMuscl : public euler_kokkos::SolverBase
{

public:
  //! Decide at compile-time which data array to use for 2d or 3d
  using DataArray = typename std::conditional<dim == 2, DataArray2d, DataArray3d>::type;

  //! Data array typedef for host memory space
  using DataArrayHost = typename std::conditional<dim == 2, DataArray2dHost, DataArray3dHost>::type;

  //! Decide at compile-time which data array to use for 2d or 3d
  using VectorField = typename std::conditional<dim == 2, VectorField2d, VectorField3d>::type;

  SolverHydroMuscl(HydroParams & params, ConfigMap & configMap);
  virtual ~SolverHydroMuscl();

  /**
   * Static creation method called by the solver factory.
   */
  static SolverBase *
  create(HydroParams & params, ConfigMap & configMap)
  {
    SolverHydroMuscl<dim> * solver = new SolverHydroMuscl<dim>(params, configMap);

    return solver;
  }

  DataArray     U;     //!< hydrodynamics conservative variables arrays
  DataArrayHost Uhost; //!< U mirror on host memory space
  DataArray     U2;    //!< hydrodynamics conservative variables arrays
  DataArray     Q;     //!< hydrodynamics primitive    variables array

  // implementation 0 and 1
  DataArray Fluxes_x; //!< implementation 0
  DataArray Fluxes_y; //!< implementation 0
  DataArray Fluxes_z; //!< implementation 0

  // implementation 1 only
  DataArray Slopes_x; //!< implementation 1 only
  DataArray Slopes_y; //!< implementation 1 only
  DataArray Slopes_z; //!< implementation 1 only

  /* Gravity field */
  VectorField gravity_field;

  // riemann_solver_t riemann_solver_fn; //!< riemann solver function pointer

  /*
   * methods
   */

  // fill boundaries / ghost 2d / 3d
  void
  make_boundaries(DataArray Udata);

  // host routines (initialization)
  void
  init_implode(DataArray Udata); // 2d and 3d
  void
  init_blast(DataArray Udata); // 2d and 3d
  void
  init_kelvin_helmholtz(DataArray Udata); // 2d and 3d
  void
  init_gresho_vortex(DataArray Udata); // 2d and 3d
  void
  init_four_quadrant(DataArray Udata); // 2d only
  void
  init_isentropic_vortex(DataArray Udata); // 2d only
  void
  init_rayleigh_taylor(DataArray Udata, VectorField gravity_field); // 2d and 3d

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
  godunov_unsplit(real_t dt);

  void
  godunov_unsplit_impl(DataArray data_in, DataArray data_out, real_t dt);

  void
  convertToPrimitives(DataArray Udata);

  // void computeTrace(DataArray Udata, real_t dt);

  void
  computeFluxesAndUpdate(DataArray Udata, real_t dt);

  // output
  void
  save_solution_impl();

  int isize, jsize, ksize;

}; // class SolverHydroMuscl

// =======================================================
// ==== CLASS SolverHydroMuscl IMPL ======================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
template <int dim>
SolverHydroMuscl<dim>::SolverHydroMuscl(HydroParams & params, ConfigMap & configMap)
  : SolverBase(params, configMap)
  , U()
  , U2()
  , Q()
  , Fluxes_x()
  , Fluxes_y()
  , Fluxes_z()
  , Slopes_x()
  , Slopes_y()
  , Slopes_z()
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

    if (params.implementationVersion == 0)
    {

      Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
      Fluxes_y = DataArray("Fluxes_y", isize, jsize, nbvar);

      total_mem_size += isize * jsize * nbvar * sizeof(real_t) * 2; // 1+1 for Fluxes_x+Fluxes_y
    }
    else if (params.implementationVersion == 1)
    {

      Slopes_x = DataArray("Slope_x", isize, jsize, nbvar);
      Slopes_y = DataArray("Slope_y", isize, jsize, nbvar);

      // direction splitting (only need one flux array)
      Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
      Fluxes_y = Fluxes_x;

      total_mem_size +=
        isize * jsize * nbvar * sizeof(real_t) * 3; // 1+1+1 for Slopes_x+Slopes_y+Fluxes_x
    }

    if (m_gravity.enabled)
    {
      gravity_field = VectorField("gravity field", isize, jsize);
      total_mem_size += isize * jsize * 2;
    }
  }
  else
  {

    U = DataArray("U", isize, jsize, ksize, nbvar);
    Uhost = Kokkos::create_mirror(U);
    U2 = DataArray("U2", isize, jsize, ksize, nbvar);
    Q = DataArray("Q", isize, jsize, ksize, nbvar);

    total_mem_size += isize * jsize * ksize * nbvar * sizeof(real_t) * 3; // 1+1+1=3 for U+U2+Q

    if (params.implementationVersion == 0)
    {

      Fluxes_x = DataArray("Fluxes_x", isize, jsize, ksize, nbvar);
      Fluxes_y = DataArray("Fluxes_y", isize, jsize, ksize, nbvar);
      Fluxes_z = DataArray("Fluxes_z", isize, jsize, ksize, nbvar);

      total_mem_size += isize * jsize * ksize * nbvar * sizeof(real_t) * 3; // 1+1+1=3 Fluxes
    }
    else if (params.implementationVersion == 1)
    {

      Slopes_x = DataArray("Slope_x", isize, jsize, ksize, nbvar);
      Slopes_y = DataArray("Slope_y", isize, jsize, ksize, nbvar);
      Slopes_z = DataArray("Slope_z", isize, jsize, ksize, nbvar);

      // direction splitting (only need one flux array)
      Fluxes_x = DataArray("Fluxes_x", isize, jsize, ksize, nbvar);
      Fluxes_y = Fluxes_x;
      Fluxes_z = Fluxes_x;

      total_mem_size += isize * jsize * ksize * nbvar * sizeof(real_t) * 4; // 1+1+1+1=4 Slopes
    }

    if (m_gravity.enabled)
    {
      gravity_field = VectorField("gravity field", isize, jsize, ksize);
      total_mem_size += isize * jsize * ksize * 3;
    }

  } // dim == 2 / 3

  // perform init condition
  init(U);

  // initialize boundaries
  make_boundaries(U);

  // copy U into U2
  Kokkos::deep_copy(U2, U);

  // compute initialize time step
  compute_dt();

  int myRank = 0;
#ifdef EULER_KOKKOS_USE_MPI
  myRank = params.myRank;
#endif // EULER_KOKKOS_USE_MPI

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

} // SolverHydroMuscl::SolverHydroMuscl

// =======================================================
// =======================================================
/**
 *
 */
template <int dim>
SolverHydroMuscl<dim>::~SolverHydroMuscl()
{} // SolverHydroMuscl::~SolverHydroMuscl

// =======================================================
// =======================================================
template <int dim>
void
SolverHydroMuscl<dim>::make_boundaries(DataArray Udata)
{

  // this routine is specialized for 2d / 3d

} // SolverHydroMuscl<dim>::make_boundaries

// 2d specialization
template <>
void
SolverHydroMuscl<2>::make_boundaries(DataArray Udata);

// 3d specialization
template <>
void
SolverHydroMuscl<3>::make_boundaries(DataArray Udata);

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
template <int dim>
void
SolverHydroMuscl<dim>::init_implode(DataArray Udata)
{

  ImplodeParams iparams = ImplodeParams(configMap);

  // alias to actual device functor
  using InitImplodeFunctor =
    typename std::conditional<dim == 2, InitImplodeFunctor2D, InitImplodeFunctor3D>::type;

  // perform init
  InitImplodeFunctor::apply(params, iparams, Udata);

} // SolverHydroMuscl::init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
template <int dim>
void
SolverHydroMuscl<dim>::init_blast(DataArray Udata)
{

  BlastParams blastParams = BlastParams(configMap);

  // alias to actual device functor
  using InitBlastFunctor =
    typename std::conditional<dim == 2, InitBlastFunctor2D, InitBlastFunctor3D>::type;

  // perform init
  InitBlastFunctor::apply(params, blastParams, Udata);

} // SolverHydroMuscl::init_blast

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
SolverHydroMuscl<dim>::init_kelvin_helmholtz(DataArray Udata)
{

  KHParams khParams = KHParams(configMap);

  // alias to actual device functor
  using InitKelvinHelmholtzFunctor = typename std::
    conditional<dim == 2, InitKelvinHelmholtzFunctor2D, InitKelvinHelmholtzFunctor3D>::type;

  // perform init
  InitKelvinHelmholtzFunctor::apply(params, khParams, Udata);

} // SolverHydroMuscl::init_kelvin_helmholtz

// =======================================================
// =======================================================
/**
 * Hydrodynamical Gresho vortex Test.
 *
 * \sa https://www.cfd-online.com/Wiki/Gresho_vortex
 * \sa https://arxiv.org/abs/1409.7395 - section 4.2.3
 * \sa https://arxiv.org/abs/1612.03910
 *
 */
template <int dim>
void
SolverHydroMuscl<dim>::init_gresho_vortex(DataArray Udata)
{

  GreshoParams gvParams = GreshoParams(configMap);

  // alias to actual device functor
  using InitGreshoVortexFunctor =
    typename std::conditional<dim == 2, InitGreshoVortexFunctor2D, InitGreshoVortexFunctor3D>::type;

  // perform init
  InitGreshoVortexFunctor::apply(params, gvParams, Udata);

} // SolverHydroMuscl<dim>::init_gresho_vortex

// =======================================================
// =======================================================
/**
 * Init four quadrant (piecewise constant).
 *
 * Four quadrant 2D riemann problem.
 *
 * See article: Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340
 */
template <int dim>
void
SolverHydroMuscl<dim>::init_four_quadrant(DataArray Udata)
{

  // specialized only for 2d
  std::cerr << "You shouldn't be here: four quadrant problem is not implemented in 3D !\n";

} // SolverHydroMuscl<dim>::init_four_quadrant

// 2d specialization
template <>
void
SolverHydroMuscl<2>::init_four_quadrant(DataArray Udata);

// =======================================================
// =======================================================
/**
 * Isentropic vortex advection test.
 * https://www.cfd-online.com/Wiki/2-D_vortex_in_isentropic_flow
 * https://hal.archives-ouvertes.fr/hal-01485587/document
 */
template <int dim>
void
SolverHydroMuscl<dim>::init_isentropic_vortex(DataArray Udata)
{

  // specialized only for 2d
  std::cerr << "You shouldn't be here: isentropic vortex is not implemented in 3D !\n";

} // SolverHydroMuscl::init_isentropic_vortex

// 2d specialization
template <>
void
SolverHydroMuscl<2>::init_isentropic_vortex(DataArray Udata);

// =======================================================
// =======================================================
/**
 * Hydrodynamical Rayleigh Taylor instability Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html
 *
 */
template <int dim>
void
SolverHydroMuscl<dim>::init_rayleigh_taylor(DataArray Udata, VectorField gravity)
{

  RayleighTaylorInstabilityParams rtiParams = RayleighTaylorInstabilityParams(configMap);

  // alias to actual device functor
  using RTIFunctor = typename std::conditional<dim == 2,
                                               RayleighTaylorInstabilityFunctor2D,
                                               RayleighTaylorInstabilityFunctor3D>::type;

  // perform init
  RTIFunctor::apply(params, rtiParams, Udata, gravity);

} // SolverHydroMuscl::init_rayleigh_taylor

// =======================================================
// =======================================================
template <int dim>
void
SolverHydroMuscl<dim>::init_restart(DataArray Udata)
{

  int myRank = 0;
#ifdef EULER_KOKKOS_USE_MPI
  myRank = params.myRank;
#endif // EULER_KOKKOS_USE_MPI

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

  // in case of turbulence problem, we also need to re-initialize the
  // random forcing field
  // if (!problemName.compare("turbulence")) {
  //   this->init_randomForcing();
  // }

  // in case of Ornstein-Uhlenbeck turbulence problem,
  // we also need to re-initialize the random forcing field
  // if (!problemName.compare("turbulence-Ornstein-Uhlenbeck")) {

  //   bool restartEnabled = true;

  //   std::string forcing_filename = configMap.getString("turbulence-Ornstein-Uhlenbeck",
  //   "forcing_input_file",  "");

  //   if (restartUpscaleEnabled) {

  //     // use default parameter when restarting and upscaling
  //     pForcingOrnsteinUhlenbeck -> init_forcing(false);

  //   } else if ( forcing_filename.size() != 0) {

  //     // if forcing filename is provided, we use it
  //     pForcingOrnsteinUhlenbeck -> init_forcing(false); // call to allocate
  //     pForcingOrnsteinUhlenbeck -> input_forcing(forcing_filename);

  //   } else {

  //     // the forcing parameter filename is build upon configMap information
  //     pForcingOrnsteinUhlenbeck -> init_forcing(restartEnabled, timeStep);

  //   }

  // } // end restart problem turbulence-Ornstein-Uhlenbeck


  if (myRank == 0)
  {
    std::cout << "### This is a restarted run ! Current time is " << m_t << " ###\n";
  }

  // some extra stuff that need to be done here (useful when MRI is activated)
  // restart_run_extra_work();


} // SolverHydroMuscl<dim>::init_restart

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
template <int dim>
double
SolverHydroMuscl<dim>::compute_dt_local()
{

  real_t    dt;
  real_t    invDt = ZERO_F;
  DataArray Udata;

  // which array is the current one ?
  if (m_iteration % 2 == 0)
    Udata = U;
  else
    Udata = U2;

  if (m_gravity.enabled)
  {

    // alias to actual device functor
    using ComputeDtFunctor = typename std::
      conditional<dim == 2, ComputeDtGravityFunctor2D, ComputeDtGravityFunctor3D>::type;

    // call device functor
    ComputeDtFunctor::apply(params, params.settings.cfl, gravity_field, Udata, invDt);
  }
  else
  {

    // regular cfl

    // alias to actual device functor
    using ComputeDtFunctor =
      typename std::conditional<dim == 2, ComputeDtFunctor2D, ComputeDtFunctor3D>::type;

    // call device functor
    ComputeDtFunctor::apply(params, Udata, invDt);
  }

  dt = params.settings.cfl / invDt;

  return dt;

} // SolverHydroMuscl::compute_dt_local

// =======================================================
// =======================================================
template <int dim>
void
SolverHydroMuscl<dim>::next_iteration_impl()
{

  int myRank = 0;

#ifdef EULER_KOKKOS_USE_MPI
  myRank = params.myRank;
#endif // EULER_KOKKOS_USE_MPI

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

  // compute new dt
  timers[TIMER_DT]->start();
  compute_dt();
  timers[TIMER_DT]->stop();

  // perform one step integration
  godunov_unsplit(m_dt);

} // SolverHydroMuscl::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
template <int dim>
void
SolverHydroMuscl<dim>::godunov_unsplit(real_t dt)
{

  if (m_iteration % 2 == 0)
  {
    godunov_unsplit_impl(U, U2, dt);
  }
  else
  {
    godunov_unsplit_impl(U2, U, dt);
  }

} // SolverHydroMuscl::godunov_unsplit

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of Godunov scheme
// ///////////////////////////////////////////
template <int dim>
void
SolverHydroMuscl<dim>::godunov_unsplit_impl(DataArray data_in, DataArray data_out, real_t dt)
{

  // 2d / 3d implementation are specialized in implementation file

} // SolverHydroMuscl<dim>::godunov_unsplit_impl

// 2d version
template <>
void
SolverHydroMuscl<2>::godunov_unsplit_impl(DataArray data_in, DataArray data_out, real_t dt);

// 3d version
template <>
void
SolverHydroMuscl<3>::godunov_unsplit_impl(DataArray data_in, DataArray data_out, real_t dt);

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
template <int dim>
void
SolverHydroMuscl<dim>::convertToPrimitives(DataArray Udata)
{

  // alias to actual device functor
  using ConvertToPrimitivesFunctor = typename std::
    conditional<dim == 2, ConvertToPrimitivesFunctor2D, ConvertToPrimitivesFunctor3D>::type;

  // call device functor
  ConvertToPrimitivesFunctor::apply(params, Udata, Q);

} // SolverHydroMuscl::convertToPrimitives

// =======================================================
// =======================================================
template <int dim>
void
SolverHydroMuscl<dim>::save_solution_impl()
{

  timers[TIMER_IO]->start();
  if (m_iteration % 2 == 0)
    save_data(U, Uhost, m_times_saved, m_t);
  else
    save_data(U2, Uhost, m_times_saved, m_t);

  timers[TIMER_IO]->stop();

} // SolverHydroMuscl::save_solution_impl()

} // namespace muscl

} // namespace euler_kokkos

#endif // SOLVER_HYDRO_MUSCL_H_
