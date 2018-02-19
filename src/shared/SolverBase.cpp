#include "SolverBase.h"

#include "shared/utils.h"
#include "shared/BoundariesFunctors.h"

#ifdef USE_MPI
#include "shared/mpiBorderUtils.h"
#include "utils/mpiUtils/MpiCommCart.h"
#endif // USE_MPI

#include "utils/io/IO_Writer.h"

namespace euler_kokkos {

// =======================================================
// ==== CLASS SolverBase IMPL ============================
// =======================================================

// =======================================================
// =======================================================
SolverBase::SolverBase (HydroParams& params, ConfigMap& configMap) :
  params(params),
  configMap(configMap),
  solver_type(SOLVER_UNDEFINED)
{

  /*
   * init some variables by reading parameter file.
   */
  read_config();

  /*
   * other variables initialization.
   */
  m_times_saved = 0;
  m_nCells = -1;
  m_nDofsPerCell = -1;
  
  // create the timers
  timers[TIMER_TOTAL]      = std::make_shared<Timer>();
  timers[TIMER_IO]         = std::make_shared<Timer>();
  timers[TIMER_DT]         = std::make_shared<Timer>();
  timers[TIMER_BOUNDARIES] = std::make_shared<Timer>();
  timers[TIMER_NUM_SCHEME] = std::make_shared<Timer>();

  // init variables names
  m_variables_names[ID] = "rho";
  m_variables_names[IP] = "energy";
  m_variables_names[IU] = "mx"; // momentum component X
  m_variables_names[IV] = "my"; // momentum component Y
  m_variables_names[IW] = "mz"; // momentum component Z
  m_variables_names[IA] = "bx"; // mag field X
  m_variables_names[IB] = "by"; // mag field Y
  m_variables_names[IC] = "bz"; // mag field Z

  // init io writer is/should/must be called outside of constructor
  // right now we moved that in SolverFactory's method create
  //init_io_writer();
  
#ifdef USE_MPI
  const int gw = params.ghostWidth;
  const int isize = params.isize;
  const int jsize = params.jsize;
  const int ksize = params.ksize;
  const int nbvar = params.nbvar;
    
  if (params.dimType == TWO_D) {
    borderBufSend_xmin_2d = DataArray2d("borderBufSend_xmin",    gw, jsize, nbvar);
    borderBufSend_xmax_2d = DataArray2d("borderBufSend_xmax",    gw, jsize, nbvar);
    borderBufSend_ymin_2d = DataArray2d("borderBufSend_ymin", isize,    gw, nbvar);
    borderBufSend_ymax_2d = DataArray2d("borderBufSend_ymax", isize,    gw, nbvar);

    borderBufRecv_xmin_2d = DataArray2d("borderBufRecv_xmin",    gw, jsize, nbvar);
    borderBufRecv_xmax_2d = DataArray2d("borderBufRecv_xmax",    gw, jsize, nbvar);
    borderBufRecv_ymin_2d = DataArray2d("borderBufRecv_ymin", isize,    gw, nbvar);
    borderBufRecv_ymax_2d = DataArray2d("borderBufRecv_ymax", isize,    gw, nbvar);
  } else {
    borderBufSend_xmin_3d = DataArray3d("borderBufSend_xmin",    gw, jsize, ksize, nbvar);
    borderBufSend_xmax_3d = DataArray3d("borderBufSend_xmax",    gw, jsize, ksize, nbvar);
    borderBufSend_ymin_3d = DataArray3d("borderBufSend_ymin", isize,    gw, ksize, nbvar);
    borderBufSend_ymax_3d = DataArray3d("borderBufSend_ymax", isize,    gw, ksize, nbvar);
    borderBufSend_zmin_3d = DataArray3d("borderBufSend_zmin", isize, jsize,    gw, nbvar);
    borderBufSend_zmax_3d = DataArray3d("borderBufSend_zmax", isize, jsize,    gw, nbvar);
    
    borderBufRecv_xmin_3d = DataArray3d("borderBufRecv_xmin",    gw, jsize, ksize, nbvar);
    borderBufRecv_xmax_3d = DataArray3d("borderBufRecv_xmax",    gw, jsize, ksize, nbvar);
    borderBufRecv_ymin_3d = DataArray3d("borderBufRecv_ymin", isize,    gw, ksize, nbvar);
    borderBufRecv_ymax_3d = DataArray3d("borderBufRecv_ymax", isize,    gw, ksize, nbvar);
    borderBufRecv_zmin_3d = DataArray3d("borderBufRecv_zmin", isize, jsize,    gw, nbvar);
    borderBufRecv_zmax_3d = DataArray3d("borderBufRecv_zmax", isize, jsize,    gw, nbvar);
  }
#endif // USE_MPI
  
} // SolverBase::SolverBase

// =======================================================
// =======================================================
SolverBase::~SolverBase()
{

  // m_io_writer is now a shared (managed) pointer
  //delete m_io_writer;
  
} // SolverBase::~SolverBase

// =======================================================
// =======================================================
void
SolverBase::read_config()
{

  m_t     = configMap.getFloat("run", "tCurrent", 0.0);
  m_tEnd  = configMap.getFloat("run", "tEnd", 0.0);
  m_dt    = m_tEnd;
  m_cfl   = configMap.getFloat("hydro", "cfl", 1.0);
  m_iteration = 0;

  m_problem_name = configMap.getString("hydro", "problem", "unknown");

  m_solver_name = configMap.getString("run", "solver_name", "unknown");

  /* restart run : default is no */
  m_restart_run_enabled = configMap.getInteger("run", "restart_enabled", 0);
  m_restart_run_filename = configMap.getString ("run", "restart_filename", "");

} // SolverBase::read_config

// =======================================================
// =======================================================
void
SolverBase::compute_dt()
{

#ifdef USE_MPI

  // get local time step
  double dt_local = compute_dt_local();

  // synchronize all MPI processes
  params.communicator->synchronize();

  // perform MPI_Reduceall to get global time step
  double dt_global;
  params.communicator->allReduce(&dt_local, &dt_global, 1, params.data_type, hydroSimu::MpiComm::MIN);

  m_dt = dt_global;
  
#else

  m_dt = compute_dt_local();
  
#endif

  // correct m_dt if necessary
  if (m_t+m_dt > m_tEnd) {
    m_dt = m_tEnd - m_t;
  }

} // SolverBase::compute_dt

// =======================================================
// =======================================================
double
SolverBase::compute_dt_local()
{

  // the actual numerical scheme must provide it a genuine implementation

  return m_tEnd;
  
} // SolverBase::compute_dt_local

// =======================================================
// =======================================================
int
SolverBase::finished()
{

  return m_t >= (m_tEnd - 1e-14) || m_iteration >= params.nStepmax;
  
} // SolverBase::finished

// =======================================================
// =======================================================
void
SolverBase::next_iteration()
{

  // setup a timer here (?)
  
  // genuine implementation called here
  next_iteration_impl();

  // perform some stats here (?)
  
  // incremenent
  ++m_iteration;
  m_t += m_dt;

} // SolverBase::next_iteration

// =======================================================
// =======================================================
void
SolverBase::next_iteration_impl()
{

  // This is application dependent
  
} // SolverBase::next_iteration_impl

// =======================================================
// =======================================================
void
SolverBase::save_solution()
{

  // save solution to output file
  save_solution_impl();
  
  // increment output file number
  ++m_times_saved;
  
} // SolverBase::save_solution

// =======================================================
// =======================================================
void
SolverBase::save_solution_impl()
{
} // SolverBase::save_solution_impl

// =======================================================
// =======================================================
void
SolverBase::read_restart_file()
{

  // TODO
  
} // SolverBase::read_restart_file

// =======================================================
// =======================================================
int
SolverBase::should_save_solution()
{
  
  double interval = m_tEnd / params.nOutput;

  // params.nOutput == 0 means no output at all
  // params.nOutput < 0  means always output 
  if (params.nOutput < 0) {
    return 1;
  }

  if ((m_t - (m_times_saved - 1) * interval) > interval) {
    return 1;
  }

  /* always write the last time step */
  if (ISFUZZYNULL (m_t - m_tEnd)) {
    return 1;
  }

  return 0;
  
} // SolverBase::should_save_solution

// =======================================================
// =======================================================
void
SolverBase::save_data(DataArray2d             U,
		      DataArray2d::HostMirror Uh,
		      int iStep,
		      real_t time)
{
  m_io_writer->save_data(U, Uh, iStep, time, "");
}

// =======================================================
// =======================================================
void
SolverBase::save_data(DataArray3d             U,
		      DataArray3d::HostMirror Uh,
		      int iStep,
		      real_t time)
{
  m_io_writer->save_data(U, Uh, iStep, time, "");
}

// =======================================================
// =======================================================
void
SolverBase::save_data_debug(DataArray2d             U,
			    DataArray2d::HostMirror Uh,
			    int iStep,
			    real_t time,
			    std::string debug_name)
{
  m_io_writer->save_data(U, Uh, iStep, time, debug_name);
}

// =======================================================
// =======================================================
void
SolverBase::make_boundary(DataArray2d Udata, FaceIdType faceId, bool mhd_enabled)
{

  const int ghostWidth=params.ghostWidth;
  int nbIter = ghostWidth*std::max(params.isize,params.jsize);

  if (faceId == FACE_XMIN) {

    if (mhd_enabled) {
    
      MakeBoundariesFunctor2D_MHD<FACE_XMIN>::apply(params, Udata, nbIter);
            
    } else {

      MakeBoundariesFunctor2D<FACE_XMIN>::apply(params, Udata, nbIter);

    }

  }
    
  if (faceId == FACE_XMAX) {

    if (mhd_enabled) {
    
      MakeBoundariesFunctor2D_MHD<FACE_XMAX>::apply(params, Udata, nbIter);
            
    } else {

      MakeBoundariesFunctor2D<FACE_XMAX>::apply(params, Udata, nbIter);

    }

  }

  if (faceId == FACE_YMIN) {

    if (mhd_enabled) {
    
      MakeBoundariesFunctor2D_MHD<FACE_YMIN>::apply(params, Udata, nbIter);
            
    } else {

      MakeBoundariesFunctor2D<FACE_YMIN>::apply(params, Udata, nbIter);

    }

  }

  if (faceId == FACE_YMAX) {

    if (mhd_enabled) {
    
      MakeBoundariesFunctor2D_MHD<FACE_YMAX>::apply(params, Udata, nbIter);
            
    } else {

      MakeBoundariesFunctor2D<FACE_YMAX>::apply(params, Udata, nbIter);

    }

  }
  
} // SolverBase::make_boundary - 2d

// =======================================================
// =======================================================
void
SolverBase::make_boundary(DataArray3d Udata, FaceIdType faceId, bool mhd_enabled)
{
  
  const int ghostWidth=params.ghostWidth;
  
  int max_size = std::max(params.isize,params.jsize);
  max_size = std::max(max_size,params.ksize);
  int nbIter = ghostWidth * max_size * max_size;

  if (faceId == FACE_XMIN) {

    if (mhd_enabled) {
    
      MakeBoundariesFunctor3D_MHD<FACE_XMIN>::apply(params, Udata, nbIter);
            
    } else {

      MakeBoundariesFunctor3D<FACE_XMIN>::apply(params, Udata, nbIter);

    }

  }
    
  if (faceId == FACE_XMAX) {

    if (mhd_enabled) {
    
      MakeBoundariesFunctor3D_MHD<FACE_XMAX>::apply(params, Udata, nbIter);
            
    } else {

      MakeBoundariesFunctor3D<FACE_XMAX>::apply(params, Udata, nbIter);

    }

  }

  if (faceId == FACE_YMIN) {

    if (mhd_enabled) {
    
      MakeBoundariesFunctor3D_MHD<FACE_YMIN>::apply(params, Udata, nbIter);
            
    } else {

      MakeBoundariesFunctor3D<FACE_YMIN>::apply(params, Udata, nbIter);

    }

  }

  if (faceId == FACE_YMAX) {

    if (mhd_enabled) {
    
      MakeBoundariesFunctor3D_MHD<FACE_YMAX>::apply(params, Udata, nbIter);
            
    } else {

      MakeBoundariesFunctor3D<FACE_YMAX>::apply(params, Udata, nbIter);

    }

  }

  if (faceId == FACE_ZMIN) {

    if (mhd_enabled) {
    
      MakeBoundariesFunctor3D_MHD<FACE_ZMIN>::apply(params, Udata, nbIter);
            
    } else {

      MakeBoundariesFunctor3D<FACE_ZMIN>::apply(params, Udata, nbIter);

    }

  }

  if (faceId == FACE_ZMAX) {

    if (mhd_enabled) {
    
      MakeBoundariesFunctor3D_MHD<FACE_ZMAX>::apply(params, Udata, nbIter);
            
    } else {

      MakeBoundariesFunctor3D<FACE_ZMAX>::apply(params, Udata, nbIter);

    }

  }

} // SolverBase::make_boundary - 3d

// =======================================================
// =======================================================
void
SolverBase::make_boundaries_serial(DataArray2d Udata, bool mhd_enabled)
{

  make_boundary(Udata, FACE_XMIN, mhd_enabled);
  make_boundary(Udata, FACE_XMAX, mhd_enabled);
  make_boundary(Udata, FACE_YMIN, mhd_enabled);
  make_boundary(Udata, FACE_YMAX, mhd_enabled);
  
} // SolverBase::make_boundaries_serial - 2d
  
// =======================================================
// =======================================================
void
SolverBase::make_boundaries_serial(DataArray3d Udata, bool mhd_enabled)
{

  make_boundary(Udata, FACE_XMIN, mhd_enabled);
  make_boundary(Udata, FACE_XMAX, mhd_enabled);
  make_boundary(Udata, FACE_YMIN, mhd_enabled);
  make_boundary(Udata, FACE_YMAX, mhd_enabled);
  make_boundary(Udata, FACE_ZMIN, mhd_enabled);
  make_boundary(Udata, FACE_ZMAX, mhd_enabled);
    
} // SolverBase::make_boundaries_serial - 3d

#ifdef USE_MPI
// =======================================================
// =======================================================
void
SolverBase::make_boundaries_mpi(DataArray2d Udata, bool mhd_enabled)
{

  using namespace hydroSimu;
  
  // for each direction:
  // 1. copy boundary to MPI buffer
  // 2. send/recv buffer
  // 3. test if BC is BC_PERIODIC / BC_COPY then ... else ..

  // ======
  // XDIR
  // ======
  copy_boundaries(Udata,XDIR);
  transfert_boundaries_2d(XDIR);
    
  if (params.neighborsBC[X_MIN] == BC_COPY ||
      params.neighborsBC[X_MIN] == BC_PERIODIC) {
    copy_boundaries_back(Udata, XMIN);
  } else {
    make_boundary(Udata, FACE_XMIN, mhd_enabled);
  }
  
  if (params.neighborsBC[X_MAX] == BC_COPY ||
      params.neighborsBC[X_MAX] == BC_PERIODIC) {
    copy_boundaries_back(Udata, XMAX);
  } else {
    make_boundary(Udata, FACE_XMAX, mhd_enabled);
  }
  
  params.communicator->synchronize();
  
  // ======
  // YDIR
  // ======
  copy_boundaries(Udata,YDIR);
  transfert_boundaries_2d(YDIR);
  
  if (params.neighborsBC[Y_MIN] == BC_COPY ||
      params.neighborsBC[Y_MIN] == BC_PERIODIC) {
    copy_boundaries_back(Udata, YMIN);
  } else {
    make_boundary(Udata, FACE_YMIN, mhd_enabled);
  }
  
  if (params.neighborsBC[Y_MAX] == BC_COPY ||
      params.neighborsBC[Y_MAX] == BC_PERIODIC) {
    copy_boundaries_back(Udata, YMAX);
  } else {
    make_boundary(Udata, FACE_YMAX, mhd_enabled);
  }
  
  params.communicator->synchronize();
  
} // SolverBase::make_boundaries_mpi - 2d

// =======================================================
// =======================================================
void
SolverBase::make_boundaries_mpi(DataArray3d Udata, bool mhd_enabled)
{
  
  using namespace hydroSimu;

  // ======
  // XDIR
  // ======
  copy_boundaries(Udata,XDIR);
  transfert_boundaries_3d(XDIR);
  
  if (params.neighborsBC[X_MIN] == BC_COPY ||
      params.neighborsBC[X_MIN] == BC_PERIODIC) {
    copy_boundaries_back(Udata, XMIN);
  } else {
    make_boundary(Udata, FACE_XMIN, mhd_enabled);
  }
  
  if (params.neighborsBC[X_MAX] == BC_COPY ||
      params.neighborsBC[X_MAX] == BC_PERIODIC) {
    copy_boundaries_back(Udata, XMAX);
  } else {
    make_boundary(Udata, FACE_XMAX, mhd_enabled);
  }
  
  params.communicator->synchronize();

  // ======
  // YDIR
  // ======
  copy_boundaries(Udata,YDIR);
  transfert_boundaries_3d(YDIR);
  
  if (params.neighborsBC[Y_MIN] == BC_COPY ||
      params.neighborsBC[Y_MIN] == BC_PERIODIC) {
    copy_boundaries_back(Udata, YMIN);
  } else {
    make_boundary(Udata, FACE_YMIN, mhd_enabled);
  }
  
  if (params.neighborsBC[Y_MAX] == BC_COPY ||
      params.neighborsBC[Y_MAX] == BC_PERIODIC) {
    copy_boundaries_back(Udata, YMAX);
  } else {
    make_boundary(Udata, FACE_YMAX, mhd_enabled);
  }
  
  params.communicator->synchronize();

  // ======
  // ZDIR
  // ======
  copy_boundaries(Udata,ZDIR);
  transfert_boundaries_3d(ZDIR);
  
  if (params.neighborsBC[Z_MIN] == BC_COPY ||
      params.neighborsBC[Z_MIN] == BC_PERIODIC) {
    copy_boundaries_back(Udata, ZMIN);
  } else {
    make_boundary(Udata, FACE_ZMIN, mhd_enabled);
  }
  
  if (params.neighborsBC[Z_MAX] == BC_COPY ||
      params.neighborsBC[Z_MAX] == BC_PERIODIC) {
    copy_boundaries_back(Udata, ZMAX);
  } else {
    make_boundary(Udata, FACE_ZMAX, mhd_enabled);
  }
  
  params.communicator->synchronize();

} // SolverBase::make_boundaries_mpi - 3d

// =======================================================
// =======================================================
void
SolverBase::copy_boundaries(DataArray2d Udata, Direction dir)
{

  const int isize = params.isize;
  const int jsize = params.jsize;
  //const int ksize = params.ksize;
  const int gw    = params.ghostWidth;

  if (dir == XDIR) {
    
    const int nbIter = gw * jsize;
    
    CopyDataArray_To_BorderBuf<XMIN, TWO_D>::apply(borderBufSend_xmin_2d, Udata, gw, nbIter);
    CopyDataArray_To_BorderBuf<XMAX, TWO_D>::apply(borderBufSend_xmax_2d, Udata, gw, nbIter);
    
  }

  else if (dir == YDIR) {
    
    const int nbIter = isize * gw;
    
    CopyDataArray_To_BorderBuf<YMIN, TWO_D>::apply(borderBufSend_ymin_2d, Udata, gw, nbIter);
    CopyDataArray_To_BorderBuf<YMAX, TWO_D>::apply(borderBufSend_ymax_2d, Udata, gw, nbIter);
    
  }

  Kokkos::fence();
  
} // SolverBase::copy_boundaries - 2d

// =======================================================
// =======================================================
void
SolverBase::copy_boundaries(DataArray3d Udata, Direction dir)
{

  const int isize = params.isize;
  const int jsize = params.jsize;
  const int ksize = params.ksize;
  const int gw    = params.ghostWidth;

  if (dir == XDIR) {
    
    const int nbIter = gw * jsize * ksize;
    
    CopyDataArray_To_BorderBuf<XMIN, THREE_D>::apply(borderBufSend_xmin_3d, Udata, gw, nbIter);
    CopyDataArray_To_BorderBuf<XMAX, THREE_D>::apply(borderBufSend_xmax_3d, Udata, gw, nbIter);
  }

  else if (dir == YDIR) {
    
    const int nbIter = isize * gw * ksize;
    
    CopyDataArray_To_BorderBuf<YMIN, THREE_D>::apply(borderBufSend_ymin_3d, Udata, gw, nbIter);
    CopyDataArray_To_BorderBuf<YMAX, THREE_D>::apply(borderBufSend_ymax_3d, Udata, gw, nbIter);
    
  }
  
  else if (dir == ZDIR) {
    
    const int nbIter = isize * jsize * gw;
    
    CopyDataArray_To_BorderBuf<ZMIN, THREE_D>::apply(borderBufSend_zmin_3d, Udata, gw, nbIter);
    CopyDataArray_To_BorderBuf<ZMAX, THREE_D>::apply(borderBufSend_zmax_3d, Udata, gw, nbIter);
    
  }

  Kokkos::fence();

} // SolverBase::copy_boundaries - 3d

// =======================================================
// =======================================================
void
SolverBase::transfert_boundaries_2d(Direction dir)
{

  const int data_type = params.data_type;

  using namespace hydroSimu;
  
  /*
   * use MPI_Sendrecv
   */
  
  // two borders to send, two borders to receive

  if (dir == XDIR) {

    params.communicator->sendrecv(borderBufSend_xmin_2d.ptr_on_device(),
				  borderBufSend_xmin_2d.size(),
				  data_type, params.neighborsRank[X_MIN], 111,
				  borderBufRecv_xmax_2d.ptr_on_device(),
				  borderBufRecv_xmax_2d.size(),
				  data_type, params.neighborsRank[X_MAX], 111);
    
    params.communicator->sendrecv(borderBufSend_xmax_2d.ptr_on_device(),
				  borderBufSend_xmax_2d.size(),
				  data_type, params.neighborsRank[X_MAX], 111,
				  borderBufRecv_xmin_2d.ptr_on_device(),
				  borderBufRecv_xmin_2d.size(),
				  data_type, params.neighborsRank[X_MIN], 111);
    
  } else if (dir == YDIR) {

    params.communicator->sendrecv(borderBufSend_ymin_2d.ptr_on_device(),
				  borderBufSend_ymin_2d.size(),
				  data_type, params.neighborsRank[Y_MIN], 211,
				  borderBufRecv_ymax_2d.ptr_on_device(),
				  borderBufRecv_ymax_2d.size(),
				  data_type, params.neighborsRank[Y_MAX], 211);
    
    params.communicator->sendrecv(borderBufSend_ymax_2d.ptr_on_device(),
				  borderBufSend_ymax_2d.size(),
				  data_type, params.neighborsRank[Y_MAX], 211,
				  borderBufRecv_ymin_2d.ptr_on_device(),
				  borderBufRecv_ymin_2d.size(),
				  data_type, params.neighborsRank[Y_MIN], 211);
  }
  
} // SolverBase::transfert_boundaries_2d

// =======================================================
// =======================================================
void
SolverBase::transfert_boundaries_3d(Direction dir)
{

  const int data_type = params.data_type;

  using namespace hydroSimu;

  if (dir == XDIR) {

    params.communicator->sendrecv(borderBufSend_xmin_3d.ptr_on_device(),
				  borderBufSend_xmin_3d.size(),
				  data_type, params.neighborsRank[X_MIN], 111,
				  borderBufRecv_xmax_3d.ptr_on_device(),
				  borderBufRecv_xmax_3d.size(),
				  data_type, params.neighborsRank[X_MAX], 111);
    
    params.communicator->sendrecv(borderBufSend_xmax_3d.ptr_on_device(),
				  borderBufSend_xmax_3d.size(),
				  data_type, params.neighborsRank[X_MAX], 111,
				  borderBufRecv_xmin_3d.ptr_on_device(),
				  borderBufRecv_xmin_3d.size(),
				  data_type, params.neighborsRank[X_MIN], 111);

  } else if (dir == YDIR) {

    params.communicator->sendrecv(borderBufSend_ymin_3d.ptr_on_device(),
				  borderBufSend_ymin_3d.size(),
				  data_type, params.neighborsRank[Y_MIN], 211,
				  borderBufRecv_ymax_3d.ptr_on_device(),
				  borderBufRecv_ymax_3d.size(),
				  data_type, params.neighborsRank[Y_MAX], 211);
    
    params.communicator->sendrecv(borderBufSend_ymax_3d.ptr_on_device(),
				  borderBufSend_ymax_3d.size(),
				  data_type, params.neighborsRank[Y_MAX], 211,
				  borderBufRecv_ymin_3d.ptr_on_device(),
				  borderBufRecv_ymin_3d.size(),
				  data_type, params.neighborsRank[Y_MIN], 211);

  } else if (dir == ZDIR) {

    params.communicator->sendrecv(borderBufSend_zmin_3d.ptr_on_device(),
				  borderBufSend_zmin_3d.size(),
				  data_type, params.neighborsRank[Z_MIN], 311,
				  borderBufRecv_zmax_3d.ptr_on_device(),
				  borderBufRecv_zmax_3d.size(),
				  data_type, params.neighborsRank[Z_MAX], 311);
    
    params.communicator->sendrecv(borderBufSend_zmax_3d.ptr_on_device(),
				  borderBufSend_zmax_3d.size(),
				  data_type, params.neighborsRank[Z_MAX], 311,
				  borderBufRecv_zmin_3d.ptr_on_device(),
				  borderBufRecv_zmin_3d.size(),
				  data_type, params.neighborsRank[Z_MIN], 311);

  }
  
} // SolverBase::transfert_boundaries_3d

// =======================================================
// =======================================================
void
SolverBase::copy_boundaries_back(DataArray2d Udata, BoundaryLocation loc)
{

  const int isize = params.isize;
  const int jsize = params.jsize;
  //const int ksize = params.ksize;
  const int gw    = params.ghostWidth;

  if (loc == XMIN) {
    
    const int nbIter = gw * jsize;
    
    CopyBorderBuf_To_DataArray<XMIN, TWO_D>::apply(Udata, borderBufRecv_xmin_2d, gw, nbIter);

  }

  if (loc == XMAX) {

    const int nbIter = gw * jsize;
    
    CopyBorderBuf_To_DataArray<XMAX, TWO_D>::apply(Udata, borderBufRecv_xmax_2d, gw, nbIter);
    
  }

  if (loc == YMIN) {
    
    const int nbIter = isize * gw;
    
    CopyBorderBuf_To_DataArray<YMIN, TWO_D>::apply(Udata, borderBufRecv_ymin_2d, gw, nbIter);

  }

  if (loc == YMAX) {
    
    const int nbIter = isize * gw;

    CopyBorderBuf_To_DataArray<YMAX, TWO_D>::apply(Udata, borderBufRecv_ymax_2d, gw, nbIter);
    
  }
  
} // SolverBase::copy_boundaries_back - 2d

// =======================================================
// =======================================================
void
SolverBase::copy_boundaries_back(DataArray3d Udata, BoundaryLocation loc)
{

  const int isize = params.isize;
  const int jsize = params.jsize;
  const int ksize = params.ksize;
  const int gw    = params.ghostWidth;

  if (loc == XMIN) {
    
    const int nbIter = gw * jsize * ksize;
    
    CopyBorderBuf_To_DataArray<XMIN, THREE_D>::apply(Udata, borderBufRecv_xmin_3d, gw, nbIter);

  }

  if (loc == XMAX) {

    const int nbIter = gw * jsize * ksize;
    
    CopyBorderBuf_To_DataArray<XMAX, THREE_D>::apply(Udata, borderBufRecv_xmax_3d, gw, nbIter);
    
  }

  if (loc == YMIN) {
    
    const int nbIter = isize * gw * ksize;
    
    CopyBorderBuf_To_DataArray<YMIN, THREE_D>::apply(Udata, borderBufRecv_ymin_3d, gw, nbIter);

  }

  if (loc == YMAX) {
    
    const int nbIter = isize * gw * ksize;
    
    CopyBorderBuf_To_DataArray<YMAX, THREE_D>::apply(Udata, borderBufRecv_ymax_3d, gw, nbIter);
    
  }
  
  if (loc == ZMIN) {
    
    const int nbIter = isize * jsize * gw;
    
    CopyBorderBuf_To_DataArray<ZMIN, THREE_D>::apply(Udata, borderBufRecv_zmin_3d, gw, nbIter);

  }

  if (loc == ZMAX) {
    
    const int nbIter = isize * jsize * gw;

    CopyBorderBuf_To_DataArray<ZMAX, THREE_D>::apply(Udata, borderBufRecv_zmax_3d, gw, nbIter);
    
  }
  
} // SolverBase::copy_boundaries_back - 3d

#endif // USE_MPI

// =======================================================
// =======================================================
void SolverBase::init_io_writer()
{
  
  m_io_writer = std::make_shared<io::IO_Writer>(params, configMap, m_variables_names);

} // SolverBase::init_io_writer

// =======================================================
// =======================================================
void
SolverBase::save_data_debug(DataArray3d             U,
			    DataArray3d::HostMirror Uh,
			    int iStep,
			    real_t time,
			    std::string debug_name)
{
  m_io_writer->save_data(U, Uh, iStep, time, debug_name);
}

} // namespace euler_kokkos
