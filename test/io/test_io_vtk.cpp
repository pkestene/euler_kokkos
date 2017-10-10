/**
 * Testing VTK io (serial and parallel).
 */
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <type_traits> // for std::conditional

// minimal kokkos support
#include "shared/kokkos_shared.h"

#include "shared/HydroState.h" // for constants
#include "shared/real_type.h"   // choose between single and double precision
#include "shared/HydroParams.h" // read parameter file

// MPI support
#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // USE_MPI

// VTK IO implementation (to be tested)
#include "utils/io/IO_VTK.h"

// ===========================================================
// ===========================================================
// create some fake data
template<unsigned int dim>
class InitData
{

public:
  //! Decide at compile-time which data array type to use
  using DataArray  = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;
  
  InitData(HydroParams params, DataArray data) :
    params(params),
    data(data) {};
  ~InitData() {};

  //! functor for 2d 
  template<unsigned int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index)  const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    const int nx = params.nx;
    const int ny = params.ny;

#ifdef USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
#endif // USE_MPI
    
    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    int i,j;
    index2coord(index,i,j,isize,jsize);

    real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;

    data(i,j,ID) = x+y;
    
  }

  //! functor for 3d 
  template<unsigned int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index)  const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;

#ifdef USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
    const int k_mpi = params.myMpiPos[IZ];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
    const int k_mpi = 0;
#endif // USE_MPI
    
    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t zmin = params.zmin;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;
    real_t z = zmin + dz/2 + (k+nz*k_mpi-ghostWidth)*dz;

    data(i,j,k,ID) = x+y+z;
  }

  HydroParams params;
  DataArray data;
  
}; // class InitData


// ===========================================================
// ===========================================================
int main(int argc, char* argv[])
{

  // Create MPI session if MPI enabled
#ifdef USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
#endif // USE_MPI

  Kokkos::initialize(argc, argv);
  
  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";
    
    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if ( Kokkos::hwloc::available() ) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
          << "] )"
          << std::endl ;

    }
#if defined( CUDA )
    Kokkos::Cuda::print_configuration( msg );
#else
    Kokkos::OpenMP::print_configuration( msg );
#endif
    std::cout << msg.str();
    std::cout << "##########################\n";
  }

  if (argc != 2) {
    fprintf(stderr, "Error: wrong number of argument; input filename must be the only parameter on the command line\n");
    Kokkos::finalize();
    exit(EXIT_FAILURE);
  }
  
  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = std::string(argv[1]);
  ConfigMap configMap(input_file);
  
  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

  std::map<int, std::string> var_names;
  var_names[ID] = "rho";
  var_names[IP] = "energy";
  var_names[IU] = "mx";
  var_names[IV] = "my";
  var_names[IW] = "mz";
  
  // =================
  // ==== 2D test ====
  // =================
  if (params.nz == 1) {
    
    DataArray2d     data("data",params.isize,params.jsize,HYDRO_2D_NBVAR);
    DataArray2dHost data_host = Kokkos::create_mirror(data);

    // create fake data
    InitData<2> functor(params, data);
    Kokkos::parallel_for(params.isize*params.jsize, functor);

    // save to file
#ifdef USE_MPI
    ppkMHD::io::save_VTK_2D_mpi(data, data_host, params, configMap, HYDRO_2D_NBVAR, var_names, 0, "");
#else
    ppkMHD::io::save_VTK_2D(data, data_host, params, configMap, HYDRO_2D_NBVAR, var_names, 0, "");
#endif
    
  }

  // =================
  // ==== 3D test ====
  // =================
  if (params.nz > 1) {
    
    DataArray3d     data("data",params.isize,params.jsize,params.ksize,HYDRO_3D_NBVAR);
    DataArray3dHost data_host = Kokkos::create_mirror(data);

    // create fake data
    InitData<3> functor(params, data);
    Kokkos::parallel_for(params.isize*params.jsize*params.ksize, functor);

    // save to file
#ifdef USE_MPI
    ppkMHD::io::save_VTK_3D_mpi(data, data_host, params, configMap, HYDRO_3D_NBVAR, var_names, 0, "");
#else
    ppkMHD::io::save_VTK_3D(data, data_host, params, configMap, HYDRO_3D_NBVAR, var_names, 0, "");
#endif
    
  }

 
#ifdef CUDA
  Kokkos::Cuda::finalize();
  Kokkos::HostSpace::execution_space::finalize();
#else
  Kokkos::finalize();
#endif

  return EXIT_SUCCESS;
  
}
