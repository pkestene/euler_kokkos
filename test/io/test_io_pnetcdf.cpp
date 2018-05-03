/**
 * Testing PNETCDF io (serial and parallel).
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
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>

// PNETCDF IO implementation (to be tested)
#include "utils/io/IO_PNETCDF.h"

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

    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
    
    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    int i,j;
    index2coord(index,i,j,isize,jsize);

    real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;

    data(i,j,ID) =   x+y;
    data(i,j,IE) = 2*x+y;
    data(i,j,IU) = 3*x+y;
    data(i,j,IV) = 4*x+y;
    
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

    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
    const int k_mpi = params.myMpiPos[IZ];
    
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

    data(i,j,k,ID) =   x+y+z;
    data(i,j,k,IE) = 2*x+y+z;
    data(i,j,k,IU) = 3*x+y+z;
    data(i,j,k,IV) = 4*x+y+z;
    data(i,j,k,IW) = 5*x+y+z;

    // data(i,j,k,ID) = index + 0*isize*jsize*ksize;//  x+y+z;
    // data(i,j,k,IE) = index + 1*isize*jsize*ksize;//2*x+y+z;
    // data(i,j,k,IU) = index + 2*isize*jsize*ksize;//3*x+y+z;
    // data(i,j,k,IV) = index + 3*isize*jsize*ksize;//4*x+y+z;
    // data(i,j,k,IW) = index + 4*isize*jsize*ksize;//5*x+y+z;
  }

  HydroParams params;
  DataArray data;
  
}; // class InitData


// ===========================================================
// ===========================================================
int main(int argc, char* argv[])
{

  // namespace alias
  namespace io = ::euler_kokkos::io;

  // Create MPI session if MPI enabled
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);

  Kokkos::initialize(argc, argv);

  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank==0) {
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
    Kokkos::print_configuration( msg );
    std::cout << msg.str();
    std::cout << "##########################\n";
  }

  // if (argc != 2) {
  //   fprintf(stderr, "Error: wrong number of argument; input filename must be the only parameter on the command line\n");
  //   Kokkos::finalize();
  //   exit(EXIT_FAILURE);
  // }
  
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
  var_names[IU] = "rho_vx";
  var_names[IV] = "rho_vy";
  var_names[IW] = "rho_vz";
  
  // =================
  // ==== 2D test ====
  // =================
  if (params.nz == 1) {

    if (mpi_rank==0)
      std::cout << "2D test\n";
    
    DataArray2d     data("data",params.isize,params.jsize,HYDRO_2D_NBVAR);
    DataArray2dHost data_host = Kokkos::create_mirror(data);

    // create fake data
    InitData<2> functor(params, data);
    Kokkos::parallel_for(params.isize*params.jsize, functor);

    // save to file
    io::Save_PNETCDF<TWO_D> writer(data, data_host, params, configMap, HYDRO_2D_NBVAR, var_names, 0, 0.0, "");
    writer.save();
    
  }

  // =================
  // ==== 3D test ====
  // =================
  if (params.nz > 1) {
    
    if (mpi_rank==0)
      std::cout << "3D test\n";

    DataArray3d     data("data",params.isize,params.jsize,params.ksize,HYDRO_3D_NBVAR);
    DataArray3dHost data_host = Kokkos::create_mirror(data);

    // create fake data
    InitData<3> functor(params, data);
    Kokkos::parallel_for(params.isize*params.jsize*params.ksize, functor);

    // save to file
    io::Save_PNETCDF<THREE_D> writer(data, data_host, params, configMap, HYDRO_3D_NBVAR, var_names, 0, 0.0, "");
    writer.save();
    
  }
 
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
