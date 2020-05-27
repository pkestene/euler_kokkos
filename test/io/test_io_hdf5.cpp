/**
 * Testing HDF5 io (serial and parallel).
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

// HDF5 IO implementation (to be tested)
#include "utils/io/IO_HDF5.h"

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
  void operator()(const typename std::enable_if<dim_==2, int>::type& index)  const
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

    data(i,j,ID) =   x+y;
    data(i,j,IE) = 2*x+y;
    data(i,j,IU) = 3*x+y;
    data(i,j,IV) = 4*x+y;
    
  }

  //! functor for 3d 
  template<unsigned int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename std::enable_if<dim_==3, int>::type& index)  const
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
#ifdef USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
#endif // USE_MPI
  
  Kokkos::initialize(argc, argv);

  int mpi_rank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif  
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

  int rank = 0;
#ifdef USE_MPI
  rank = params.myRank;
#endif

  // for upscale test
  ConfigMap configMapUp(configMap);
  {
    int nx = configMap.getInteger("mesh","nx",0);
    configMapUp.setInteger("mesh","nx",2*nx);

    int ny = configMap.getInteger("mesh","ny",0);
    configMapUp.setInteger("mesh","ny",2*ny);

    if (params.nz > 1) {
      int nz = configMap.getInteger("mesh","nz",0);
      configMapUp.setInteger("mesh","nz",2*nz);
    }

    // enable upscale upon reading
    configMapUp.setBool("run","restart_upscale",true);
    
  }
  //if (rank==0)
  //std::cout << configMapUp << std::endl;

  HydroParams paramsUp = HydroParams();
  paramsUp.setup(configMapUp);


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

    if (rank==0)
      std::cout << "2D test - create data\n";
    
    DataArray2d     data("data",params.isize,params.jsize,HYDRO_2D_NBVAR);
    DataArray2dHost data_host = Kokkos::create_mirror(data);

    DataArray2d     data2("data2",params.isize,params.jsize,HYDRO_2D_NBVAR);
    DataArray2dHost data2_host = Kokkos::create_mirror(data2);

    // create fake data
    InitData<2> functor(params, data);
    Kokkos::parallel_for(params.isize*params.jsize, functor);

    // save to file
    if (rank==0)
      std::cout << "2D test -- save data\n";
    
#ifdef USE_MPI
    io::Save_HDF5_mpi<TWO_D> writer(data, data_host, params, configMap, HYDRO_2D_NBVAR, var_names, 0, 0.0, "");
    writer.save();
    io::writeXdmfForHdf5Wrapper(params, configMap, var_names, 1, false);
#else
    io::Save_HDF5<TWO_D> writer(data, data_host, params, configMap, HYDRO_2D_NBVAR, var_names, 0, 0.0, "");
    writer.save();
    io::writeXdmfForHdf5Wrapper(params, configMap, var_names, 1, false);
#endif

    // try to reload file
    {
      if (rank==0)
	std::cout << "2D test -- reload and save data for comparison\n";
      
#ifdef USE_MPI
      io::Load_HDF5_mpi<TWO_D> reader(data2, params, configMap, HYDRO_2D_NBVAR, var_names);
      reader.load("output2d_0000000.h5");
      
      configMap.setString("output","outputPrefix","output2d_save");
      {
	io::Save_HDF5_mpi<TWO_D> writer(data2, data2_host, params, configMap, HYDRO_2D_NBVAR, var_names, 0, 0.0, "");
	writer.save();
	io::writeXdmfForHdf5Wrapper(params, configMap, var_names, 1, false);
      }
      // the two files should contain the same data
#else
      io::Load_HDF5<TWO_D> reader(data2, params, configMap, HYDRO_2D_NBVAR, var_names);
      reader.load("output2d_0000000.h5");
      
      configMap.setString("output","outputPrefix","output2d_save");
      {
	io::Save_HDF5<TWO_D> writer(data2, data2_host, params, configMap, HYDRO_2D_NBVAR, var_names, 0, 0.0, "");
	writer.save();
	io::writeXdmfForHdf5Wrapper(params, configMap, var_names, 1, false);
      }
      // the two files should contain the same data
#endif
    } // end reload test
    
    // reload and upscale test
    {
      DataArray2d     data2up("data2up",paramsUp.isize,paramsUp.jsize,HYDRO_2D_NBVAR);
      DataArray2dHost data2up_host = Kokkos::create_mirror(data2up);

      if (rank==0)
	std::cout << "2D test -- reload, upscale and save data\n";

      // set restart filename
      configMapUp.setString("run","restart_filename","output2d_0000000.h5");
      
#ifdef USE_MPI
       io::Load_HDF5_mpi<TWO_D> reader(data2up, paramsUp, configMapUp, HYDRO_2D_NBVAR, var_names);
       reader.load("output2d_0000000.h5");
      
       configMapUp.setString("output","outputPrefix","output2d_upscale_save");
       {
	 io::Save_HDF5_mpi<TWO_D> writer(data2up, data2up_host, paramsUp, configMapUp, HYDRO_2D_NBVAR, var_names, 0, 0.0, "");
	 writer.save();
	 io::writeXdmfForHdf5Wrapper(paramsUp, configMapUp, var_names, 1, false);
       }
       // the two files should contain the same data
#else
       io::Load_HDF5<TWO_D> reader(data2up, paramsUp, configMapUp, HYDRO_2D_NBVAR, var_names);
       reader.load("output2d_0000000.h5");
      
       configMapUp.setString("output","outputPrefix","output2d_upscale_save");
       {
	 io::Save_HDF5<TWO_D> writer(data2up, data2up_host, paramsUp, configMapUp, HYDRO_2D_NBVAR, var_names, 0, 0.0, "");
	 writer.save();
	 io::writeXdmfForHdf5Wrapper(paramsUp, configMapUp, var_names, 1, false);
       }
       // the two files should contain the same data
#endif
    } // end reload and upscale test

  } // end 2d test

  // TODO : add more testing for other options :
  // allghostincluded , halfResolution, ...
  // add a functor to compared data re-read with expected data, for all field
  
  // =================
  // ==== 3D test ====
  // =================
  if (params.nz > 1) {
    
    if (rank==0)
      std::cout << "3D test -- create data\n";

    DataArray3d     data("data",
			 params.isize,
			 params.jsize,
			 params.ksize,
			 HYDRO_3D_NBVAR);
    DataArray3dHost data_host = Kokkos::create_mirror(data);

    DataArray3d     data2("data2",
			  params.isize,
			  params.jsize,
			  params.ksize,
			  HYDRO_3D_NBVAR);
    DataArray3dHost data2_host = Kokkos::create_mirror(data2);

    // create fake data
    InitData<3> functor(params, data);
    Kokkos::parallel_for(params.isize*params.jsize*params.ksize, functor);

    // ================ save to file =======================
    // save to file
    if (rank==0)
      std::cout << "3D test -- save data\n";

#ifdef USE_MPI
    io::Save_HDF5_mpi<THREE_D> writer(data, data_host, params, configMap, HYDRO_3D_NBVAR, var_names, 0, 0.0, "");
    writer.save();
    io::writeXdmfForHdf5Wrapper(params, configMap, var_names, 1, false);
#else
    io::Save_HDF5<THREE_D> writer(data, data_host, params, configMap, HYDRO_3D_NBVAR, var_names, 0, 0.0, "");
    writer.save();
    io::writeXdmfForHdf5Wrapper(params, configMap, var_names, 1, false);
#endif // USE_MPI

    // try to reload file
    {
      if (rank==0)
	std::cout << "3D test -- reload and save data for comparison\n";
      
#ifdef USE_MPI
      io::Load_HDF5_mpi<THREE_D> reader(data2, params, configMap, HYDRO_3D_NBVAR, var_names);
      reader.load("output3d_0000000.h5");
      
      configMap.setString("output","outputPrefix","output3d_save");
      {
	io::Save_HDF5_mpi<THREE_D> writer(data2, data2_host, params, configMap, HYDRO_3D_NBVAR, var_names, 0, 0.0, "");
	writer.save();
	io::writeXdmfForHdf5Wrapper(params, configMap, var_names, 1, false);
      }
      // the two files should contain the same data
#else
      io::Load_HDF5<THREE_D> reader(data2, params, configMap, HYDRO_3D_NBVAR, var_names);
      reader.load("output3d_0000000.h5");
      
      configMap.setString("output","outputPrefix","output3d_save");
      {
	io::Save_HDF5<THREE_D> writer(data2, data2_host, params, configMap, HYDRO_3D_NBVAR, var_names, 0, 0.0, "");
	writer.save();
	io::writeXdmfForHdf5Wrapper(params, configMap, var_names, 1, false);
      }
      // the two files should contain the same data
#endif // USE_MPI
      
    } // end reload test

    // reload and upscale test
    {
      DataArray3d     data2up("data2up",
			      paramsUp.isize,
			      paramsUp.jsize,
			      paramsUp.ksize,
			      HYDRO_3D_NBVAR);
      DataArray3dHost data2up_host = Kokkos::create_mirror(data2up);
      
      if (rank==0)
	std::cout << "3D test -- reload, upscale and save data\n";
      
      // set restart filename
      configMapUp.setString("run","restart_filename","output3d_0000000.h5");
      
#ifdef USE_MPI
      io::Load_HDF5_mpi<THREE_D> reader(data2up, paramsUp, configMapUp, HYDRO_3D_NBVAR, var_names);
      reader.load("output3d_0000000.h5");
      
      configMapUp.setString("output","outputPrefix","output3d_upscale_save");
      {
	io::Save_HDF5_mpi<THREE_D> writer(data2up, data2up_host, paramsUp, configMapUp, HYDRO_3D_NBVAR, var_names, 0, 0.0, "");
	writer.save();
	io::writeXdmfForHdf5Wrapper(paramsUp, configMapUp, var_names, 1, false);
      }
      // the two files should contain the same data
#else
      io::Load_HDF5<THREE_D> reader(data2up, paramsUp, configMapUp, HYDRO_3D_NBVAR, var_names);
      reader.load("output3d_0000000.h5");
      
      configMapUp.setString("output","outputPrefix","output3d_upscale_save");
      {
	io::Save_HDF5<THREE_D> writer(data2up, data2up_host, paramsUp, configMapUp, HYDRO_3D_NBVAR, var_names, 0, 0.0, "");
	writer.save();
	io::writeXdmfForHdf5Wrapper(paramsUp, configMapUp, var_names, 1, false);
      }
      // the two files should contain the same data
#endif
    } // end reload and upscale test

  } // end 3D

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
