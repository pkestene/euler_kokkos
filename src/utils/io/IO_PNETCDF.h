#ifndef IO_PNETCDF_H_
#define IO_PNETCDF_H_

#include <iostream>
#include <type_traits>

#include <map>
#include <string>

#include <shared/kokkos_shared.h>
//class HydroParams;
//class ConfigMap;
#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"

#ifdef USE_MPI
#include "utils/mpiUtils/MpiComm.h"
#endif // USE_MPI

// for Parallel-netCDF support
#include <pnetcdf.h>

#define PNETCDF_HANDLE_ERROR {                        \
    if (err != NC_NOERR)                              \
        printf("Error at line %d (%s)\n", __LINE__,   \
               ncmpi_strerror(err));                  \
}

#include "IO_common.h"

namespace euler_kokkos { namespace io {

// =======================================================
// =======================================================
/**
 *
 */
template<DimensionType d>
class Save_PNETCDF
{
public:
  //! Decide at compile-time which data array type to use
  using DataArray  = typename std::conditional<d==TWO_D,DataArray2d,DataArray3d>::type;
  using DataArrayHost  = typename std::conditional<d==TWO_D,DataArray2dHost,DataArray3dHost>::type;

  Save_PNETCDF(DataArray     Udata,
	       DataArrayHost Uhost,
	       HydroParams& params,
	       ConfigMap& configMap,
	       int nbvar,
	       const std::map<int, std::string>& variables_names,
	       int iStep,
	       real_t totalTime,
	       std::string debug_name) :
    Udata(Udata), Uhost(Uhost), params(params), configMap(configMap),
    nbvar(nbvar), variables_names(variables_names),
    iStep(iStep), totalTime(totalTime), debug_name(debug_name)
  {};
  ~Save_PNETCDF() {};

  template<DimensionType d_ = d>
  void copy_buffer(typename std::enable_if<d_==TWO_D, real_t>::type *&data,
		   int iStop, int jStop, int kStop, int iVar, KokkosLayout layout)
  {
    if (layout == KOKKOS_LAYOUT_RIGHT) { // transpose array to make data contiguous in memory
      int dI=0;
      for (int j= 0; j < jStop; j++) {
	for(int i = 0; i < iStop; i++) {
	  data[dI] = Uhost(i,j,iVar);
	  dI++;
	}
      }
    } else {
      //data = Uhost.data() + isize*jsize*nvar;
      int dI=0;
      for(int i = 0; i < iStop; i++) {
	for (int j= 0; j < jStop; j++) {
	  data[dI] = Uhost(i,j,iVar);
	  dI++;
	}
      }
    }

  } // copy_buffer

  template<DimensionType d_=d>
  void copy_buffer(typename std::enable_if<d_==THREE_D, real_t>::type *&data,
		   int iStop, int jStop, int kStop, int iVar, KokkosLayout layout)
  {
    if (layout == KOKKOS_LAYOUT_RIGHT) { // transpose array to make data contiguous in memory
 
      int dI=0;
      for (int k= 0; k < kStop; k++)
	for (int j= 0; j < jStop; j++)
	  for(int i = 0; i < iStop; i++) {
	    data[dI] = Uhost(i,j,k,iVar);
	    dI++;
	  }
    } else {
      //data = Uhost.data() + isize*jsize*ksize*nvar;
      int dI=0;
      for(int i = 0; i < iStop; i++)
	for (int j= 0; j < jStop; j++)
	  for (int k= 0; k < kStop; k++) {
	    data[dI] = Uhost(i,j,k,iVar);
	    dI++;
	  }
    }

  } // copy_buffer / 3D
  
  // =======================================================
  // =======================================================
  /**
   * Dump computation results (conservative variables) into a file
   * (Parallel-netCDF file format) over MPI. 
   * File extension is nc. 
   * 
   * NetCDF file creation supports:
   * - CDF-2 (using creation mode NC_64BIT_OFFSET)
   * - CDF-5 (using creation mode NC_64BIT_DATA)
   *
   * \note NetCDF file can be viewed by ncBrowse; see also ncdump.
   * \warning ncdump does not support CDF-5 format ! 
   *
   * All MPI pieces are written in the same file with parallel
   * IO (MPI pieces are directly re-assembled by Parallel-netCDF library).
   *
   * \param[in] U A reference to a hydro simulation HostArray
   * \param[in] nStep The current time step, used to label results filename. 
   *
   * If library Parallel-netCDF is not available, do nothing.
   */
  void save()
  {
    using namespace hydroSimu; // for MpiComm (this namespace is tout-pourri)
    
    // sub-domain sizes
    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;

    // domain decomposition sizes
    const int mx = params.mx;
    const int my = params.my;
    const int mz = params.mz;

    // sub-domaine sizes with ghost cells
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;

    const int ghostWidth = params.ghostWidth;

    const int dimType = params.dimType;

    const bool ghostIncluded = configMap.getBool("output","ghostIncluded",false);
    const bool allghostIncluded = configMap.getBool("output","allghostIncluded",false);

    const bool reassembleInFile = configMap.getBool("output", "reassembleInFile", true);
    const bool mhdEnabled = params.mhdEnabled;
    
    const int myRank = params.myRank;

    // netcdf file id
    int ncFileId;
    int err;

    // file creation mode
    int ncCreationMode = NC_CLOBBER;
    bool useCDF5 = configMap.getBool("output","pnetcdf_cdf5",false);
    if (useCDF5)
      ncCreationMode = NC_CLOBBER|NC_64BIT_DATA;
    else // use CDF-2 file format
      ncCreationMode = NC_CLOBBER|NC_64BIT_OFFSET;

    // verbose log ?
    bool pnetcdf_verbose = configMap.getBool("output","pnetcdf_verbose",false);

    int dimIds[3], varIds[nbvar];
    MPI_Offset starts[3], counts[3], write_size, sum_write_size;
    MPI_Info mpi_info_used;
    //char str[512];

    // time measurement variables
    double write_timing, max_write_timing, write_bw;

    /*
     * creation date
     */
    std::string stringDate;
    int stringDateSize;
    if (myRank==0) {
      stringDate = current_date();
      stringDateSize = stringDate.size();
    }
    // broadcast stringDate size to all other MPI tasks
    params.communicator->bcast(&stringDateSize, 1, MpiComm::INT, 0);
    
    // broadcast stringDate to all other MPI task
    if (myRank != 0) stringDate.reserve(stringDateSize);
    char* cstr = const_cast<char*>(stringDate.c_str());
    params.communicator->bcast(cstr, stringDateSize, MpiComm::CHAR, 0);

    /*
     * get MPI coords corresponding to MPI rank iPiece
     */
    int coords[3];
    if (dimType == TWO_D) {
      params.communicator->getCoords(myRank,2,coords);
    } else {
      params.communicator->getCoords(myRank,3,coords);
    }
    
    /*
     * make filename string
     */
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << iStep;
    std::string baseName       = outputPrefix+"_"+outNum.str();
    std::string ncFilename     = outputPrefix+"_"+outNum.str()+".nc";
    std::string ncFilenameFull = outputDir+"/"+ncFilename;

    // copy device data to host
    Kokkos::deep_copy(Uhost, Udata);

    // here we need to check Uhost memory layout
    KokkosLayout layout;
    if (std::is_same<typename DataArray::array_layout, Kokkos::LayoutLeft>::value)
      layout = KOKKOS_LAYOUT_LEFT;
    else
      layout = KOKKOS_LAYOUT_RIGHT;

    
    // measure time ??
    if (pnetcdf_verbose) {
      MPI_Barrier(params.communicator->getComm());
      write_timing = MPI_Wtime();
    }

    /* 
     * Create NetCDF file
     */
    err = ncmpi_create(params.communicator->getComm(), ncFilenameFull.c_str(), 
		       ncCreationMode,
                       MPI_INFO_NULL, &ncFileId);
    if (err != NC_NOERR) {
      printf("Error: ncmpi_create() file %s (%s)\n",ncFilenameFull.c_str(),ncmpi_strerror(err));
      MPI_Abort(params.communicator->getComm(), -1);
      exit(1);
    }
    
    /*
     * Define dimensions
     */
    int gsizes[3];
    if (dimType == TWO_D) {
      gsizes[1] = mx*nx+2*ghostWidth;
      gsizes[0] = my*ny+2*ghostWidth;
      
      err = ncmpi_def_dim(ncFileId, "x", gsizes[0], &dimIds[0]);
      PNETCDF_HANDLE_ERROR;
      
      err = ncmpi_def_dim(ncFileId, "y", gsizes[1], &dimIds[1]);
      PNETCDF_HANDLE_ERROR;
    
    } else { 
      gsizes[2] = mx*nx+2*ghostWidth;
      gsizes[1] = my*ny+2*ghostWidth;
      gsizes[0] = mz*nz+2*ghostWidth;
      
      err = ncmpi_def_dim(ncFileId, "x", gsizes[0], &dimIds[0]);
      PNETCDF_HANDLE_ERROR;
      
      err = ncmpi_def_dim(ncFileId, "y", gsizes[1], &dimIds[1]);
      PNETCDF_HANDLE_ERROR;

      err = ncmpi_def_dim(ncFileId, "z", gsizes[2], &dimIds[2]);
      PNETCDF_HANDLE_ERROR;
    }
    
    /* 
     * Define variables
     */
    nc_type ncDataType;
    MPI_Datatype mpiDataType;

    if (sizeof(real_t) == sizeof(float)) {
      ncDataType  = NC_FLOAT;
      mpiDataType = MPI_FLOAT;
    } else {
      ncDataType  = NC_DOUBLE;
      mpiDataType = MPI_DOUBLE;
    }
    
    if (dimType==TWO_D) {
      err = ncmpi_def_var(ncFileId, "rho", ncDataType, 2, dimIds, &varIds[ID]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "E", ncDataType, 2, dimIds, &varIds[IP]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "rho_vx", ncDataType, 2, dimIds, &varIds[IU]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "rho_vy", ncDataType, 2, dimIds, &varIds[IV]);
      PNETCDF_HANDLE_ERROR;

      if (mhdEnabled) {
	err = ncmpi_def_var(ncFileId, "rho_vz", ncDataType, 2, dimIds, &varIds[IW]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_def_var(ncFileId, "Bx", ncDataType, 2, dimIds, &varIds[IA]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_def_var(ncFileId, "By", ncDataType, 2, dimIds, &varIds[IB]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_def_var(ncFileId, "Bz", ncDataType, 2, dimIds, &varIds[IC]);
	PNETCDF_HANDLE_ERROR;
      }
      
    } else { // THREE_D

      err = ncmpi_def_var(ncFileId, "rho", ncDataType, 3, dimIds, &varIds[ID]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "E", ncDataType, 3, dimIds, &varIds[IP]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "rho_vx", ncDataType, 3, dimIds, &varIds[IU]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "rho_vy", ncDataType, 3, dimIds, &varIds[IV]);
      PNETCDF_HANDLE_ERROR;
      err = ncmpi_def_var(ncFileId, "rho_vz", ncDataType, 3, dimIds, &varIds[IW]);
      PNETCDF_HANDLE_ERROR;

      if (mhdEnabled) {
	err = ncmpi_def_var(ncFileId, "Bx", ncDataType, 3, dimIds, &varIds[IA]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_def_var(ncFileId, "By", ncDataType, 3, dimIds, &varIds[IB]);
	PNETCDF_HANDLE_ERROR;
	err = ncmpi_def_var(ncFileId, "Bz", ncDataType, 3, dimIds, &varIds[IC]);
	PNETCDF_HANDLE_ERROR;
      }

    } // end THREE_D

    /*
     * global attributes
     */
    // did we use CDF-2 or CDF-5
    {
      int useCDF5_int = useCDF5 ? 1 : 0;
      err = ncmpi_put_att_int(ncFileId, NC_GLOBAL, "CDF-5 mode", NC_INT, 1, &useCDF5_int);
      PNETCDF_HANDLE_ERROR;
    }
    
    // write time step number
    err = ncmpi_put_att_int(ncFileId, NC_GLOBAL, "time step", NC_INT, 1, &iStep);
    PNETCDF_HANDLE_ERROR;

    // write total time
    {
      double timeValue = (double) totalTime;
      err = ncmpi_put_att_double(ncFileId, NC_GLOBAL, "total time", NC_DOUBLE, 1, &timeValue);
      PNETCDF_HANDLE_ERROR;
    }

    // for information MPI config used
    {
      std::ostringstream mpiConf;
      mpiConf << "MPI configuration used to write file: "
	      << "mx,my,mz="
	      << mx << "," << my << "," << mz << " "
	      << "nx,ny,nz="
	      << nx << "," << ny << "," << nz;

      err = ncmpi_put_att_text(ncFileId, NC_GLOBAL, "MPI conf", mpiConf.str().size(), mpiConf.str().c_str());
      PNETCDF_HANDLE_ERROR;	    
    }

    /*
     * write creation date
     */
    {
      err = ncmpi_put_att_text(ncFileId, NC_GLOBAL, "creation date", stringDateSize, stringDate.c_str());
      PNETCDF_HANDLE_ERROR;	    
    }

    /* 
     * exit the define mode 
     */
    err = ncmpi_enddef(ncFileId);
    PNETCDF_HANDLE_ERROR;

    /* 
     * Get all the MPI_IO hints used
     */
    err = ncmpi_get_file_info(ncFileId, &mpi_info_used);
    PNETCDF_HANDLE_ERROR;

    /*
     * Write heavy data (take care of row-major / column major format !)
     */
    // use non-overlapped domain
    if (dimType == TWO_D) {
      
      counts[IY] = nx;
      counts[IX] = ny;
      
      starts[IY] = coords[IX]*nx;
      starts[IX] = coords[IY]*ny;
      
      // take care of borders along X
      if (coords[IX]==mx-1) {
	counts[IY] += 2*ghostWidth;
      }
      
      // take care of borders along Y
      if (coords[IY]==my-1) {
	counts[IX] += 2*ghostWidth;
      }
      
    } else { // THREE_D
      
      counts[IZ] = nx;
      counts[IY] = ny;
      counts[IX] = nz;
      
      starts[IZ] = coords[IX]*nx;
      starts[IY] = coords[IY]*ny;
      starts[IX] = coords[IZ]*nz;
      
      // take care of borders along X
      if (coords[IX]==mx-1) {
	counts[IZ] += 2*ghostWidth;
      }
      // take care of borders along Y
      if (coords[IY]==my-1) {
	counts[IY] += 2*ghostWidth;
      }
      // take care of borders along Z
      if (coords[IZ]==mz-1) {
	counts[IX] += 2*ghostWidth;
      }

    } // end THREE_D
    

    int nItems = counts[IX]*counts[IY];
    if (dimType==THREE_D)
      nItems *= counts[IZ];

    { // data need to be allocated and copied from U
      
      real_t* data;
      
      data = (real_t *) malloc(nItems*sizeof(real_t));
      
      int iStop=nx, jStop=ny, kStop=nz;

      if (coords[IX]== mx-1) iStop=nx+2*ghostWidth;
      if (coords[IY]== my-1) jStop=ny+2*ghostWidth;
      if (coords[IZ]== mz-1) kStop=nz+2*ghostWidth;

      for (int iVar=0; iVar<nbvar; iVar++) {
	
	// copy needed data into data !
	copy_buffer(data,iStop,jStop,kStop,iVar,layout);

	// write on disk
	err = ncmpi_put_vara_all(ncFileId, varIds[iVar], starts, counts, data, nItems, mpiDataType);
	PNETCDF_HANDLE_ERROR;
	
      } // end for iVar

      free(data);
      
    } // end non-overlap mode
    
    /* 
     * close the file 
     */
    err = ncmpi_close(ncFileId);
    PNETCDF_HANDLE_ERROR;

    // verbose log about memory bandwidth
    if (pnetcdf_verbose) {

      write_timing = MPI_Wtime() - write_timing;

      write_size = nbvar * isize * jsize * ksize * sizeof(real_t);
      //write_size = nbvar * U.section() * sizeof(real_t);
      //write_size = U.sizeBytes();
      sum_write_size = write_size *  params.nProcs;
      
      MPI_Reduce(&write_timing, &max_write_timing, 1, MPI_DOUBLE, MPI_MAX, 0, params.communicator->getComm());

      if (myRank==0) {
	printf("########################################################\n");
	printf("############## Parallel-netCDF bandwidth ###############\n");
	printf("########################################################\n");
	printf("Local  array size %d x %d x %d reals(%zu bytes), write size = %.2f MB\n",
	       nx+2*ghostWidth,
	       ny+2*ghostWidth,
	       nz+2*ghostWidth,
	       sizeof(real_t),
	       1.0*write_size/1048576.0);
	sum_write_size /= 1048576.0;
	printf("Global array size %d x %d x %d reals(%zu bytes), write size = %.2f GB\n",
	       mx*nx+2*ghostWidth,
	       my*ny+2*ghostWidth,
	       mz*nz+2*ghostWidth,
	       sizeof(real_t),
	       1.0*sum_write_size/1024);
	
	write_bw = sum_write_size/max_write_timing;
	printf(" procs    Global array size  exec(sec)  write(MB/s)\n");
	printf("-------  ------------------  ---------  -----------\n");
	printf(" %4d    %4d x %4d x %4d %8.2f  %10.2f\n", params.nProcs,
	       mx*nx+2*ghostWidth,
	       my*ny+2*ghostWidth,
	       mz*nz+2*ghostWidth,
	       max_write_timing, write_bw);
	printf("########################################################\n");
      } // end (myRank == 0)

    } // pnetcdf_verbose
    
    /*
     * Print MPI information
     */
    bool pnetcdf_print_mpi_info = configMap.getBool("output","pnetcdf_print_mpi_info",false);
    if (pnetcdf_print_mpi_info and myRank==0) {
      
      int     flag;
      char    info_cb_nodes[64], info_cb_buffer_size[64];
      char    info_striping_factor[64], info_striping_unit[64];

      char undefined_char[]="undefined";

      strcpy(info_cb_nodes,        undefined_char);
      strcpy(info_cb_buffer_size,  undefined_char);
      strcpy(info_striping_factor, undefined_char);
      strcpy(info_striping_unit,   undefined_char);
      
      char cb_nodes_char[]       ="cb_nodes";
      char cb_buffer_size_char[] ="cb_buffer_size";
      char striping_factor_char[]="striping_factor";
      char striping_unit_char[]  ="striping_unit";

      MPI_Info_get(mpi_info_used, cb_nodes_char       , 64, info_cb_nodes, &flag);
      MPI_Info_get(mpi_info_used, cb_buffer_size_char , 64, info_cb_buffer_size, &flag);
      MPI_Info_get(mpi_info_used, striping_factor_char, 64, info_striping_factor, &flag);
      MPI_Info_get(mpi_info_used, striping_unit_char  , 64, info_striping_unit, &flag);
      
      printf("MPI hint: cb_nodes        = %s\n", info_cb_nodes);
      printf("MPI hint: cb_buffer_size  = %s\n", info_cb_buffer_size);
      printf("MPI hint: striping_factor = %s\n", info_striping_factor);
      printf("MPI hint: striping_unit   = %s\n", info_striping_unit);
      
    } // pnetcdf_print_mpi_info

  } // save

  
  DataArray     Udata;
  DataArrayHost Uhost;
  HydroParams& params;
  ConfigMap& configMap;
  int nbvar;
  const std::map<int, std::string>& variables_names;
  int iStep;
  real_t totalTime;
  std::string debug_name;
  
}; // class Save_PNETCDF

} // namespace io

} // namespace euler_kokkos

#endif // IO_PNETCDF_H_
