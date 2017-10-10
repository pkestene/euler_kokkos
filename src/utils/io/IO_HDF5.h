#ifndef IO_HDF5_H_
#define IO_HDF5_H_

#include <iostream>
#include <type_traits>

// for HDF5 file format output
#ifdef USE_HDF5
#include <hdf5.h>

#define HDF5_MESG(mesg)				\
  std::cerr << "HDF5 :" << mesg << std::endl;

#define HDF5_CHECK(val, mesg) do {				\
    if (val<0) {						\
      std::cerr << "*** HDF5 ERROR ***\n";			\
      std::cerr << "    HDF5_CHECK (" << mesg << ") failed with status : " \
		<< status << "\n";				\
    }								\
  } while(0)

#endif // USE_HDF5

#include <map>
#include <string>

#include <shared/kokkos_shared.h>
//class HydroParams;
//class ConfigMap;
#include "shared/HydroParams.h"
#include "shared/utils.h"
#include "utils/config/ConfigMap.h"

#ifdef USE_MPI
#include "utils/mpiUtils/MpiComm.h"
#endif // USE_MPI

#include "IO_common.h"

namespace euler_kokkos { namespace io {

// =======================================================
// =======================================================
/**
 * Write a wrapper file using the Xmdf file format (XML) to allow
 * Paraview/Visit to read these h5 files as a time series.
 *
 * \param[in] params a HydroParams struct (to retrieve geometry).
 * \param[in] totalNumberOfSteps The number of time steps computed.
 * \param[in] singleStep boolean; if true we only write header for
 *  the last step.
 * \param[in] ghostIncluded boolean; if true include ghost cells
 *
 * If library HDF5 is not available, do nothing.
 */
void writeXdmfForHdf5Wrapper(HydroParams& params,
			     ConfigMap& configMap,
			     int totalNumberOfSteps,
			     bool singleStep);

// =======================================================
// =======================================================
/**
 *
 */
template<DimensionType d>
class Save_HDF5
{
public:
  //! Decide at compile-time which data array type to use
  using DataArray  = typename std::conditional<d==TWO_D,DataArray2d,DataArray3d>::type;
  using DataArrayHost  = typename std::conditional<d==TWO_D,DataArray2dHost,DataArray3dHost>::type;

  Save_HDF5(DataArray     Udata,
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
  ~Save_HDF5() {};

  template<DimensionType d_ = d>
  void copy_buffer(typename std::enable_if<d_==TWO_D, real_t>::type *& data,
		   int isize, int jsize, int ksize, int nvar, KokkosLayout layout)
  {
    if (layout == KOKKOS_LAYOUT_RIGHT) { // transpose array to make data contiguous in memory
      for (int j=0; j<jsize; ++j) {
	for (int i=0; i<isize; ++i) {
	  int index = i+isize*j;
	  data[index]=Uhost(i,j,nvar);
	}
      }
    } else {
      data = Uhost.ptr_on_device() + isize*jsize*nvar;
    }

  } // copy_buffer

  template<DimensionType d_=d>
  void copy_buffer(typename std::enable_if<d_==THREE_D, real_t>::type *& data,
		   int isize, int jsize, int ksize, int nvar, KokkosLayout layout)
  {
    if (layout == KOKKOS_LAYOUT_RIGHT) { // transpose array to make data contiguous in memory
      for (int k=0; k<ksize; ++k) {
	for (int j=0; j<jsize; ++j) {
	  for (int i=0; i<isize; ++i) {
	    int index = i+isize*j+isize*jsize*k;
	    data[index]=Uhost(i,j,k,nvar);
	  }
	}
      }
      
    } else {
      data = Uhost.ptr_on_device() + isize*jsize*ksize*nvar;
    }

  } // copy_buffer / 3D
  
  // =======================================================
  // =======================================================
  /**
   * Dump computation results (conservative variables) into a file
   * (HDF5 file format) file extension is h5. File can be viewed by
   * hdfview; see also h5dump.
   *
   * \sa writeXdmfForHdf5Wrapper this routine write a Xdmf wrapper file for paraview.
   *
   * If library HDF5 is not available, do nothing.
   * \param[in] Udata device data to save
   * \param[in,out] Uhost host data temporary array before saving to file
   */
  void save()
  {

    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;

    const int ghostWidth = params.ghostWidth;

    const int dimType = params.dimType;

    const bool ghostIncluded = configMap.getBool("output","ghostIncluded",false);

    const bool mhdEnabled = params.mhdEnabled;
    
    // copy device data to host
    Kokkos::deep_copy(Uhost, Udata);

    // here we need to check Uhost memory layout
    KokkosLayout layout;
    if (Uhost.stride_0()==1 and Uhost.stride_0()!=1)
      layout = KOKKOS_LAYOUT_LEFT;
    if (Uhost.stride_0()>1)
      layout = KOKKOS_LAYOUT_RIGHT;
  
    herr_t status = 0;
    UNUSED(status);
    
    // make filename string
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << iStep;
    std::string baseName         = outputPrefix+"_"+outNum.str();
    std::string hdf5Filename     = baseName+".h5";
    std::string hdf5FilenameFull = outputDir+"/"+hdf5Filename;
   
    // data size actually written on disk
    int nxg = nx;
    int nyg = ny;
    int nzg = nz;
    if (ghostIncluded) {
      nxg += 2*ghostWidth;
      nyg += 2*ghostWidth;
      nzg += 2*ghostWidth;
    }

    /*
     * write HDF5 file
     */
    // Create a new file using default properties.
    hid_t file_id = H5Fcreate(hdf5FilenameFull.c_str(), H5F_ACC_TRUNC |  H5F_ACC_DEBUG, H5P_DEFAULT, H5P_DEFAULT);

    // Create the data space for the dataset in memory and in file.
    hsize_t  dims_memory[3];
    hsize_t  dims_file[3];
    hid_t dataspace_memory, dataspace_file;
    if (dimType == TWO_D) {
      dims_memory[0] = jsize;
      dims_memory[1] = isize;
      dims_file[0] = nyg;
      dims_file[1] = nxg;
      dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
      dataspace_file   = H5Screate_simple(2, dims_file  , NULL);
    } else {
      dims_memory[0] = ksize;
      dims_memory[1] = jsize;
      dims_memory[2] = isize;
      dims_file[0] = nzg;
      dims_file[1] = nyg;
      dims_file[2] = nxg;
      dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
      dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
    }

    // Create the datasets.
    hid_t dataType;
    if (sizeof(real_t) == sizeof(float))
      dataType = H5T_NATIVE_FLOAT;
    else
      dataType = H5T_NATIVE_DOUBLE;
    

    // select data with or without ghost zones
    if (ghostIncluded) {
      if (dimType == TWO_D) {
	hsize_t  start[2] = {0, 0}; // ghost zone width
	hsize_t stride[2] = {1, 1};
	hsize_t  count[2] = {(hsize_t) nyg, (hsize_t) nxg};
	hsize_t  block[2] = {1, 1}; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      } else {
	hsize_t  start[3] = {0, 0, 0}; // ghost zone width
	hsize_t stride[3] = {1, 1, 1};
	hsize_t  count[3] = {(hsize_t) nzg, (hsize_t) nyg, (hsize_t) nxg};
	hsize_t  block[3] = {1, 1, 1}; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      }      
    } else {
      if (dimType == TWO_D) {
	hsize_t  start[2] = {(hsize_t) ghostWidth, (hsize_t) ghostWidth}; // ghost zone width
	hsize_t stride[2] = {1, 1};
	hsize_t  count[2] = {(hsize_t) ny, (hsize_t) nx};
	hsize_t  block[2] = {1, 1}; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      } else {
	hsize_t  start[3] = {(hsize_t) ghostWidth, (hsize_t) ghostWidth, (hsize_t) ghostWidth}; // ghost zone width
	hsize_t stride[3] = {1, 1, 1};
	hsize_t  count[3] = {(hsize_t) nz, (hsize_t) ny, (hsize_t) nx};
	hsize_t  block[3] = {1, 1, 1}; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);      
      }
    }

    /*
     * property list for compression
     */
    // get compression level (0=no compression; 9 is highest level of compression)
    int compressionLevel = configMap.getInteger("output", "outputHdf5CompressionLevel", 0);
    if (compressionLevel < 0 or compressionLevel > 9) {
      std::cerr << "Invalid value for compression level; must be an integer between 0 and 9 !!!" << std::endl;
      std::cerr << "compression level is then set to default value 0; i.e. no compression !!" << std::endl;
      compressionLevel = 0;
    }

    hid_t propList_create_id = H5Pcreate(H5P_DATASET_CREATE);

    if (dimType == TWO_D) {
      const hsize_t chunk_size2D[2] = {(hsize_t) ny, (hsize_t) nx};
      status = H5Pset_chunk (propList_create_id, 2, chunk_size2D);
      HDF5_CHECK(status, "Can not set hdf5 chunck sizes");
    } else { // THREE_D
      const hsize_t chunk_size3D[3] = {(hsize_t) nz, (hsize_t) ny, (hsize_t) nx};
      status = H5Pset_chunk (propList_create_id, 3, chunk_size3D);
      HDF5_CHECK(status, "Can not set hdf5 chunck sizes");
    }
    H5Pset_shuffle (propList_create_id);
    H5Pset_deflate (propList_create_id, compressionLevel);
    
    /*
     * write heavy data to HDF5 file
     */
    real_t* data;
  
    // Some adjustement needed to take into account that strides / layout need
    // to be checked at runtime
    // if memory layout is KOKKOS_LAYOUT_RIGHT, we need an extra buffer.
    if (layout == KOKKOS_LAYOUT_RIGHT) {

      if (dimType == TWO_D)
	data = new real_t[isize*jsize];
      else
	data = new real_t[isize*jsize*ksize];

    }      
  
    // write density
    hid_t dataset_id = H5Dcreate2(file_id, "/density", dataType, dataspace_file, 
				  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    copy_buffer(data, isize, jsize, ksize, ID, layout);
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    
    // write total energy
    dataset_id = H5Dcreate2(file_id, "/energy", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    copy_buffer(data, isize, jsize, ksize, IP, layout);
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    
    // write momentum X
    dataset_id = H5Dcreate2(file_id, "/momentum_x", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    copy_buffer(data, isize, jsize, ksize, IU, layout);
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    
    // write momentum Y
    dataset_id = H5Dcreate2(file_id, "/momentum_y", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    copy_buffer(data, isize, jsize, ksize, IV, layout);
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
  
    // write momentum Z (only if 3D hydro)
    if (dimType == THREE_D and !mhdEnabled) {
      dataset_id = H5Dcreate2(file_id, "/momentum_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IW, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    }
    
    if (mhdEnabled) {
      // write momentum mz
      dataset_id = H5Dcreate2(file_id, "/momentum_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IW, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
      // write magnetic field components
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_x", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IA, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_y", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IB, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IC, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
    }

    // free memory if necessary
    if (layout == KOKKOS_LAYOUT_RIGHT) {
      delete[] data;
    }
  
    // write time step as an attribute to root group
    hid_t ds_id;
    hid_t attr_id;
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "time step", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &iStep);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }
    
    // write total time 
    {
      double timeValue = (double) totalTime;

      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "total time", H5T_NATIVE_DOUBLE, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &timeValue);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    // write geometry information (just to be consistent)
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "nx", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &nx);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }
    
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "ny", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &ny);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }
    
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "nz", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &nz);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    // write information about ghost zone
    {
      int tmpVal = ghostIncluded ? 1 : 0;
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "ghost zone included", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &tmpVal);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    // write date as an attribute to root group
    std::string dataString = current_date();
    const char *dataChar = dataString.c_str();
    hsize_t   dimsAttr[1] = {1};
    hid_t type = H5Tcopy (H5T_C_S1);
    status = H5Tset_size (type, H5T_VARIABLE);
    hid_t root_id = H5Gopen2(file_id, "/", H5P_DEFAULT);
    hid_t dataspace_id = H5Screate_simple(1, dimsAttr, NULL);
    attr_id = H5Acreate2(root_id, "creation date", type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, type, &dataChar);
    status = H5Aclose(attr_id);
    status = H5Gclose(root_id);
    status = H5Tclose(type);
    status = H5Sclose(dataspace_id);

    // close/release resources.
    H5Pclose(propList_create_id);
    H5Sclose(dataspace_memory);
    H5Sclose(dataspace_file);
    H5Dclose(dataset_id);
    H5Fflush(file_id, H5F_SCOPE_LOCAL);
    H5Fclose(file_id);

    //(void) status;
  
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
  
}; // class Save_HDF5

#ifdef USE_MPI
// =======================================================
// =======================================================
/**
 *
 */
template<DimensionType d>
class Save_HDF5_mpi
{
public:
  //! Decide at compile-time which data array type to use
  using DataArray  = typename std::conditional<d==TWO_D,DataArray2d,DataArray3d>::type;
  using DataArrayHost  = typename std::conditional<d==TWO_D,DataArray2dHost,DataArray3dHost>::type;

  Save_HDF5_mpi(DataArray     Udata,
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
  ~Save_HDF5_mpi() {};

  template<DimensionType d_ = d>
  void copy_buffer(typename std::enable_if<d_==TWO_D, real_t>::type *& data,
		   int isize, int jsize, int ksize, int nvar, KokkosLayout layout)
  {
    if (layout == KOKKOS_LAYOUT_RIGHT) { // transpose array to make data contiguous in memory
      for (int j=0; j<jsize; ++j) {
	for (int i=0; i<isize; ++i) {
	  int index = i+isize*j;
	  data[index]=Uhost(i,j,nvar);
	}
      }
    } else {
      data = Uhost.ptr_on_device() + isize*jsize*nvar;
    }

  } // copy_buffer

  template<DimensionType d_ = d>
  void copy_buffer(typename std::enable_if<d_==THREE_D, real_t>::type *& data,
		   int isize, int jsize, int ksize, int nvar, KokkosLayout layout)
  {
    if (layout == KOKKOS_LAYOUT_RIGHT) { // transpose array to make data contiguous in memory
      for (int k=0; k<ksize; ++k) {
	for (int j=0; j<jsize; ++j) {
	  for (int i=0; i<isize; ++i) {
	    int index = i+isize*j+isize*jsize*k;
	    data[index]=Uhost(i,j,k,nvar);
	  }
	}
      }
      
    } else {
      data = Uhost.ptr_on_device() + isize*jsize*ksize*nvar;
    }

  } // copy_buffer / 3D
  
  // =======================================================
  // =======================================================
  /**
   * Dump computation results (conservative variables) into a file
   * (HDF5 file format) file extension is h5. File can be viewed by
   * hdfview; see also h5dump.
   *
   * \sa writeXdmfForHdf5Wrapper this routine write a Xdmf wrapper file for paraview.
   *
   * If library HDF5 is not available, do nothing.
   * \param[in] Udata device data to save
   * \param[in,out] Uhost host data temporary array before saving to file
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

    // time measurement variables
    double write_timing, max_write_timing, write_bw;
    MPI_Offset write_size, sum_write_size;

    // verbose log ?
    bool hdf5_verbose = configMap.getBool("output","hdf5_verbose",false);

    // copy device data to host
    Kokkos::deep_copy(Uhost, Udata);

    // here we need to check Uhost memory layout
    KokkosLayout layout;
    if (Uhost.stride_0()==1 and Uhost.stride_0()!=1)
      layout = KOKKOS_LAYOUT_LEFT;
    if (Uhost.stride_0()>1)
      layout = KOKKOS_LAYOUT_RIGHT;
  
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

    herr_t status;
    (void) status;

     // make filename string
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << iStep;
    std::string baseName         = outputPrefix+"_"+outNum.str();
    std::string hdf5Filename     = outputPrefix+"_"+outNum.str()+".h5";
    std::string hdf5FilenameFull = outputDir+"/"+outputPrefix+"_"+outNum.str()+".h5";
   
    // measure time ??
    if (hdf5_verbose) {
      //MPI_Barrier(params.communicator->getComm());
      params.communicator->synchronize();
      write_timing = MPI_Wtime();
    }

    /*
     * write HDF5 file
     */
    // Create a new file using property list with parallel I/O access.
    MPI_Info mpi_info     = MPI_INFO_NULL;
    hid_t    propList_create_id = H5Pcreate(H5P_FILE_ACCESS);
    status = H5Pset_fapl_mpio(propList_create_id, params.communicator->getComm(), mpi_info);
    HDF5_CHECK(status, "Can not access MPI IO parameters");
    
    hid_t    file_id  = H5Fcreate(hdf5FilenameFull.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, propList_create_id);
    H5Pclose(propList_create_id);

    // Create the data space for the dataset in memory and in file.
    hsize_t  dims_file[3];
    hsize_t  dims_memory[3];
    hsize_t  dims_chunk[3];
    hid_t dataspace_memory;
    //hid_t dataspace_chunk;
    hid_t dataspace_file;

    /*
     * reassembleInFile is false
     */
    if (!reassembleInFile) {

      if (allghostIncluded or ghostIncluded) {
	
	if (dimType == TWO_D) {
	  
	  dims_file[0] = (ny+2*ghostWidth)*(mx*my);
	  dims_file[1] = (nx+2*ghostWidth);
	  dims_memory[0] = jsize;
	  dims_memory[1] = isize;
	  dims_chunk[0] = ny+2*ghostWidth;
	  dims_chunk[1] = nx+2*ghostWidth;
	  dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(2, dims_file  , NULL);

	} else { // THREE_D

	  dims_file[0] = (nz+2*ghostWidth)*(mx*my*mz);
	  dims_file[1] =  ny+2*ghostWidth;
	  dims_file[2] =  nx+2*ghostWidth;
	  dims_memory[0] = ksize; 
	  dims_memory[1] = jsize;
	  dims_memory[2] = isize;
	  dims_chunk[0] = nz+2*ghostWidth;
	  dims_chunk[1] = ny+2*ghostWidth;
	  dims_chunk[2] = nx+2*ghostWidth;
	  dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
	  
	} // end THREE_D
      
      } else { // no ghost zones are saved
	
	if (dimType == TWO_D) {
	  
	  dims_file[0] = (ny)*(mx*my);
	  dims_file[1] = nx;
	  dims_memory[0] = jsize; 
	  dims_memory[1] = isize;
	  dims_chunk[0] = ny;
	  dims_chunk[1] = nx;
	  dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(2, dims_file  , NULL);

	} else {

	  dims_file[0] = (nz)*(mx*my*mz);
	  dims_file[1] = ny;
	  dims_file[2] = nx;
	  dims_memory[0] = ksize; 
	  dims_memory[1] = jsize;
	  dims_memory[2] = isize;
	  dims_chunk[0] = nz;
	  dims_chunk[1] = ny;
	  dims_chunk[2] = nx;
	  dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
	  
	} // end THREE_D
	      
      } // end - no ghost zones are saved

    } else { 
      /*
       * reassembleInFile is true
       */

      if (allghostIncluded) {
	
	if (dimType == TWO_D) {
	  
	  dims_file[0] = my*(ny+2*ghostWidth);
	  dims_file[1] = mx*(nx+2*ghostWidth);
	  dims_memory[0] = jsize; 
	  dims_memory[1] = isize;
	  dims_chunk[0] = ny+2*ghostWidth;
	  dims_chunk[1] = nx+2*ghostWidth;
	  dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(2, dims_file  , NULL);

	} else {

	  dims_file[0] = mz*(nz+2*ghostWidth);
	  dims_file[1] = my*(ny+2*ghostWidth);
	  dims_file[2] = mx*(nx+2*ghostWidth);
	  dims_memory[0] = ksize; 
	  dims_memory[1] = jsize;
	  dims_memory[2] = isize;
	  dims_chunk[0] = nz+2*ghostWidth;
	  dims_chunk[1] = ny+2*ghostWidth;
	  dims_chunk[2] = nx+2*ghostWidth;
	  dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
	  
	}
	
      } else if (ghostIncluded) { // only external ghost zones
	
	if (dimType == TWO_D) {
	  
	  dims_file[0] = ny*my+2*ghostWidth;
	  dims_file[1] = nx*mx+2*ghostWidth;
	  dims_memory[0] = jsize; 
	  dims_memory[1] = isize;
	  dims_chunk[0] = ny+2*ghostWidth;
	  dims_chunk[1] = nx+2*ghostWidth;
	  dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(2, dims_file  , NULL);

	} else {

	  dims_file[0] = nz*mz+2*ghostWidth;
	  dims_file[1] = ny*my+2*ghostWidth;
	  dims_file[2] = nx*mx+2*ghostWidth;
	  dims_memory[0] = ksize;
	  dims_memory[1] = jsize;
	  dims_memory[2] = isize;
	  dims_chunk[0] = nz+2*ghostWidth;
	  dims_chunk[1] = ny+2*ghostWidth;
	  dims_chunk[2] = nx+2*ghostWidth;
	  dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(3, dims_file  , NULL);

	}
	
      } else { // no ghost zones are saved
      
	if (dimType == TWO_D) {

	  dims_file[0] = ny*my;
	  dims_file[1] = nx*mx;
	  dims_memory[0] = jsize;
	  dims_memory[1] = isize;
	  dims_chunk[0] = ny;
	  dims_chunk[1] = nx;
	  dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(2, dims_file  , NULL);

	} else {

	  dims_file[0] = nz*mz;
	  dims_file[1] = ny*my;
	  dims_file[2] = nx*mx;
	  dims_memory[0] = ksize;
	  dims_memory[1] = jsize;
	  dims_memory[2] = isize;
	  dims_chunk[0] = nz;
	  dims_chunk[1] = ny;
	  dims_chunk[2] = nx;
	  dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
	  dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
	  
	}
	
      } // end ghostIncluded / allghostIncluded

    } // end reassembleInFile is true

    // Create the chunked datasets.
    hid_t dataType;
    if (sizeof(real_t) == sizeof(float))
      dataType = H5T_NATIVE_FLOAT;
    else
      dataType = H5T_NATIVE_DOUBLE;
    
    /*
     * Memory space hyperslab :
     * select data with or without ghost zones
     */
    if (ghostIncluded or allghostIncluded) {
      
      if (dimType == TWO_D) {
	hsize_t  start[2] = { 0, 0 }; // no start offset
	hsize_t stride[2] = { 1, 1 };
	hsize_t  count[2] = { 1, 1 };
	hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      } else {
	hsize_t  start[3] = { 0, 0, 0 }; // no start offset
	hsize_t stride[3] = { 1, 1, 1 };
	hsize_t  count[3] = { 1, 1, 1 };
	hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);      
      }
      
    } else { // no ghost zones
      
      if (dimType == TWO_D) {
	hsize_t  start[2] = { (hsize_t) ghostWidth,  (hsize_t) ghostWidth }; // ghost zone width
	hsize_t stride[2] = {                    1,                     1 };
	hsize_t  count[2] = {                    1,                     1 };
	hsize_t  block[2] = {(hsize_t)           ny, (hsize_t)         nx }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
      } else {
	hsize_t  start[3] = { (hsize_t) ghostWidth,  (hsize_t) ghostWidth, (hsize_t) ghostWidth }; // ghost zone width
	hsize_t stride[3] = { 1,  1,  1 };
	hsize_t  count[3] = { 1,  1,  1 };
	hsize_t  block[3] = {(hsize_t) nz, (hsize_t) ny, (hsize_t) nx }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);      
      }
      
    } // end ghostIncluded or allghostIncluded
    
    /*
     * File space hyperslab :
     * select where we want to write our own piece of the global data
     * according to MPI rank.
     */

    /*
     * reassembleInFile is false
     */
    if (!reassembleInFile) {

      if (dimType == TWO_D) {
	
	hsize_t  start[2] = { myRank*dims_chunk[0], 0 };
	//hsize_t  start[2] = { 0, myRank*dims_chunk[1]};
	hsize_t stride[2] = { 1,  1 };
	hsize_t  count[2] = { 1,  1 };
	hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	
      } else { // THREE_D
	
	hsize_t  start[3] = { myRank*dims_chunk[0], 0, 0 };
	hsize_t stride[3] = { 1,  1,  1 };
	hsize_t  count[3] = { 1,  1,  1 };
	hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	
      } // end THREE_D -- allghostIncluded
      
    } else {

      /*
       * reassembleInFile is true
       */

      if (allghostIncluded) {
	
	if (dimType == TWO_D) {
	  
	  hsize_t  start[2] = { coords[1]*dims_chunk[0], coords[0]*dims_chunk[1]};
	  hsize_t stride[2] = { 1,  1 };
	  hsize_t  count[2] = { 1,  1 };
	  hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	  status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	  
	} else { // THREE_D
	  
	  hsize_t  start[3] = { coords[2]*dims_chunk[0], coords[1]*dims_chunk[1], coords[0]*dims_chunk[2]};
	  hsize_t stride[3] = { 1,  1,  1 };
	  hsize_t  count[3] = { 1,  1,  1 };
	  hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	  status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	  
	}
	
      } else if (ghostIncluded) {
	
	// global offsets
	int gOffsetStartX, gOffsetStartY, gOffsetStartZ;
	
	if (dimType == TWO_D) {
	  gOffsetStartY  = coords[1]*ny;
	  gOffsetStartX  = coords[0]*nx;
	  
	  hsize_t  start[2] = { (hsize_t) gOffsetStartY, (hsize_t) gOffsetStartX };
	  hsize_t stride[2] = { 1,  1 };
	  hsize_t  count[2] = { 1,  1 };
	  hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	  status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	  
	} else { // THREE_D
	  
	  gOffsetStartZ  = coords[2]*nz;
	  gOffsetStartY  = coords[1]*ny;
	  gOffsetStartX  = coords[0]*nx;
	  
	  hsize_t  start[3] = { (hsize_t) gOffsetStartZ, (hsize_t) gOffsetStartY, (hsize_t) gOffsetStartX };
	  hsize_t stride[3] = { 1,  1,  1 };
	  hsize_t  count[3] = { 1,  1,  1 };
	  hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	  status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	  
	}
	
      } else { // no ghost zones
	
	if (dimType == TWO_D) {
	  
	  hsize_t  start[2] = { coords[1]*dims_chunk[0], coords[0]*dims_chunk[1]};
	  hsize_t stride[2] = { 1,  1 };
	  hsize_t  count[2] = { 1,  1 };
	  hsize_t  block[2] = { dims_chunk[0], dims_chunk[1] }; // row-major instead of column-major here
	  status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	  
	} else { // THREE_D
	  
	  hsize_t  start[3] = { coords[2]*dims_chunk[0], coords[1]*dims_chunk[1], coords[0]*dims_chunk[2]};
	  hsize_t stride[3] = { 1,  1,  1 };
	  hsize_t  count[3] = { 1,  1,  1 };
	  hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
	  status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
	  
	} // end THREE_D
	
      } // end ghostIncluded / allghostIncluded

    } // end reassembleInFile is true
    
    /*
     *
     * write heavy data to HDF5 file
     *
     */
    real_t* data;

    // Some adjustement needed to take into account that strides / layout need
    // to be checked at runtime
    // if memory layout is KOKKOS_LAYOUT_RIGHT, we need an extra buffer.
    if (layout == KOKKOS_LAYOUT_RIGHT) {

      if (dimType == TWO_D)
	data = new real_t[isize*jsize];
      else
	data = new real_t[isize*jsize*ksize];

    }

    propList_create_id = H5Pcreate(H5P_DATASET_CREATE);
    if (dimType == TWO_D)
      H5Pset_chunk(propList_create_id, 2, dims_chunk);
    else
      H5Pset_chunk(propList_create_id, 3, dims_chunk);

    // please note that HDF5 parallel I/O does not support yet filters
    // so we can't use here H5P_deflate to perform compression !!!
    // Weak solution : call h5repack after the file is created
    // (performance of that has not been tested)

    // take care not to use parallel specific features if the HDF5
    // library available does not support them !!
    hid_t propList_xfer_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(propList_xfer_id, H5FD_MPIO_COLLECTIVE);

    hid_t dataset_id;

    /*
     * write density    
     */
    dataset_id = H5Dcreate2(file_id, "/density", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    copy_buffer(data, isize, jsize, ksize, ID, layout);
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
    H5Dclose(dataset_id);
    
    /*
     * write energy
     */
    dataset_id = H5Dcreate2(file_id, "/energy", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    copy_buffer(data, isize, jsize, ksize, IP, layout);
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
    H5Dclose(dataset_id);
    
    /*
     * write momentum X
     */
    dataset_id = H5Dcreate2(file_id, "/momentum_x", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    copy_buffer(data, isize, jsize, ksize, IU, layout);
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
    H5Dclose(dataset_id);
    
    /*
     * write momentum Y
     */
    dataset_id = H5Dcreate2(file_id, "/momentum_y", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    copy_buffer(data, isize, jsize, ksize, IV, layout);
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
    H5Dclose(dataset_id);
    
    /*
     * write momentum Z (only if 3D or MHD enabled)
     */
    if (dimType == THREE_D and !mhdEnabled) {
      dataset_id = H5Dcreate2(file_id, "/momentum_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IW, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
      H5Dclose(dataset_id);
    }
    
    if (mhdEnabled) {
      // write momentum z
      dataset_id = H5Dcreate2(file_id, "/momentum_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IW, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
      H5Dclose(dataset_id);

      // write magnetic field components
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_x", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IA, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
      H5Dclose(dataset_id);
      
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_y", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IB, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
      H5Dclose(dataset_id);
      
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IC, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
      H5Dclose(dataset_id);

    }

    // free memory if necessary
    if (layout == KOKKOS_LAYOUT_RIGHT) {
      delete[] data;
    }

    // write time step number
    hid_t ds_id   = H5Screate(H5S_SCALAR);
    hid_t attr_id;
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "time step", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &iStep);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    // write total time 
    {
      double timeValue = (double) totalTime;

      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "total time", H5T_NATIVE_DOUBLE, 
    				 ds_id,
    				 H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &timeValue);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    // write information about ghost zone
    {
      int tmpVal = ghostIncluded ? 1 : 0;
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "external ghost zones only included", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &tmpVal);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    {
      int tmpVal = allghostIncluded ? 1 : 0;
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "all ghost zones included", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &tmpVal);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    // write information about reassemble MPI pieces in file
    {
      int tmpVal = reassembleInFile ? 1 : 0;
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "reassemble MPI pieces in file", H5T_NATIVE_INT, 
			   ds_id,
			   H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &tmpVal);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    // write local geometry information (just to be consistent)
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "nx", H5T_NATIVE_INT, 
				 ds_id,
				 H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &nx);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }
    
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "ny", H5T_NATIVE_INT, 
				 ds_id,
				 H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &ny);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }
    
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "nz", H5T_NATIVE_INT, 
				 ds_id,
				 H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &nz);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }
    
    // write MPI topology sizes
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "mx", H5T_NATIVE_INT, 
				 ds_id,
				 H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &mx);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }
    
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "my", H5T_NATIVE_INT, 
				 ds_id,
				 H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &my);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }
    
    {
      ds_id   = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(file_id, "mz", H5T_NATIVE_INT, 
				 ds_id,
				 H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, H5T_NATIVE_INT, &mz);
      status = H5Sclose(ds_id);
      status = H5Aclose(attr_id);
    }

    /*
     * write creation date
     */
    {
      hsize_t   dimsAttr[1] = {1};
      hid_t memtype = H5Tcopy (H5T_C_S1);
      status = H5Tset_size (memtype, stringDateSize+1);
      hid_t root_id = H5Gopen(file_id, "/", H5P_DEFAULT);
      hid_t dataspace_id = H5Screate_simple(1, dimsAttr, NULL);
      attr_id = H5Acreate(root_id, "creation date", memtype, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Awrite(attr_id, memtype, cstr);
      status = H5Aclose(attr_id);
      status = H5Gclose(root_id);
      status = H5Tclose(memtype);
      status = H5Sclose(dataspace_id);
    }

    // close/release resources.
    H5Pclose(propList_create_id);
    H5Pclose(propList_xfer_id);
    H5Sclose(dataspace_memory);
    H5Sclose(dataspace_file);
    H5Fclose(file_id);


    // verbose log about memory bandwidth
    if (hdf5_verbose) {

      write_timing = MPI_Wtime() - write_timing;
      
      if (dimType == TWO_D)
	write_size = nbvar * isize * jsize * sizeof(real_t);
      else
	write_size = nbvar * isize * jsize * ksize * sizeof(real_t);
      //write_size = U.sizeBytes();
      sum_write_size = write_size *  params.nProcs;
      
      MPI_Reduce(&write_timing, &max_write_timing, 1, MPI_DOUBLE, MPI_MAX, 0, params.communicator->getComm());

      if (myRank==0) {
	printf("########################################################\n");
	printf("################### HDF5 bandwidth #####################\n");
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

    } // hdf5_verbose

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
  
}; // class Save_HDF5_mpi

#endif // USE_MPI

} // namespace io

} // namespace euler_kokkos

#endif // IO_HDF5_H_
