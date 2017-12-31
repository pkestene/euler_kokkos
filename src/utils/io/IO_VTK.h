#ifndef IO_VTK_H_
#define IO_VTK_H_

#include <map>
#include <string>

#include <shared/kokkos_shared.h>
struct HydroParams;
class ConfigMap;

namespace euler_kokkos { namespace io {

// ///////////////////////////////////////////////////////
// output routine (VTK file format, ASCII, VtkImageData)
// Take care that VTK uses row major (i+j*nx)
// To make sure OpenMP and CUDA version give the same
// results, we transpose the OpenMP data.
// ///////////////////////////////////////////////////////
/**
 * \param[in] Udata device data to save
 * \param[in,out] Uhost host data temporary array before saving to file
 */
void save_VTK_2D(DataArray2d             Udata,
		 DataArray2d::HostMirror Uhost,
		 HydroParams& params,
		 ConfigMap& configMap,
		 int nbvar,
		 const std::map<int, std::string>& variables_names,
		 int iStep,
		 std::string debug_name);

// ///////////////////////////////////////////////////////
// output routine (VTK file format, ASCII, VtkImageData)
// Take care that VTK uses row major (i+j*nx+k*nx*ny)
// To make sure OpenMP and CUDA version give the same
// results, we transpose the OpenMP data.
// ///////////////////////////////////////////////////////
void save_VTK_3D(DataArray3d             Udata,
		 DataArray3d::HostMirror Uhost,
		 HydroParams& params,
		 ConfigMap& configMap,
		 int nbvar,
		 const std::map<int, std::string>& variables_names,
		 int iStep,
		 std::string debug_name);


#ifdef USE_MPI
/**
 * \param[in] Udata device data to save
 * \param[in,out] Uhost host data temporary array before saving to file
 */
void save_VTK_2D_mpi(DataArray2d             Udata,
		     DataArray2d::HostMirror Uhost,
		     HydroParams& params,
		     ConfigMap& configMap,
		     int nbvar,
		     const std::map<int, std::string>& variables_names,
		     int iStep,
		     std::string debug_name);

/**
 * \param[in] Udata device data to save
 * \param[in,out] Uhost host data temporary array before saving to file
 */
void save_VTK_3D_mpi(DataArray3d             Udata,
		     DataArray3d::HostMirror Uhost,
		     HydroParams& params,
		     ConfigMap& configMap,
		     int nbvar,
		     const std::map<int, std::string>& variables_names,
		     int iStep,
		     std::string debug_name);

/**
 * Write Parallel VTI header. 
 * Must be done by a single MPI process.
 *
 */
void write_pvti_header(std::string headerFilename,
		       std::string outputPrefix,
		       HydroParams& params,
		       int nbvar,
		       const std::map<int, std::string>& varNames,
		       int iStep);
#endif // USE_MPI

} // namespace io

} // namespace euler_kokkos

#endif // IO_VTK_H_
