#ifndef IO_READ_WRITE_H_
#define IO_READ_WRITE_H_

#include <map>
#include <string>

#include <shared/kokkos_shared.h>
//class HydroParams;
//class ConfigMap;
#include <shared/HydroParams.h>
#include <utils/config/ConfigMap.h>

#include "IO_ReadWriteBase.h"

namespace euler_kokkos { namespace io {

/**
 * 
 */
class IO_ReadWrite : public IO_ReadWriteBase {

public:
  IO_ReadWrite(HydroParams& params,
	       ConfigMap& configMap,
	       std::map<int, std::string>& variables_names);
  
  //! destructor
  virtual ~IO_ReadWrite() {};

  //! hydro parameters
  HydroParams& params;

  //! configuration file reader
  ConfigMap& configMap;

  //! override base class method
  virtual void save_data(DataArray2d             Udata,
			 DataArray2d::HostMirror Uhost,
			 int iStep,
			 real_t time,
			 std::string debug_name);

  //! override base class method
  virtual void save_data(DataArray3d             Udata,
			 DataArray3d::HostMirror Uhost,
			 int iStep,
			 real_t time,
			 std::string debug_name);

  //! public interface to save data.
  virtual void save_data_impl(DataArray2d             Udata,
			      DataArray2d::HostMirror Uhost,
			      int iStep,
			      real_t time,
			      std::string debug_name);
  
  virtual void save_data_impl(DataArray3d             Udata,
			      DataArray3d::HostMirror Uhost,
			      int iStep,
			      real_t time,
			      std::string debug_name);
    
  //! override base class method
  virtual void load_data(DataArray2d             Udata,
			 DataArray2d::HostMirror Uhost,
			 int& iStep,
			 real_t& time);

  //! override base class method
  virtual void load_data(DataArray3d             Udata,
			 DataArray3d::HostMirror Uhost,
			 int& iStep,
			 real_t& time);

  //! public interface to load data.
  virtual void load_data_impl(DataArray2d             Udata,
			      DataArray2d::HostMirror Uhost,
			      int& iStep,
			      real_t& time);
  
  virtual void load_data_impl(DataArray3d             Udata,
			      DataArray3d::HostMirror Uhost,
			      int& iStep,
			      real_t& time);
    
  //! names of variables to load/save (inherited from Solver)
  std::map<int, std::string>& variables_names;

  bool vtk_enabled;
  bool hdf5_enabled;
  bool pnetcdf_enabled;
  
}; // class IO_ReadWrite

} // namespace io

} // namespace euler_kokkos

#endif // IO_READ_WRITE_H_
