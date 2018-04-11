#ifndef IO_READ_WRITE_BASE_H_
#define IO_READ_WRITE_BASE_H_

#include <shared/kokkos_shared.h>
#include <string>

namespace euler_kokkos { namespace io {

/**
 * Base class exposing the public interface methods load_data / save_data.
 *
 * \note should this class be templated by DataArray ?
 */
class IO_ReadWriteBase {

public:
  IO_ReadWriteBase() {};
  virtual ~IO_ReadWriteBase() {};

  virtual void save_data(DataArray2d             Udata,
			 DataArray2d::HostMirror Uhost,
			 int iStep,
			 real_t time,
			 std::string debug_name) {};

  virtual void save_data(DataArray3d             Udata,
			 DataArray3d::HostMirror Uhost,
			 int iStep,
			 real_t time,
			 std::string debug_name) {};

  virtual void load_data(DataArray2d             Udata,
			 DataArray2d::HostMirror Uhost,
			 int& iStep,
			 real_t& time) {};
  
  virtual void load_data(DataArray3d             Udata,
			 DataArray3d::HostMirror Uhost,
			 int& iStep,
			 real_t& time) {};

}; // class IO_ReadWriteBase

} // namespace io

} // namespace euler_kokkos

#endif // IO_READ_WRITE_BASE_H_
