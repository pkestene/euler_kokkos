#ifndef IO_WRITER_BASE_H_
#define IO_WRITER_BASE_H_

#include <shared/kokkos_shared.h>
#include <string>

namespace euler_kokkos { namespace io {

/**
 * Base class exposing the public interface method save_data.
 *
 * \note should this class be templated by DataArray ?
 */
class IO_WriterBase {

public:
  IO_WriterBase() {};
  virtual ~IO_WriterBase() {};

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

}; // class IO_WriterBase

} // namespace io

} // namespace euler_kokkos

#endif // IO_WRITER_BASE_H_
