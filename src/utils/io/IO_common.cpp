#include <ctime>   // for std::time_t, std::tm, std::localtime
#include <sstream>
#include <iomanip> // for std::put_time (only g++ >= 5)

#include "IO_common.h"

namespace euler_kokkos { namespace io {

// =======================================================
// =======================================================
std::string current_date()
{
  
  /* get current time */
  std::time_t     now = std::time(nullptr);
  
  /* Format and print the time, "ddd yyyy-mm-dd hh:mm:ss zzz" */
  std::tm tm = *std::localtime(&now);
  
  // old versions of g++ don't have std::put_time,
  // so we provide a slight work arround
#if defined(__GNUC__) && (__GNUC__ < 5)
  
  char foo[64];
  
  std::strftime(foo, sizeof(foo), "%Y-%m-%d %H:%M:%S %Z", &tm);
  return std::string(foo);
  
#else
  
  std::stringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S %Z");
  
  return ss.str();

#endif

} // current_date

} // namespace io

} // namespace euler_kokkos
