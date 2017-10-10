#include "utils.h"

#include <ctime>   // for std::time_t, std::tm, std::localtime
#include <sstream> // string stream
#include <string>  // string
#include <iomanip> // for std::put_time
#include <iostream>

// =======================================================
// =======================================================
void print_current_date(std::ostream& stream)
{
  
  /* get current time */
  std::time_t     now = std::time(nullptr); 
  
  /* Format and print the time, "ddd yyyy-mm-dd hh:mm:ss zzz" */
  std::tm tm = *std::localtime(&now);
  
  // old versions of g++ don't have std::put_time,
  // so we provide a slight work arround
#if defined(__GNUC__) && (__GNUC__ < 5)

  char foo[64];

  if(0 < std::strftime(foo, sizeof(foo), "%Y-%m-%d %H:%M:%S %Z", &tm)) 
    stream << "-- " << foo << "\n";

#else

  std::stringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S %Z");

  const std::string tmp = ss.str();
  //const char *cstr = tmp.c_str();

  stream << "-- " << tmp << "\n";

#endif

} // print_current_date

// =======================================================
// =======================================================
std::string get_current_date()
{
  
  /* get current time */
  std::time_t     now = std::time(nullptr); 
  
  /* Format and print the time, "ddd yyyy-mm-dd hh:mm:ss zzz" */
  std::tm tm = *std::localtime(&now);
  
  // old versions of g++ don't have std::put_time,
  // so we provide a slight work arround
#if defined(__GNUC__) && (__GNUC__ < 5)

  char foo[64];

  if(0 < std::strftime(foo, sizeof(foo), "%Y-%m-%d %H:%M:%S %Z", &tm)) 
    return std::string(foo);
  else
    return std::string("undefined");
    
#else

  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S %Z");

  return oss.str();
  
#endif

} // get_current_date
