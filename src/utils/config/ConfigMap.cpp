/**
 * \file ConfigMap.cpp
 * \brief Implement ConfigMap, essentially a INIReader with additional get methods.
 *
 */
#include <shared/euler_kokkos_config.h> // for EULER_KOKKOS_USE_MPI
#include "ConfigMap.h"
#include <cstdlib> // for strtof
#include <sstream>
#include <fstream>

#ifdef EULER_KOKKOS_USE_MPI
#  include <mpi.h>
#endif

namespace euler_kokkos
{

// =======================================================
// =======================================================
ConfigMap::ConfigMap(std::string filename)
  : INIReader(filename)
{} // ConfigMap::ConfigMap

// =======================================================
// =======================================================
ConfigMap::ConfigMap(char *& buffer, int buffer_size)
  : INIReader(buffer, buffer_size)
{} // ConfigMap::ConfigMap

// =======================================================
// =======================================================
ConfigMap::~ConfigMap() {} // ConfigMap::~ConfigMap

// =======================================================
// =======================================================
float
ConfigMap::getFloat(std::string section, std::string name, float default_value) const
{
  std::string  valstr = getString(section, name, "");
  const char * value = valstr.c_str();
  char *       end;
  // This parses "1234" (decimal) and also "0x4D2" (hex)
  float valFloat = strtof(value, &end);
  return end > value ? valFloat : default_value;
} // ConfigMap::getFloat

// =======================================================
// =======================================================
void
ConfigMap::setFloat(std::string section, std::string name, float value)
{

  std::stringstream ss;
  ss << value;

  setString(section, name, ss.str());

} // ConfigMap::setFloat

// =======================================================
// =======================================================
bool
ConfigMap::getBool(std::string section, std::string name, bool default_value) const
{
  bool        val = default_value;
  std::string valstr = getString(section, name, "");

  if (!valstr.compare("1") or !valstr.compare("yes") or !valstr.compare("true") or
      !valstr.compare("on"))
    val = true;
  if (!valstr.compare("0") or !valstr.compare("no") or !valstr.compare("false") or
      !valstr.compare("off"))
    val = false;

  // if valstr is empty, return the default value
  if (!valstr.size())
    val = default_value;

  return val;

} // ConfigMap::getBool

// =======================================================
// =======================================================
void
ConfigMap::setBool(std::string section, std::string name, bool value)
{

  if (value)
    setString(section, name, "true");
  else
    setString(section, name, "false");

} // ConfigMap::setBool

// =======================================================
// =======================================================
ConfigMap
broadcast_parameters(std::string filename)
{
#ifdef EULER_KOKKOS_USE_MPI

  int myRank;
  int nTasks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nTasks);

  char * buffer = nullptr;
  int    buffer_size = 0;

  // MPI rank 0 reads parameter file
  if (myRank == 0)
  {

    // open file and go to the end to get file size in bytes
    std::ifstream filein(filename.c_str(), std::ifstream::ate);
    if (filein.is_open())
    {
      int file_size = filein.tellg();

      filein.seekg(0); // rewind

      buffer_size = file_size;
      buffer = new char[buffer_size];

      filein.read(buffer, buffer_size);
    }
    else
    {
      printf(
        "Error : can't read input parameter file \"%s\"\nPlease check file actually exists !\n",
        filename.c_str());
    }
  }

  // broadcast buffer size (collective)
  MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (buffer_size > 0)
  {

    // all other MPI task need to allocate buffer
    if (myRank > 0)
    {
      // printf("I'm rank %d allocating buffer of size %d\n",myRank,buffer_size);
      buffer = new char[buffer_size];
    }

    // broastcast buffer itself (collective)
    MPI_Bcast(&buffer[0], buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // now all MPI rank should have buffer filled, try to build a ConfigMap
    ConfigMap configMap(buffer, buffer_size);

    if (buffer)
      delete[] buffer;

    return configMap;
  }

  // return default configMap (surely not what we wanted)
  char *    null_buffer;
  ConfigMap defaultConfig(null_buffer, 0);

  return defaultConfig;

#else
  ConfigMap configMap(filename);
  return configMap;
#endif

} // broadcast_parameters

} // namespace euler_kokkos
